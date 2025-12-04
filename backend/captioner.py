"""
Caption and describe images using a reusable pipeline with GPU/CPU auto-selection.
"""
from __future__ import annotations

import io
import os
import pathlib
from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:  # Optional heavy deps; handled gracefully if missing.
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover - environment dependent
    torch = None  # type: ignore
    _HAS_TORCH = False

try:  # pragma: no cover - environment dependent
    import cv2  # type: ignore
    CV2_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - environment dependent
    cv2 = None  # type: ignore
    CV2_IMPORT_ERROR = exc

try:  # pragma: no cover - environment dependent
    from transformers import AutoModelForImageTextToText, AutoProcessor, pipeline

    _HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover - environment dependent
    AutoModelForImageTextToText = None  # type: ignore
    AutoProcessor = None  # type: ignore
    pipeline = None  # type: ignore
    _HAS_TRANSFORMERS = False


DEFAULT_MODEL = "Salesforce/blip-image-captioning-large"
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR")
PPE_DETECTION_MODEL = os.getenv("PPE_DETECTION_MODEL", "qualcomm/PPE-Detection")
PPE_DETECTION_ENABLED = os.getenv("PPE_DETECTION_ENABLED", "true").lower() == "true"
HAZARD_DETECTION_MODEL = os.getenv("HAZARD_DETECTION_MODEL", "facebook/detr-resnet-50")
HAZARD_DETECTION_ENABLED = os.getenv("HAZARD_DETECTION_ENABLED", "true").lower() == "true"
HAZARD_THRESHOLD = float(os.getenv("HAZARD_THRESHOLD", "0.4"))
_PIPELINE = None
_PIPELINE_DEVICE: Optional["torch.device"] = None
_PPE_PIPELINE = None
_PPE_DEVICE: Optional["torch.device"] = None
_HAZARD_PIPELINE = None
_HAZARD_DEVICE: Optional["torch.device"] = None


def resolve_ml_device(pref: str = "auto") -> Optional["torch.device"]:
    """
    Resolve the preferred ML device to a torch.device, if possible.
    Returns None if torch is unavailable or no device is needed.
    """
    if not _HAS_TORCH or torch is None:
        return None

    pref = pref.lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    if pref == "cpu":
        return torch.device("cpu")
    if pref.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        return torch.device(pref)
    raise RuntimeError(f"Unknown ml-device '{pref}'. Use auto|cpu|cuda|cuda:N.")


def _require_cv2():
    if cv2 is None:
        raise RuntimeError(
            "OpenCV (cv2) is not available. Install it with `pip install opencv-python` "
            f"to enable image analysis ({CV2_IMPORT_ERROR})."
        )
    return cv2


def _load_pipeline(model: str, device: Optional["torch.device"]):
    global _PIPELINE, _PIPELINE_DEVICE
    if _PIPELINE is not None and _PIPELINE_DEVICE == device:
        return _PIPELINE
    if not _HAS_TRANSFORMERS or pipeline is None:
        raise RuntimeError("transformers is not installed; ML captioning unavailable.")

    pipe = None
    if AutoModelForImageTextToText and AutoProcessor:
        try:
            processor = AutoProcessor.from_pretrained(
                model, use_fast=True, cache_dir=MODEL_CACHE_DIR, trust_remote_code=True
            )
            model_obj = AutoModelForImageTextToText.from_pretrained(
                model, cache_dir=MODEL_CACHE_DIR, trust_remote_code=True
            )
            pipe = pipeline(
                "image-to-text",
                model=model_obj,
                image_processor=processor,
                device=device,
                trust_remote_code=True,
            )
        except Exception:
            pipe = pipeline("image-to-text", model=model, device=device)
    else:
        pipe = pipeline(
            "image-to-text",
            model=model,
            device=device,
            trust_remote_code=True,
            model_kwargs={"cache_dir": MODEL_CACHE_DIR},
        )

    _PIPELINE = pipe
    _PIPELINE_DEVICE = device
    return pipe


def _load_ppe_pipeline(model: str, device: Optional["torch.device"]):
    global _PPE_PIPELINE, _PPE_DEVICE
    if _PPE_PIPELINE is not None and _PPE_DEVICE == device:
        return _PPE_PIPELINE
    if not _HAS_TRANSFORMERS or pipeline is None:
        raise RuntimeError("transformers is not installed; PPE detection unavailable.")
    ppe_pipe = pipeline(
        "object-detection",
        model=model,
        device=device,
        model_kwargs={"cache_dir": MODEL_CACHE_DIR},
        trust_remote_code=True,
    )
    _PPE_PIPELINE = ppe_pipe
    _PPE_DEVICE = device
    return ppe_pipe


def _load_hazard_pipeline(model: str, device: Optional["torch.device"]):
    global _HAZARD_PIPELINE, _HAZARD_DEVICE
    if _HAZARD_PIPELINE is not None and _HAZARD_DEVICE == device:
        return _HAZARD_PIPELINE
    if not _HAS_TRANSFORMERS or pipeline is None:
        raise RuntimeError("transformers is not installed; hazard detection unavailable.")
    hz_pipe = pipeline(
        "object-detection",
        model=model,
        device=device,
        model_kwargs={"cache_dir": MODEL_CACHE_DIR},
        trust_remote_code=True,
    )
    _HAZARD_PIPELINE = hz_pipe
    _HAZARD_DEVICE = device
    return hz_pipe


def run_ppe_detection(image: Image.Image, device: Optional["torch.device"], threshold: float = 0.4):
    if not PPE_DETECTION_ENABLED:
        return []
    try:
        ppe_pipe = _load_ppe_pipeline(PPE_DETECTION_MODEL, device)
    except Exception:
        return []
    try:
        results = ppe_pipe(image)
        detections = []
        for r in results:
            score = r.get("score", 0.0)
            if score < threshold:
                continue
            detections.append(
                {
                    "label": r.get("label"),
                    "score": float(score),
                    "box": r.get("box"),
                }
            )
        return detections
    except Exception:
        return []


def run_hazard_detection(image: Image.Image, device: Optional["torch.device"], threshold: float = HAZARD_THRESHOLD):
    if not HAZARD_DETECTION_ENABLED:
        return []
    try:
        hz_pipe = _load_hazard_pipeline(HAZARD_DETECTION_MODEL, device)
    except Exception:
        return []
    try:
        results = hz_pipe(image)
        detections = []
        for r in results:
            score = r.get("score", 0.0)
            if score < threshold:
                continue
            detections.append(
                {
                    "label": r.get("label"),
                    "score": float(score),
                    "box": r.get("box"),
                }
            )
        return detections
    except Exception:
        return []


def describe_with_model(image: Image.Image, model: str, device: Optional["torch.device"]) -> str:
    pipe = _load_pipeline(model, device)
    result = pipe(image)
    if result and "generated_text" in result[0]:
        return result[0]["generated_text"]
    return "Model returned no caption."


def analyze_image_stats(image: Image.Image) -> dict:
    cv2_local = _require_cv2()
    img = image.convert("RGB")
    arr = np.asarray(img)
    brightness = float(arr.mean())
    contrast = float(arr.std())
    color_means = arr.reshape(-1, 3).mean(axis=0)
    dominant_channel = ["red", "green", "blue"][int(np.argmax(color_means))]

    edges = cv2_local.Canny(arr, 60, 180)
    edge_density = float(edges.mean() / 255)

    hsv = cv2_local.cvtColor(arr, cv2_local.COLOR_RGB2HSV)
    saturation = float(hsv[:, :, 1].mean())

    h, w, _ = arr.shape
    center = arr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    corners = np.concatenate(
        [
            arr[: h // 4, : w // 4].reshape(-1, 3),
            arr[: h // 4, 3 * w // 4 :].reshape(-1, 3),
            arr[3 * h // 4 :, : w // 4].reshape(-1, 3),
            arr[3 * h // 4 :, 3 * w // 4 :].reshape(-1, 3),
        ],
        axis=0,
    )
    center_brightness = float(center.mean())
    corner_brightness = float(corners.mean())

    if brightness < 60:
        mood = "very dark"
    elif brightness < 100:
        mood = "dim"
    elif brightness < 170:
        mood = "balanced"
    else:
        mood = "bright"

    if edge_density < 0.02:
        detail = "very simple with few shapes"
    elif edge_density < 0.08:
        detail = "has some distinct edges"
    else:
        detail = "quite busy with many edges and textures"

    light_tendency = (
        "center is brighter than edges"
        if center_brightness - corner_brightness > 10
        else "edges brighter than center"
        if corner_brightness - center_brightness > 10
        else "light is fairly even"
    )

    return {
        "brightness": brightness,
        "contrast": contrast,
        "dominant_channel": dominant_channel,
        "edge_density": edge_density,
        "saturation": saturation,
        "mood": mood,
        "detail": detail,
        "lighting_pattern": light_tendency,
    }


def format_narrative(caption: Optional[str], stats: dict) -> str:
    """
    Build a structured 4–6 sentence narrative that emphasizes detailed safety-related observations:
    1. Identify the main subject and primary actions, including any visible work behaviors, body posture, PPE usage (correct or incorrect), equipment handling, and interaction with the environment.
    2. Describe the surrounding setting and operational context, using visual cues such as lighting, shadows, edge density, equipment layout, confined spaces, elevated work areas, or proximity to hazards (e.g., pressure lines, rotating machinery, vehicles).
    3. Capture color palette and scene mood to help infer visibility, environmental conditions, or potential risks (e.g., low visibility, glare, weather-related hazards).
    4. Highlight textures and fine details, including condition of PPE, tools, surfaces, cables, spill presence, trip hazards, corrosion, leaks, signage clarity, housekeeping, and equipment integrity.
    5. Provide an overall safety assessment, noting clear indicators of safe or unsafe acts and conditions, including PPE compliance, ergonomics, risk potential, and any observable deviations from standard field safety practices.
    6. If possible, quantify observations with simple metrics, such as number of workers, number of PPE items detected/missing, or count of visible hazards.
    """
    parts = []

    if caption:
        parts.append(f"Subject and activity: {caption}.")
    else:
        parts.append("A subject is present, but the model did not provide a specific label.")

    detail = stats.get("detail", "has some distinct edges")
    lighting_pattern = stats.get("lighting_pattern", "light is fairly even")
    mood = stats.get("mood", "balanced")
    edge_density = stats.get("edge_density", 0.0)
    if edge_density < 0.03:
        setting = "The background looks simple with large smooth regions"
    elif edge_density < 0.08:
        setting = "The background shows some structure and defined shapes"
    else:
        setting = "The background appears busy with many edges and textures"
    parts.append(
        f"{setting}; lighting suggests it is {mood} with {lighting_pattern}, informing visibility of tools, equipment, and nearby hazards."
    )

    dom_color = stats.get("dominant_channel", "neutral")
    saturation = stats.get("saturation", 0.0)
    parts.append(
        f"Color and mood: palette leans toward {dom_color} with saturation around {saturation:.1f}, shaping visibility and possible glare/low-light conditions."
    )

    parts.append(
        f"Textures and fine details: surface detail {detail}, hinting at condition of PPE, tools, surfaces, or cables, and how tidy or cluttered the workspace is."
    )

    detections = stats.get("ppe_detections") or []
    ppe_summary = ""
    if detections:
        labels = [d.get("label", "item") for d in detections]
        counts = {}
        for lbl in labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        ppe_summary = "; ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        parts.append(f"PPE and safety cues: detected {ppe_summary}.")
    else:
        parts.append("PPE and safety cues: none explicitly detected; visually confirm PPE compliance and safe posture.")

    hazards = stats.get("hazard_detections") or []
    if hazards:
        hz_labels = [h.get("label", "item") for h in hazards]
        hz_counts = {}
        for lbl in hz_labels:
            hz_counts[lbl] = hz_counts.get(lbl, 0) + 1
        hazard_summary = "; ".join(f"{k}: {v}" for k, v in sorted(hz_counts.items()))
        parts.append(f"Hazard cues: detected {hazard_summary}; assess proximity and controls.")
    else:
        parts.append("Hazard cues: none explicitly detected; check for nearby vehicles, machinery, trip hazards, or elevated work areas.")

    people_count = stats.get("people_count")
    if people_count is not None:
        parts.append(f"People observed: {people_count}.")

    brightness = stats.get("brightness", 0.0)
    contrast = stats.get("contrast", 0.0)
    edge_density = stats.get("edge_density", 0.0)
    parts.append(
        f"Overall safety take: brightness {brightness:.1f}, contrast {contrast:.1f}, edge density {edge_density:.3f}; check for clear ergonomics, PPE compliance, and proximity to potential hazards."
    )

    return " ".join(parts)


def process_image_bytes(
    data: bytes,
    model: str = DEFAULT_MODEL,
    device_pref: str = "auto",
) -> Tuple[str, dict, Optional[str]]:
    """
    Generate a narrative for the image bytes.
    Returns (narrative, stats, caption) where caption may be None if ML unavailable.
    """
    image = Image.open(io.BytesIO(data))
    device = resolve_ml_device(device_pref)
    caption: Optional[str] = None
    try:
        caption = describe_with_model(image, model=model, device=device)
    except Exception as exc:
        caption = f"ML captioning unavailable ({exc})."

    stats = analyze_image_stats(image)
    detections = run_ppe_detection(image, device)
    if detections:
        stats["ppe_detections"] = detections
    hazard_dets = run_hazard_detection(image, device)
    if hazard_dets:
        stats["hazard_detections"] = hazard_dets
        people = [d for d in hazard_dets if d.get("label", "").lower() == "person"]
        stats["people_count"] = len(people)
    narrative = format_narrative(caption, stats)
    return narrative, stats, caption
