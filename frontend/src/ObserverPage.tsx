import React, { useEffect, useRef, useState } from "react";
import { getDefaultApiBase } from "./api";
import { useStoredState } from "./useStoredState";

type CaptionResponse = {
  narrative: string;
  caption?: string | null;
  stats?: Record<string, unknown>;
  device?: string | null;
  model?: string;
  entry_id?: number;
};

const ObserverPage: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const defaultApi = getDefaultApiBase();
  const [apiBase, setApiBase] = useStoredState<string>("observer.apiBase", defaultApi);
  const [email, setEmail] = useStoredState<string>("observer.email", "");
  const [site, setSite] = useStoredState<string>("observer.site", "");
  const [location, setLocation] = useStoredState<string>("observer.location", "");
  const [tags, setTags] = useStoredState<string>("observer.tags", "");
  const [severity, setSeverity] = useStoredState<string>("observer.severity", "minor");
  const [notes, setNotes] = useState("");
  const [cameraFacing, setCameraFacing] = useState<"user" | "environment">(
    "environment",
  );
  const [isStreaming, setIsStreaming] = useState(false);
  const [status, setStatus] = useState("Idle");
  const [response, setResponse] = useState<CaptionResponse | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    return () => stopStream();
  }, []);

  const startStream = async () => {
    try {
      if (
        window.isSecureContext === false &&
        !window.location.hostname.includes("localhost") &&
        !window.location.hostname.includes("127.0.0.1")
      ) {
        setStatus("Camera requires HTTPS on mobile (use https or localhost).");
        return;
      }
      stopStream();
      setStatus("Requesting camera…");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: cameraFacing },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setIsStreaming(true);
      setStatus("Camera ready");
    } catch (err) {
      console.error(err);
      const message =
        err instanceof Error && err.message.toLowerCase().includes("permission")
          ? "Camera permission denied. Allow camera access in browser settings."
          : "Unable to access camera. Ensure permissions are granted and the page is served over HTTPS.";
      setStatus(message);
    }
  };

  const stopStream = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setIsStreaming(false);
  };

  const captureFrame = async () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    setPreviewUrl(canvas.toDataURL("image/jpeg"));
    const blob: Blob | null = await new Promise((resolve) =>
      canvas.toBlob((b) => resolve(b), "image/jpeg", 0.9),
    );
    if (!blob) {
      setStatus("Failed to capture frame.");
      return;
    }
    await sendToBackend(blob, "capture.jpg");
  };

  const onFileSelected = async (file?: File) => {
    if (!file) return;
    setPreviewUrl(URL.createObjectURL(file));
    await sendToBackend(file, file.name);
  };

  const sendToBackend = async (data: Blob | File, filename: string) => {
    if (!email || !email.includes("@")) {
      setStatus("Please enter a valid email.");
      return;
    }
    const form = new FormData();
    form.append("email", email);
    if (site) form.append("site", site);
    if (location) form.append("location", location);
    if (tags) form.append("tags", tags);
    if (severity) form.append("severity", severity);
    if (notes) form.append("notes", notes);
    form.append("file", data, filename);
    setLoading(true);
    setStatus("Uploading and processing…");
    setResponse(null);
    try {
      const res = await fetch(`${apiBase}/api/caption`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        let errText = "";
        try {
          errText = await res.text();
        } catch (e) {
          errText = res.statusText;
        }
        throw new Error(errText || `Request failed with ${res.status}`);
      }
      const json = (await res.json()) as CaptionResponse;
      setResponse(json);
      setStatus("Narrative ready");
    } catch (err) {
      console.error(err);
      let msg =
        err instanceof Error ? err.message : "Failed to process image. Check backend connectivity.";
      if (msg.toLowerCase().includes("failed to fetch")) {
        msg =
          "Failed to reach backend. Verify backend URL, ensure it is running, and avoid HTTPS/HTTP mixed content.";
      }
      setStatus(msg);
    } finally {
      setLoading(false);
    }
  };

  const checkHealth = async () => {
    try {
      setStatus("Checking backend health…");
      const res = await fetch(`${apiBase}/api/health`);
      const json = await res.json();
      setStatus(`Backend OK (model: ${json.model}, device: ${json.device ?? "unknown"})`);
    } catch (err) {
      console.error(err);
      setStatus("Health check failed. Check backend URL / SSL trust.");
    }
  };

  return (
    <>
      <div className="card controls">
        <div className="upload">
          <label htmlFor="apiBase">Backend URL</label>
          <input
            id="apiBase"
            type="text"
            value={apiBase}
            onChange={(e) => setApiBase(e.target.value)}
            placeholder="https://your-backend:8000"
          />
          <div className="buttons">
            <button type="button" className="secondary" onClick={checkHealth}>
              Check backend
            </button>
          </div>
        </div>
        <div className="upload">
          <strong>Your email (identifier)</strong>
          <input
            type="email"
            placeholder="email (required)"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
          <input
            type="text"
            placeholder="site / unit (optional)"
            value={site}
            onChange={(e) => setSite(e.target.value)}
          />
          <input
            type="text"
            placeholder="location or GPS (optional)"
            value={location}
            onChange={(e) => setLocation(e.target.value)}
          />
          <input
            type="text"
            placeholder="tags (comma separated)"
            value={tags}
            onChange={(e) => setTags(e.target.value)}
          />
          <select value={severity} onChange={(e) => setSeverity(e.target.value)}>
            <option value="minor">Minor</option>
            <option value="major">Major</option>
            <option value="critical">Critical</option>
          </select>
          <textarea
            placeholder="Notes / observation details"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            rows={3}
            style={{ width: "100%", font: "inherit" }}
          />
          <div className="status">No login required; email is stored with each entry.</div>
        </div>
        <div className="buttons">
          <select
            value={cameraFacing}
            onChange={(e) =>
              setCameraFacing(e.target.value as "user" | "environment")
            }
          >
            <option value="environment">Back camera</option>
            <option value="user">Front camera</option>
          </select>
          <button onClick={startStream} disabled={isStreaming}>
            Start camera
          </button>
          <button onClick={stopStream} disabled={!isStreaming} className="secondary">
            Stop camera
          </button>
          <button onClick={captureFrame} disabled={!isStreaming || loading}>
            Capture & narrate
          </button>
        </div>
        <video ref={videoRef} autoPlay playsInline muted />
      </div>

      <div className="card upload">
        <strong>Upload from device</strong>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => onFileSelected(e.target.files?.[0])}
        />
      </div>

      {previewUrl && (
        <div className="card">
          <strong>Preview</strong>
          <img src={previewUrl} alt="Preview" />
        </div>
      )}

      {response && (
        <div className="card">
          <strong>Narrative</strong>
          <div className="response">{response.narrative}</div>
          {response.model && (
            <div className="status">
              Model: {response.model} | Device: {response.device}
            </div>
          )}
          {response.entry_id && (
            <div className="status">Saved entry ID: {response.entry_id}</div>
          )}
        </div>
      )}
      <div className="status">{status}</div>
    </>
  );
};

export default ObserverPage;
