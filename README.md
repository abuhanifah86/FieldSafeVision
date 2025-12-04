# FieldSafe Vision

FieldSafe Vision is a FastAPI + React (Vite/TypeScript) application for capturing images, generating detailed safety narratives, and reviewing observations. It combines captioning with PPE and hazard detection to highlight safe/unsafe acts and conditions.

## Features
- Mobile-friendly observer UI for camera capture (front/back) or upload with metadata (email, site, location/GPS, tags, severity, notes).
- Backend captions with `Salesforce/blip-image-captioning-large` plus optional PPE detection (`qualcomm/PPE-Detection`) and hazard cues (`facebook/detr-resnet-50`).
- Safety-focused narratives: subject/action, context, color/mood, texture/housekeeping, PPE/hazard counts, and a summary of safety cues.
- Admin/supervisor UI: review entries, view images, update status/severity/assignee/notes/closure notes, delete entries, and summaries (daily/weekly/monthly).
- GPU/CPU auto-selection; model caching via `MODEL_CACHE_DIR`.

## Demo
[Watch the demo on YouTube](https://youtu.be/4tCt9sqMIY8)

## Why FieldSafe Vision?
- Turn every snapshot into an actionable safety story: risks, PPE compliance, housekeeping, lighting, and clutter cues are surfaced instantly so supervisors can coach faster and crews stay ahead of hazards.
- Observers love it: no login required—just snap or upload, add a few details, and receive a rich narrative on mobile (front/back camera) over HTTPS.
- Leaders trust it: the admin console shows status, severity, assignee, notes, and daily/weekly/monthly summaries so patterns are obvious and interventions are prioritized.
- Built for the field: AI captions tuned for safety context, GPU/CPU auto-selection, on-disk media handling to keep performance snappy, and straightforward env configuration with HTTPS readiness for mobile camera access.
- Ready to roll out: FastAPI + Vite/TypeScript, default admin seeded, add/remove admins in seconds, keep data on your infra, and start capturing value immediately.

## Requirements
- Python 3.10+
- Node 18+ (for the Vite frontend)
- Optional GPU for faster captioning/detection

## Backend setup
```bash
pip install -r backend/requirements.txt
```

Key environment variables (optional):
- `CAPTION_MODEL` (default: `Salesforce/blip-image-captioning-large`)
- `MODEL_CACHE_DIR` (path to cache model weights)
- `PPE_DETECTION_MODEL` (default: `qualcomm/PPE-Detection`)
- `PPE_DETECTION_ENABLED` (`true`/`false`)
- `HAZARD_DETECTION_MODEL` (default: `facebook/detr-resnet-50`)
- `HAZARD_DETECTION_ENABLED` (`true`/`false`)
- `HAZARD_THRESHOLD` (default: `0.4`)
- `CAPTION_DEVICE` (`auto|cpu|cuda|cuda:N`)
- `JWT_SECRET` (override default random secret)
- `DB_URL` (default: `sqlite:///./image_narrator.db`)
- `MEDIA_ROOT` (directory for storing uploaded images on disk; default: `./media`)

Media storage:
- Uploaded images are written to `MEDIA_ROOT` with unique filenames (e.g., `entry_<timestamp>_<rand>.jpg`).
- `/api/entries` now returns the file path metadata instead of loading image blobs, and `/api/entries/{id}/image` streams from disk.
- The `media/` directory is ignored by git; back it up or mount persistent storage in production.

Run backend:
```bash
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## Frontend setup
```bash
cd frontend
npm install
npm run dev -- --host
# or npm run build
```

Env for frontend:
- `VITE_API_BASE` (e.g., `https://<backend-host>:8000`)
- `HTTPS=true` plus `SSL_CRT_FILE`/`SSL_KEY_FILE` to serve Vite over HTTPS for mobile camera access.

## One-shot start
From repo root:
```bash
bash start.sh
```
This starts backend and frontend; if self-signed certs exist in `frontend/localhost-cert.pem` and `frontend/localhost-key.pem`, both services run over HTTPS.

## Roles
- Observer: submits images + metadata (email required). No login.
- Admin/Supervisor: login via `/admin`, review/manage entries, summaries, delete, and image viewing. Default admin is seeded: `soilens@soilens.com` / `admin@soilens.com` (password stored hashed).
- Admins can add/remove other admins from the Admin page; passwords are stored hashed.

## Notes
- SQLite migrations: the backend auto-adds missing columns for new metadata on startup.
- Models download on first use; set `MODEL_CACHE_DIR` and ensure one-time network access.
- Camera access on mobile requires HTTPS or localhost.

## Ignoring local artifacts
See `.gitignore` for local caches/logs/artifacts to exclude from GitHub.
