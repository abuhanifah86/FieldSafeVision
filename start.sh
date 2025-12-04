#!/usr/bin/env bash
# Start backend (FastAPI + Uvicorn) and frontend (Vite) from one terminal.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults; override via env vars.

# HTTPS setup for mobile camera: if cert/key exist, enable HTTPS automatically for frontend.
SSL_CRT_FILE_DEFAULT="$ROOT_DIR/frontend/localhost-cert.pem"
SSL_KEY_FILE_DEFAULT="$ROOT_DIR/frontend/localhost-key.pem"
if [[ -z "${HTTPS:-}" ]] && [[ -f "$SSL_CRT_FILE_DEFAULT" && -f "$SSL_KEY_FILE_DEFAULT" ]]; then
  export HTTPS=true
  export SSL_CRT_FILE=${SSL_CRT_FILE:-$SSL_CRT_FILE_DEFAULT}
  export SSL_KEY_FILE=${SSL_KEY_FILE:-$SSL_KEY_FILE_DEFAULT}
fi

# Prefer project venv, then PATH.
if [[ -z "${PY_BIN:-}" ]]; then
  if [[ -x "$ROOT_DIR/testenv/bin/python" ]]; then
    PY_BIN="$ROOT_DIR/testenv/bin/python"
  else
    PY_BIN=$(command -v python || command -v python3)
  fi
fi

# Normalize cert paths to absolute to avoid double "frontend/frontend" resolution.
if [[ -n "${SSL_CRT_FILE:-}" && "${SSL_CRT_FILE:0:1}" != "/" ]]; then
  SSL_CRT_FILE="$ROOT_DIR/${SSL_CRT_FILE#./}"
fi
if [[ -n "${SSL_KEY_FILE:-}" && "${SSL_KEY_FILE:0:1}" != "/" ]]; then
  SSL_KEY_FILE="$ROOT_DIR/${SSL_KEY_FILE#./}"
fi

# If HTTPS cert/key exist, also start backend over HTTPS to avoid mixed-content errors from HTTPS frontends.
BACKEND_SSL_ARGS=""
if [[ -f "${SSL_CRT_FILE:-$SSL_CRT_FILE_DEFAULT}" && -f "${SSL_KEY_FILE:-$SSL_KEY_FILE_DEFAULT}" ]]; then
  BACKEND_SSL_ARGS="--ssl-certfile ${SSL_CRT_FILE:-$SSL_CRT_FILE_DEFAULT} --ssl-keyfile ${SSL_KEY_FILE:-$SSL_KEY_FILE_DEFAULT}"
fi
BACKEND_CMD=${BACKEND_CMD:-"$PY_BIN -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 $BACKEND_SSL_ARGS"}

VITE_API_BASE=${VITE_API_BASE:-"http://localhost:8000"}
# Vite HTTPS is controlled via env HTTPS/SSL_CRT_FILE/SSL_KEY_FILE in vite.config.ts; no --https flag.
FRONTEND_CMD=${FRONTEND_CMD:-"cd \"$ROOT_DIR/frontend\" && VITE_API_BASE=$VITE_API_BASE HTTPS=${HTTPS:-false} SSL_CRT_FILE=${SSL_CRT_FILE:-} SSL_KEY_FILE=${SSL_KEY_FILE:-} npm run dev -- --host"}

pids=()

cleanup() {
  echo "Stopping processes..."
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

echo "Starting backend: $BACKEND_CMD"
bash -c "$BACKEND_CMD" &
pids+=($!)

echo "Starting frontend: $FRONTEND_CMD"
bash -c "$FRONTEND_CMD" &
pids+=($!)

echo "Backend and frontend started. Press Ctrl+C to stop."
wait
