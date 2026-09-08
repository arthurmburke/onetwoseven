#!/usr/bin/env bash
set -euo pipefail

STUDIO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$STUDIO_DIR"

if [[ ! -x .venv/bin/python ]]; then
  echo "MLX Studio is not set up yet. Run ./setup.sh first."
  exit 1
fi

if [[ ! -f dist/client/index.html ]]; then
  echo "The GUI build is missing. Run ./setup.sh first."
  exit 1
fi

STUDIO_HOST="${MLX_STUDIO_HOST:-127.0.0.1}"
STUDIO_PORT="${MLX_STUDIO_PORT:-8111}"

if [[ "${MLX_STUDIO_NO_OPEN:-0}" != "1" ]] && command -v open >/dev/null 2>&1; then
  (sleep 1; open "http://${STUDIO_HOST}:${STUDIO_PORT}") &
fi

exec .venv/bin/python -m uvicorn mlx_studio.app:app --host "$STUDIO_HOST" --port "$STUDIO_PORT"
