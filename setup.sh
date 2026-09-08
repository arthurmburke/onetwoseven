#!/usr/bin/env bash
set -euo pipefail

STUDIO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$STUDIO_DIR"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "MLX Studio requires an Apple Silicon Mac."
  exit 1
fi

PYTHON_BIN="$(brew --prefix python@3.12)/libexec/bin/python3" 
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .

if command -v node >/dev/null 2>&1 && command -v pnpm >/dev/null 2>&1; then
  pnpm install
  pnpm build
elif command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
  npm install
  npm run build
elif [[ -f dist/client/index.html ]]; then
  echo "Using the included production GUI build."
else
  echo "Install Node.js 22+ with pnpm (or Corepack), then run this setup again."
  exit 1
fi

echo "Setup complete. Start MLX Studio with ./run.sh"
