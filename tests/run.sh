#!/usr/bin/env bash
# Run the whole suite in the harness venv (ComfyUI checks included), then the
# same suite in the bare venv (Pillow + numpy only: the ComfyUI checks skip).
#   tests/run.sh            both
#   tests/run.sh -k walk    extra args go to pytest (both runs)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${COMFY_HARNESS:-$HOME/.cache/comfyui-trainything}"
if [ ! -x "$ROOT/venv/bin/python" ] || [ ! -x "$ROOT/pure-venv/bin/python" ]; then
  echo "no harness at $ROOT; run tests/harness/setup_comfy.sh first" >&2
  exit 2
fi
cd "$REPO"
echo "== harness venv (ComfyUI $(cd "$ROOT/ComfyUI" && "$ROOT/venv/bin/python" -c 'import comfyui_version; print(comfyui_version.__version__)'))"
"$ROOT/venv/bin/python" -m pytest -q -p no:cacheprovider "$@"
echo "== bare venv (Pillow + numpy only)"
"$ROOT/pure-venv/bin/python" -m pytest -q -p no:cacheprovider "$@"
