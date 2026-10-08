#!/usr/bin/env bash
# Idempotent: a CPU-only ComfyUI for the pack's tests, outside the repo.
#   ~/.cache/comfyui-trainything/venv        python venv (torch CPU wheels)
#   ~/.cache/comfyui-trainything/ComfyUI     the ComfyUI checkout
#   ~/.cache/comfyui-trainything/pure-venv   Pillow + numpy + pytest only
#   ComfyUI/custom_nodes/comfyui-trainything -> this repo
set -euo pipefail
ROOT="${COMFY_HARNESS:-$HOME/.cache/comfyui-trainything}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-python3}"
mkdir -p "$ROOT"
if [ ! -x "$ROOT/venv/bin/python" ]; then
  "$PY" -m venv "$ROOT/venv"
fi
VPY="$ROOT/venv/bin/python"
"$VPY" -m pip install -q --upgrade pip
if ! "$VPY" -c "import torch" 2>/dev/null; then
  "$VPY" -m pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
if [ ! -d "$ROOT/ComfyUI/.git" ]; then
  git clone -q https://github.com/comfyanonymous/ComfyUI "$ROOT/ComfyUI"
fi
"$VPY" -m pip install -q -r "$ROOT/ComfyUI/requirements.txt"
"$VPY" -m pip install -q -r "$REPO/requirements.txt" pytest
LINK="$ROOT/ComfyUI/custom_nodes/comfyui-trainything"
if [ ! -L "$LINK" ]; then
  rm -rf "$LINK"
  ln -s "$REPO" "$LINK"
fi
# A second, bare venv: Pillow, numpy and pytest only (no torch, no ComfyUI,
# no segno), standing in for "the system python with only Pillow and numpy".
if [ ! -x "$ROOT/pure-venv/bin/python" ]; then
  "$PY" -m venv "$ROOT/pure-venv"
fi
"$ROOT/pure-venv/bin/python" -m pip install -q pillow numpy pytest
echo "harness ready: $ROOT"
