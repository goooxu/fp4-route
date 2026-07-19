#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT/hf_cache/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$ROOT/hf_cache/transformers}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/uv_cache}"
export VENV_DIR="${VENV_DIR:-$ROOT/venv}"

mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  uv venv "$VENV_DIR" --python python3
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[setup] Installing PyTorch..."
# Try CUDA wheels first; fall back to default
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  uv pip install torch --index-url https://download.pytorch.org/whl/cu128 \
    || uv pip install torch --index-url https://download.pytorch.org/whl/cu126 \
    || uv pip install torch
fi

echo "[setup] Installing project requirements..."
uv pip install -r requirements.txt

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY

echo "[setup] Done. Activate with: source $VENV_DIR/bin/activate"
