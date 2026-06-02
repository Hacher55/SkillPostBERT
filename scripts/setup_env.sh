#!/usr/bin/env bash
#
# setup_env.sh — one-shot environment setup.
#
#   1. Create (or verify) the SSE691NLP conda environment (Python 3.10)
#   2. Install all Python dependencies from requirements.txt
#   3. Detect hardware and swap in a CUDA-enabled PyTorch wheel automatically
#      (NVIDIA only; Apple Silicon MPS is included in the default wheel)
#
# After this script finishes, activate the environment and run the pipeline:
#   conda activate SSE691NLP
#   ./scripts/run_part1.sh
#
# Knobs (export before running, or inline):
#   CONDA_ENV=my-env  ./scripts/setup_env.sh
#   PYTHON_VER=3.11   ./scripts/setup_env.sh
#
# Usage:
#   ./scripts/setup_env.sh
set -uo pipefail

cd "$(dirname "$0")/.."

ENV_NAME="${CONDA_ENV:-SSE691NLP}"
PYTHON_VER="${PYTHON_VER:-3.10}"

echo "=================================================================="
echo " SkillPostBERT — environment setup"
echo "   conda env:    $ENV_NAME"
echo "   Python:       $PYTHON_VER"
echo "=================================================================="

# ---- preflight: conda ----------------------------------------------------- #
echo ""
echo "[preflight] checking for conda ..."
if ! command -v conda &>/dev/null; then
  echo "  ERROR: conda not found on PATH."
  echo "  Install Miniconda or Anaconda, open a new shell, then rerun."
  echo "  https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi
echo "  $(conda --version)"

# ---- 1. create / verify environment --------------------------------------- #
echo ""
echo "[1/3] setting up conda environment '$ENV_NAME' ..."

if conda env list | grep -qE "^\s*${ENV_NAME}[[:space:]/]"; then
  echo "  '$ENV_NAME' already exists — skipping creation."
else
  echo "  creating '$ENV_NAME' (Python $PYTHON_VER) ..."
  conda create -n "$ENV_NAME" python="$PYTHON_VER" -y
  echo "  created."
fi

# ---- 2. install Python dependencies --------------------------------------- #
echo ""
echo "[2/3] installing Python dependencies ..."
echo "  (this may take a few minutes on the first install)"
conda run -n "$ENV_NAME" pip install -r requirements.txt
echo "  dependencies installed."

# ---- 3. hardware-specific PyTorch ----------------------------------------- #
echo ""
echo "[3/3] checking hardware for GPU-accelerated PyTorch ..."

# Apple Silicon — MPS is already in the default wheel, no swap needed
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  echo "  Apple Silicon detected — MPS is included in the default PyTorch wheel."

# NVIDIA GPU — detect CUDA driver version and install the matching wheel
elif command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
  CUDA_VER=$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  echo "  GPU:  $GPU_NAME"
  echo "  CUDA: $CUDA_VER (driver)"

  MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
  MINOR=$(echo "$CUDA_VER" | cut -d. -f2)

  if [[ "$MAJOR" -gt 12 ]] || [[ "$MAJOR" -eq 12 && "$MINOR" -ge 8 ]]; then
    WHEEL="cu128"
  elif [[ "$MAJOR" -eq 12 && "$MINOR" -ge 1 ]]; then
    WHEEL="cu121"
  elif [[ "$MAJOR" -eq 11 && "$MINOR" -ge 8 ]]; then
    WHEEL="cu118"
  else
    WHEEL=""
    echo "  CUDA $CUDA_VER is older than 11.8 — no matching wheel available."
    echo "  Keeping CPU PyTorch."
  fi

  if [[ -n "$WHEEL" ]]; then
    echo "  Installing CUDA-enabled PyTorch ($WHEEL) ..."
    conda run -n "$ENV_NAME" pip install torch torchvision torchaudio \
      --index-url "https://download.pytorch.org/whl/$WHEEL"
    echo "  CUDA PyTorch installed ($WHEEL)."
  fi

else
  echo "  No GPU detected — CPU PyTorch is sufficient."
fi

echo ""
echo "=================================================================="
echo " Setup complete."
echo ""
echo " Next steps:"
echo "   1. conda activate $ENV_NAME"
echo "   2. Place kaggle.json at ~/.kaggle/kaggle.json  (see README Setup)"
echo "   3. ./scripts/run_part1.sh"
echo ""
echo " Tip: run ./scripts/check_env.sh (after activating) to verify."
echo "=================================================================="
