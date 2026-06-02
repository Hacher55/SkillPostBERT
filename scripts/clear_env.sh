#!/usr/bin/env bash
#
# clear_env.sh — removes the SSE691NLP conda environment entirely.
#
# Use this to do a clean reinstall (e.g. to change Python version or fix a
# broken environment). After clearing, rerun setup_env to rebuild it.
#
# Knobs:
#   CONDA_ENV=my-env ./scripts/clear_env.sh
#
# Usage:
#   ./scripts/clear_env.sh
set -uo pipefail

cd "$(dirname "$0")/.."

ENV_NAME="${CONDA_ENV:-SSE691NLP}"

# ---- preflight: conda ----------------------------------------------------- #
if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found on PATH."
  exit 1
fi

# ---- check the environment exists ----------------------------------------- #
if ! conda env list | grep -qE "^\s*${ENV_NAME}[[:space:]/]"; then
  echo "Conda environment '$ENV_NAME' does not exist — nothing to remove."
  exit 0
fi

# ---- guard: refuse to remove the active environment ----------------------- #
if [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]]; then
  echo "ERROR: '$ENV_NAME' is currently active."
  echo "  Run 'conda deactivate' first, then rerun this script."
  exit 1
fi

echo "=================================================================="
echo " This will permanently remove the conda environment '$ENV_NAME'."
echo " All installed packages will be deleted."
echo ""
echo " Rerun ./scripts/setup_env.sh afterwards to rebuild it."
echo ""
echo " Press Ctrl-C within 8 seconds to abort ..."
echo "=================================================================="
sleep 8

echo ""
echo "Removing environment '$ENV_NAME' ..."
conda env remove -n "$ENV_NAME" -y

echo "Done. Run ./scripts/setup_env.sh to rebuild."
