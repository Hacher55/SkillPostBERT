#!/usr/bin/env bash
# reset.sh — full clean slate. Removes all generated and downloaded artifacts
# and the conda environment:
#
#   data/raw/        downloaded Kaggle datasets
#   data/processed/  corpus, gold files
#   models/          fine-tuned checkpoints
#   results/         metrics JSON and figures
#   conda env        SSE691NLP (or $CONDA_ENV)
#
# After this, rebuild the environment and rerun the pipeline:
#   ./scripts/setup_env.sh
#   conda activate SSE691NLP
#   ./scripts/run_part1.sh
#
# Knobs:
#   CONDA_ENV=my-env ./scripts/reset.sh
#
# Usage:
#   ./scripts/reset.sh
set -uo pipefail

cd "$(dirname "$0")/.."

ENV_NAME="${CONDA_ENV:-SSE691NLP}"

echo "=================================================================="
echo " SkillPostBERT — full reset"
echo ""
echo " This will permanently delete:"
echo "   data/raw/        downloaded Kaggle datasets"
echo "   data/processed/  corpus + gold annotation files"
echo "   models/          fine-tuned checkpoints"
echo "   results/         metrics JSON + figures"
echo "   conda env        $ENV_NAME"
echo ""
echo " Press Ctrl-C within 8 seconds to abort ..."
echo "=================================================================="
sleep 8

# ---- data and artifact directories ---------------------------------------- #
clear_dir() {
    local path="$1"
    local label="$2"
    if [[ ! -d "$path" ]]; then
        echo "  $path/ — not found, skipping."
        return
    fi
    local count
    count=$(find "$path" -mindepth 1 -maxdepth 1 | wc -l)
    if [[ "$count" -eq 0 ]]; then
        echo "  $path/ — already empty."
        return
    fi
    echo "  Clearing $path/ ($label) ..."
    find "$path" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    echo "    done ($count item(s) removed)."
}

clear_dir "data/raw"       "raw data"
clear_dir "data/processed" "processed data"
clear_dir "models"         "model checkpoints"
clear_dir "results"        "results"

# ---- conda environment ---------------------------------------------------- #
echo ""
if ! command -v conda &>/dev/null; then
    echo "  conda not found — skipping environment removal."
elif ! conda env list | grep -qE "^\s*${ENV_NAME}[[:space:]/]"; then
    echo "  conda env '$ENV_NAME' — not found, skipping."
elif [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]]; then
    echo "  conda env '$ENV_NAME' — currently active, skipping."
    echo "    Run 'conda deactivate' then ./scripts/clear_env.sh to remove it."
else
    echo "  Removing conda environment '$ENV_NAME' ..."
    if conda env remove -n "$ENV_NAME" -y; then
        echo "    done."
    else
        echo "    failed — remove manually with: conda env remove -n $ENV_NAME"
    fi
fi

echo ""
echo "Reset complete."
echo "Run ./scripts/setup_env.sh to rebuild the environment, then ./scripts/run_part1.sh to start fresh."
