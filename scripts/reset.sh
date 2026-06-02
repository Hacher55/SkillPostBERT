#!/usr/bin/env bash
# reset.sh — full clean slate. Removes all generated and downloaded artefacts:
#
#   data/raw/        downloaded Kaggle datasets
#   data/processed/  corpus, gold files
#   models/          fine-tuned checkpoints
#   results/         metrics JSON and figures
#
# After this you can rerun the entire pipeline from scratch via run_part1.sh.
#
# Usage:
#   ./scripts/reset.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=================================================================="
echo " SkillPostBERT — full reset"
echo ""
echo " This will permanently delete:"
echo "   data/raw/        downloaded Kaggle datasets"
echo "   data/processed/  corpus + gold annotation files"
echo "   models/          fine-tuned checkpoints"
echo "   results/         metrics JSON + figures"
echo ""
echo " Press Ctrl-C within 8 seconds to abort ..."
echo "=================================================================="
sleep 8

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

echo ""
echo "Reset complete. Run ./scripts/run_part1.sh to start fresh."
