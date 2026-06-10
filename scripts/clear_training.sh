#!/usr/bin/env bash
# clear_training.sh — removes fine-tuned model checkpoints so training can be
# rerun. Does NOT touch raw data, processed data, or results.
#
# Usage:
#   ./scripts/clear_training.sh
set -euo pipefail

cd "$(dirname "$0")/.."

MODELS="models"

if [[ ! -d "$MODELS" ]]; then
    echo "models/ does not exist — nothing to clear."
    exit 0
fi

if [[ -z "$(ls -A "$MODELS" 2>/dev/null)" ]]; then
    echo "models/ is already empty."
    exit 0
fi

echo "Clearing $MODELS ..."
for item in "$MODELS"/*; do
    [[ -e "$item" ]] || continue
    rm -rf "$item"
    echo "  removed $(basename "$item")"
done
echo "Done. Re-run ./scripts/run_part1.sh to retrain."
