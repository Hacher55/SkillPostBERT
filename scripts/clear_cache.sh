#!/usr/bin/env bash
# clear_cache.sh — removes downloaded raw data so it will be re-fetched on the
# next run. Does NOT touch processed data, trained models, or results.
#
# Usage:
#   ./scripts/clear_cache.sh
set -euo pipefail

cd "$(dirname "$0")/.."

RAW="data/raw"

if [[ ! -d "$RAW" ]]; then
    echo "data/raw/ does not exist — nothing to clear."
    exit 0
fi

if [[ -z "$(ls -A "$RAW" 2>/dev/null)" ]]; then
    echo "data/raw/ is already empty."
    exit 0
fi

echo "Clearing $RAW ..."
for item in "$RAW"/{*,.*}; do
    [[ "$(basename "$item")" == "." || "$(basename "$item")" == ".." ]] && continue
    [[ -e "$item" ]] || continue
    rm -rf "$item"
    echo "  removed $(basename "$item")"
done
echo "Done. Re-run ./scripts/run_part1.sh to re-download."
