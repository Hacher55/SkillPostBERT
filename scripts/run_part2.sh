#!/usr/bin/env bash
#
# run_part2.sh — after you've hand-corrected data/processed/gold.conll.
#
#   apply corrections -> evaluate BERT vs baseline on gold -> cross-discipline
#   comparison (the figures/tables for your paper)
#
# Usage:
#   ./scripts/run_part2.sh
#
# Must use the SAME MODEL_DIR you trained with in scripts/run_part1.sh.
set -euo pipefail

cd "$(dirname "$0")/.."

# ---- configuration (must match run_part1.sh) ----------------------------- #
MODEL_DIR="${MODEL_DIR:-models/bert-skills-ner}"
GOLD_CONLL="${GOLD_CONLL:-data/processed/gold.conll}"
GOLD_JSONL="${GOLD_JSONL:-data/processed/gold.jsonl}"

echo "=================================================================="
echo " SkillPostBERT — pipeline part 2"
echo "   model dir:  $MODEL_DIR"
echo "   gold:       $GOLD_CONLL"
echo "=================================================================="

# ---- preflight ----------------------------------------------------------- #
if ! python -c "import torch, transformers, seqeval, evaluate, pandas, matplotlib" 2>/dev/null; then
  echo "Missing dependencies. Run: pip install -r requirements.txt"
  exit 1
fi
if [[ ! -f "$GOLD_CONLL" ]]; then
  echo "ERROR: $GOLD_CONLL not found. Run ./scripts/run_part1.sh first."
  exit 1
fi
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: trained model not found at $MODEL_DIR. Run ./scripts/run_part1.sh first."
  exit 1
fi

# Sanity nudge: did you actually edit the gold file, or apply raw weak labels?
echo ""
echo "[reminder] make sure you HAND-CORRECTED $GOLD_CONLL before this step."
echo "  Applying the un-edited template just re-scores the matcher against"
echo "  itself. Press Ctrl-C within 5 seconds if you still need to annotate ..."
sleep 5

# ---- 1. fold corrections back in ---------------------------------------- #
echo ""
echo "[1/3] applying corrected labels ..."
python -m src.evaluate --apply-conll "$GOLD_CONLL"

# ---- 2. head-to-head evaluation ----------------------------------------- #
echo ""
echo "[2/3] evaluating BERT vs keyword baseline on gold ..."
python -m src.evaluate --gold "$GOLD_JSONL" --model "$MODEL_DIR"

# ---- 3. cross-discipline comparison ------------------------------------- #
echo ""
echo "[3/3] cross-discipline skill analysis (BERT predictions) ..."
python -m src.compare --source bert --model "$MODEL_DIR"

echo ""
echo "=================================================================="
echo " Part 2 complete. Results in results/:"
echo "   comparison.json                  BERT vs baseline metrics"
echo "   top_skills_by_discipline.{csv,png}"
echo "   category_mix.{csv,png}"
echo "   skill_heatmap.png"
echo "=================================================================="
