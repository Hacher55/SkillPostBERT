#!/usr/bin/env bash
#
# run_part1.sh — everything up to the manual annotation checkpoint.
#
#   download data -> preprocess -> train BERT -> export a gold template
#
# After this finishes you HAND-CORRECT data/processed/gold.conll, then run
# run_part2.sh. The correction step can't be automated — it's what makes the
# BERT-vs-baseline comparison meaningful rather than circular.
#
# Usage:
#   ./run_part1.sh
#
# Knobs (override inline, e.g.  MODEL_NAME=distilbert-base-uncased ./run_part1.sh):
set -euo pipefail

# Run from the repo root regardless of where this is invoked from.
cd "$(dirname "$0")"

# ---- configuration (single source of truth) ------------------------------ #
MODEL_NAME="${MODEL_NAME:-bert-base-uncased}"      # or distilbert-base-uncased
MODEL_DIR="${MODEL_DIR:-models/bert-skills-ner}"   # where the trained model lands
PREPROCESS_MAX="${PREPROCESS_MAX:-2000}"           # cap postings per discipline
GOLD_N="${GOLD_N:-60}"                             # gold sample size (~20/discipline)

echo "=================================================================="
echo " SkillPostBERT — pipeline part 1"
echo "   model:        $MODEL_NAME"
echo "   output dir:   $MODEL_DIR"
echo "   max/disc:     $PREPROCESS_MAX"
echo "   gold sample:  $GOLD_N"
echo "=================================================================="

# ---- preflight: required Python packages ---------------------------------- #
echo ""
echo "[preflight] checking dependencies ..."
if ! python -c "import torch, transformers, datasets, pandas, seqeval, evaluate, yaml" 2>/dev/null; then
  echo "  Missing dependencies. Install them first:"
  echo "      pip install -r requirements.txt"
  exit 1
fi
echo "  ok."

# ---- preflight: Kaggle credentials (warn only) ---------------------------- #
if [[ ! -f "$HOME/.kaggle/kaggle.json" && -z "${KAGGLE_USERNAME:-}" ]]; then
  echo "[preflight] WARNING: no Kaggle credentials found"
  echo "  (~/.kaggle/kaggle.json missing and KAGGLE_USERNAME unset)."
  echo "  download_data.py will fail without them — see its docstring."
fi

# ---- 1. download -------------------------------------------------------- #
echo ""
echo "[1/4] downloading datasets ..."
python -m src.download_data

# ---- 2. preprocess ------------------------------------------------------ #
echo ""
echo "[2/4] preprocessing (weak-labeling BIO tags) ..."
python -m src.preprocess --model "$MODEL_NAME" --max "$PREPROCESS_MAX"

# ---- 3. train ----------------------------------------------------------- #
echo ""
echo "[3/4] training ..."
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "  GPU detected — training will be fast."
else
  echo "  WARNING: no CUDA GPU detected. Fine-tuning $MODEL_NAME on CPU is slow"
  echo "  (potentially hours). Consider a Colab GPU, or MODEL_NAME=distilbert-base-uncased."
  echo "  Press Ctrl-C within 8 seconds to abort ..."
  sleep 8
fi
python -m src.train --model "$MODEL_NAME" --output-dir "$MODEL_DIR"

# ---- 4. export gold template -------------------------------------------- #
echo ""
echo "[4/4] exporting gold annotation template ..."
python -m src.evaluate --export-gold --n "$GOLD_N" --model-name "$MODEL_NAME"

echo ""
echo "=================================================================="
echo " Part 1 complete."
echo ""
echo " NEXT — hand-correct the gold labels:"
echo "   1. Open  data/processed/gold.conll  in a text editor."
echo "   2. Fix the second column (the BIO tag) on each line:"
echo "        - add skills the matcher missed   (O  -> B-<CAT>)"
echo "        - remove false positives          (B-<CAT> -> O)"
echo "        - fix wrong categories/boundaries"
echo "   3. Then run:  ./run_part2.sh"
echo "=================================================================="
