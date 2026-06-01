# SkillPostBERT

Fine-tuning BERT for skill extraction from engineering job postings, with a
cross-discipline comparison of skill demand across **mechanical**, **electrical**,
and **software** engineering.

NLP Applications course project (ECE/SSE/CYS 691).

## What it does

1. **Extracts** skill mentions from raw job-posting text using a BERT model
   fine-tuned for Named Entity Recognition (NER).
2. **Classifies** each skill into a taxonomy: Technical, Tools, Soft Skills,
   Certifications.
3. **Compares** skill demand across the three engineering disciplines and
   visualizes the differences.
4. **Baseline:** a rule-based keyword matcher, to demonstrate the value added
   by the learned model. Both are scored on Precision / Recall / F1 per category.

## Pipeline

```
raw postings ─► preprocess ─► [ BERT NER  ]  ─► skills ─► classify ─► compare
                          └─► [ keyword   ]  ─► skills ─┘            └─► charts
                                baseline
```

## Models

- **Primary:** `bert-base-uncased`, fine-tuned for token classification (NER).
- **Lighter alternative:** `distilbert-base-uncased` for compute-constrained runs.
- **Baseline:** dictionary keyword matching (`src/baseline.py`).

## Data

- LinkedIn Job Postings Dataset (Kaggle)
- Indeed Job Scrape Dataset (Kaggle)

Place raw CSVs under `data/raw/`. They are gitignored — see `data/raw/README.md`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running the pipeline

The pipeline has a manual annotation checkpoint in the middle (you hand-correct
a small gold set so the BERT-vs-baseline comparison is honest rather than
circular), so it's split into two scripts:

```bash
./run_part1.sh        # download -> preprocess -> train -> export gold template
# ... hand-correct data/processed/gold.conll ...
./run_part2.sh        # apply corrections -> evaluate -> cross-discipline charts
```

Knobs are environment overrides, e.g. for the lighter model:

```bash
MODEL_NAME=distilbert-base-uncased MODEL_DIR=models/distilbert-skills-ner ./run_part1.sh
```

Or run the steps individually — see each module's docstring.

## Layout

```
src/
  taxonomy.py      skill dictionary + BIO label scheme
  baseline.py      keyword matcher (comparison baseline + weak labeler)
  download_data.py Kaggle dataset fetcher
  preprocess.py    CSV -> cleaned, discipline-tagged, weak-labeled corpus
  model.py         dataset loading + model factory
  train.py         fine-tuning (HF Trainer, seqeval metrics)
  evaluate.py      BERT vs baseline on a hand-corrected gold set
  compare.py       cross-discipline skill analysis + charts
  utils.py         shared lightweight helpers
configs/           YAML run configs
data/              raw (gitignored) + processed
results/           metrics + figures
```

## End-to-end

```bash
# 1. data (run on your machine — needs your Kaggle token)
python src/download_data.py
python src/preprocess.py --max 2000          # drop --max for the full corpus

# 2. train (use a GPU; DistilBERT or Colab if you have no local GPU)
python src/train.py                          # bert-base
python src/train.py --model distilbert-base-uncased \
                    --output-dir models/distilbert-skills-ner

# 3. fair evaluation vs the baseline (build a gold set once)
python src/evaluate.py --export-gold --n 60  # writes gold.jsonl + gold.conll
#   ... hand-correct gold.conll in a text editor ...
python src/evaluate.py --apply-conll data/processed/gold.conll
python src/evaluate.py --gold data/processed/gold.jsonl \
                       --model models/bert-skills-ner

# 4. the research contribution: cross-discipline analysis + figures
python src/compare.py --source weak                       # quick first pass
python src/compare.py --source bert --model models/bert-skills-ner
```

## A note on evaluation honesty

The corpus labels are produced by the keyword matcher (weak supervision), so
scoring the baseline against them is circular — it would score ~100% by
construction. The `evaluate.py` gold-set workflow exists to fix this: you
correct a small sample by hand (fast, since it's pre-filled), and both systems
are then scored against those independent labels. Report *those* numbers.

## Status

Pipeline complete and unit-tested (taxonomy, matcher boundaries, discipline
inference, metric computation, stratified splitting, CoNLL round-trip, BIO
decoding, chart generation). Remaining work is running it on real data and
writing up results.
