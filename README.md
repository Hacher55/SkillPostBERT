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

## Layout

```
src/         pipeline modules
configs/     YAML run configs
data/        raw (gitignored) + processed
notebooks/   exploration
results/     metrics + figures
```

## Status

Early scaffolding. Order of implementation: taxonomy + baseline → preprocessing
→ BERT training → evaluation → cross-discipline comparison.
