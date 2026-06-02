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

Hardware is detected automatically at runtime (Apple Silicon MPS > NVIDIA CUDA > CPU).
A GPU is strongly recommended for BERT; DistilBERT is tolerable on CPU.

## Data

- LinkedIn Job Postings Dataset (Kaggle)
- Indeed Job Scrape Dataset (Kaggle)

Place raw CSVs under `data/raw/`. They are gitignored — see `data/raw/README.md`.

---

## Setup

### Prerequisites

- Python 3.10+
- A [Kaggle account](https://www.kaggle.com) with an API token (`kaggle.json`)

### 1 — Create and activate a virtual environment

**Mac / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Place your Kaggle credentials

**Mac / Linux**
```
~/.kaggle/kaggle.json
```
Then lock down the permissions:
```bash
chmod 600 ~/.kaggle/kaggle.json
```

**Windows**
```
C:\Users\<YourUsername>\.kaggle\kaggle.json
```
No permission change needed on Windows.

---

## Running the pipeline

The pipeline has a manual annotation checkpoint in the middle — you hand-correct
a small gold set so the BERT-vs-baseline comparison is honest rather than
circular. It is split into two scripts.

> **Windows note:** The run scripts are Bash (`.sh`). Use **Git Bash** or **WSL**
> to execute them directly, or run each step individually with the `python -m`
> commands shown in the [step-by-step section](#step-by-step) below.

### Quick run (Mac / Linux / Git Bash)

```bash
./run_part1.sh
# ... hand-correct data/processed/gold.conll in any text editor ...
./run_part2.sh
```

Override the model or output directory inline:

**Mac / Linux / Git Bash**
```bash
MODEL_NAME=distilbert-base-uncased MODEL_DIR=models/distilbert-skills-ner ./run_part1.sh
```

**Windows (PowerShell)**
```powershell
$env:MODEL_NAME = "distilbert-base-uncased"
$env:MODEL_DIR  = "models/distilbert-skills-ner"
bash run_part1.sh      # requires Git Bash or WSL on PATH
```

---

## Step-by-step

Run each stage individually with `python -m`. These commands work identically
on Windows, Mac, and Linux (run from the repo root with your venv active).

### 1. Download data

```bash
python -m src.download_data
```

Fetches the LinkedIn job postings dataset from Kaggle into `data/raw/`.
Requires `kaggle.json` to be in place (see Setup above).

### 2. Preprocess

```bash
# all CSVs in data/raw/ (cap at 2 000 postings per discipline for a quick run)
python -m src.preprocess --max 2000

# single file
python -m src.preprocess --input data/raw/myfile.csv

# full corpus (no cap — slower)
python -m src.preprocess
```

Cleans text, infers engineering discipline (ME / EE / SE), tokenizes with the
BERT fast tokenizer, and weak-labels BIO tags via the keyword matcher.
Output: `data/processed/corpus.jsonl`.

### 3. Train

```bash
# default config (configs/bert_base.yaml)
python -m src.train

# lighter model — viable on CPU (still slow, but feasible)
python -m src.train --model distilbert-base-uncased \
                    --output-dir models/distilbert-skills-ner

# override hyperparameters on the CLI
python -m src.train --epochs 5 --learning-rate 3e-5
```

Hardware is detected automatically — MPS, CUDA, or CPU. On CPU, DistilBERT
is recommended; BERT-base can take several hours.

### 4. Build a gold evaluation set

```bash
# export 60 pre-filled records (≈ 20 per discipline)
python -m src.evaluate --export-gold --n 60
```

Opens two files:
- `data/processed/gold.conll` — **edit this** to fix the BIO tags
- `data/processed/gold.jsonl` — metadata (do not edit by hand)

Open `gold.conll` in any text editor. Each line is `token<TAB>tag`. Fix the
second column: add missed skills, remove false positives, correct categories.

### 5. Apply corrections and evaluate

```bash
# fold your corrections back in
python -m src.evaluate --apply-conll data/processed/gold.conll

# score BERT vs the keyword baseline on the gold set
python -m src.evaluate --gold data/processed/gold.jsonl \
                        --model models/bert-skills-ner
```

Results are printed as a comparison table and saved to `results/comparison.json`.

### 6. Cross-discipline analysis

```bash
# quick first pass using the matcher's weak labels (no trained model needed)
python -m src.compare --source weak

# full analysis using BERT predictions (the results you report)
python -m src.compare --source bert --model models/bert-skills-ner
```

Writes CSV tables and figures to `results/`.

---

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
  utils.py         shared helpers + hardware detection
configs/
  bert_base.yaml   default training hyperparameters
data/
  raw/             gitignored — place Kaggle CSVs here
  processed/       corpus.jsonl, gold files (generated)
results/           metrics JSON + figures (generated)
models/            fine-tuned checkpoints (generated, gitignored)
run_part1.sh       part 1 pipeline script (Bash)
run_part2.sh       part 2 pipeline script (Bash)
```

---

## A note on evaluation honesty

The corpus labels are produced by the keyword matcher (weak supervision), so
scoring the baseline against them is circular — it would score ~100% by
construction. The `evaluate.py` gold-set workflow exists to fix this: you
correct a small sample by hand (fast, since it's pre-filled), and both systems
are then scored against those independent labels. Report *those* numbers.

---

## Status

Pipeline complete and unit-tested (taxonomy, matcher boundaries, discipline
inference, metric computation, stratified splitting, CoNLL round-trip, BIO
decoding, chart generation). Remaining work is running it on real data and
writing up results.
