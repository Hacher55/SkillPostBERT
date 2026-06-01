"""
Preprocess raw job-posting CSVs into a clean, weak-labeled corpus for NER.

Designed to be column-agnostic: Kaggle datasets vary in their column names, so
we fuzzy-match the title and description fields rather than hard-coding them.

Steps
-----
1. Load every CSV in data/raw/ (or a specified file).
2. Identify the title + description columns by header matching.
3. Clean the description text (HTML, whitespace, boilerplate).
4. Infer engineering discipline (ME / EE / SE / other) from title + text.
5. Keep only engineering postings; drop the rest.
6. Tokenize with the model's fast tokenizer and weak-label BIO tags via the
   keyword matcher (src/baseline.py).
7. Write JSONL: one record per posting with tokens, ner_tags, discipline, meta.

Usage
-----
    python src/preprocess.py                          # all CSVs in data/raw/
    python src/preprocess.py --input data/raw/x.csv   # one file
    python src/preprocess.py --max 5000               # cap per discipline
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from .baseline import KeywordMatcher
from .taxonomy import LABEL2ID

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
DEFAULT_MODEL = "bert-base-uncased"

# Candidate header names (lowercased) for the fields we need.
TITLE_COLS = ("title", "job_title", "position", "job title", "name", "job_name")
DESC_COLS = (
    "description", "job_description", "job description", "details",
    "job_summary", "summary", "text", "job_details", "jobdescription",
)

# --------------------------------------------------------------------------- #
# Discipline inference. Each posting is scored against keyword sets; the
# highest-scoring discipline wins if it clears a small margin, else "other".
# --------------------------------------------------------------------------- #
DISCIPLINE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "me": (
        "mechanical", "thermal", "hvac", "mechatronics", "manufacturing engineer",
        "cad", "solidworks", "catia", "tolerance", "gd&t", "fea", "cfd",
        "fluid", "thermodynamics", "machining", "mechanical design",
    ),
    "ee": (
        "electrical", "electronics", "circuit", "pcb", "embedded", "firmware",
        "rf ", "signal processing", "power systems", "fpga", "vhdl", "verilog",
        "analog", "semiconductor", "hardware engineer", "controls engineer",
    ),
    "se": (
        "software", "developer", "programmer", "full stack", "backend",
        "frontend", "devops", "data engineer", "machine learning", "ml engineer",
        "python", "java ", "kubernetes", "microservices", "api", "web developer",
    ),
}
# title keywords weigh more than body keywords
TITLE_WEIGHT = 3
BODY_WEIGHT = 1
MARGIN = 2  # winner must beat runner-up by at least this much


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    norm_map = {_norm(c): c for c in columns}
    # exact match first
    for cand in candidates:
        if cand in norm_map:
            return norm_map[cand]
    # then substring (e.g. "job_description_text")
    for cand in candidates:
        for norm_col, orig in norm_map.items():
            if cand in norm_col:
                return orig
    return None


def clean_text(raw: str) -> str:
    """Strip HTML, unescape entities, collapse whitespace and bullets."""
    if not isinstance(raw, str):
        return ""
    text = html.unescape(raw)
    text = re.sub(r"<[^>]+>", " ", text)          # html tags
    text = re.sub(r"&[a-z]+;", " ", text)         # leftover entities
    text = re.sub(r"[•▪◦‣·]", " ", text)          # bullet glyphs
    text = re.sub(r"\s+", " ", text)              # collapse whitespace
    return text.strip()


def infer_discipline(title: str, body: str) -> str:
    """Return 'me' | 'ee' | 'se' | 'other'."""
    t, b = _norm(title), _norm(body)
    scores: dict[str, int] = {}
    for disc, kws in DISCIPLINE_KEYWORDS.items():
        score = 0
        for kw in kws:
            if kw in t:
                score += TITLE_WEIGHT
            if kw in b:
                score += BODY_WEIGHT
        scores[disc] = score
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top, top_score = ranked[0]
    runner_score = ranked[1][1] if len(ranked) > 1 else 0
    if top_score == 0 or (top_score - runner_score) < MARGIN:
        return "other"
    return top


# --------------------------------------------------------------------------- #
# Main preprocessing
# --------------------------------------------------------------------------- #
def process_file(
    path: Path,
    matcher: KeywordMatcher,
    tokenizer,
    max_len: int,
    per_discipline_cap: int | None,
    counts: Counter,
    records: list[dict],
) -> None:
    print(f"\nReading {path.name} ...")
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  could not read as CSV ({exc}); skipping.")
        return

    title_col = _find_column(list(df.columns), TITLE_COLS)
    desc_col = _find_column(list(df.columns), DESC_COLS)
    if desc_col is None:
        print(f"  no description-like column found in {list(df.columns)[:8]}...; skipping.")
        return
    print(f"  title col: {title_col!r}   description col: {desc_col!r}   rows: {len(df)}")

    kept = 0
    for _, row in df.iterrows():
        title = str(row[title_col]) if title_col else ""
        body = clean_text(row[desc_col])
        if len(body) < 100:  # skip near-empty postings
            continue

        discipline = infer_discipline(title, body)
        if discipline == "other":
            continue
        if per_discipline_cap and counts[discipline] >= per_discipline_cap:
            continue

        # tokenize with offsets so we can align weak labels
        enc = tokenizer(
            body,
            truncation=True,
            max_length=max_len,
            return_offsets_mapping=True,
        )
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
        offsets = enc["offset_mapping"]
        bio = matcher.bio_tags(tokens, offsets, body)
        ner_ids = [LABEL2ID[t] for t in bio]

        # skip postings where the weak labeler found nothing (no signal)
        if all(t == "O" for t in bio):
            continue

        records.append({
            "title": title.strip(),
            "discipline": discipline,
            "tokens": tokens,
            "ner_tags": ner_ids,
            "source": path.name,
        })
        counts[discipline] += 1
        kept += 1

    print(f"  kept {kept} engineering postings with skill signal.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess job postings for NER.")
    ap.add_argument("--input", type=str, default=None,
                    help="Single CSV path. Default: all CSVs in data/raw/.")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL,
                    help="Tokenizer to align BIO labels to.")
    ap.add_argument("--max-len", type=int, default=256,
                    help="Max tokens per posting.")
    ap.add_argument("--max", type=int, default=None,
                    help="Cap postings PER discipline (for quick runs).")
    ap.add_argument("--out", type=str, default=str(OUT_DIR / "corpus.jsonl"))
    args = ap.parse_args()

    inputs = ([Path(args.input)] if args.input
              else sorted(RAW_DIR.glob("*.csv")))
    if not inputs:
        raise SystemExit(
            f"No CSVs found in {RAW_DIR}. Run download_data.py first, "
            "or pass --input."
        )

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    matcher = KeywordMatcher()

    counts: Counter = Counter()
    records: list[dict] = []
    for path in inputs:
        process_file(path, matcher, tokenizer, args.max_len,
                     args.max, counts, records)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nWrote {len(records)} records -> {out_path}")
    print("Per-discipline counts:")
    for disc in ("me", "ee", "se"):
        print(f"  {disc}: {counts[disc]}")


if __name__ == "__main__":
    main()
