"""
Cross-discipline skill-demand analysis — the project's research contribution.

Decodes skill spans from BIO-tagged postings, aggregates skill frequency by
engineering discipline (ME / EE / SE), and answers:

  * What are the most in-demand skills in each discipline?
  * How does the skill-category mix differ across disciplines?
  * Which skills are shared across all three, and which are discipline-specific?

Outputs CSV tables + figures to results/.

Two sources of skill labels:
  --source weak   read the matcher's tags straight from corpus.jsonl (instant,
                  no trained model needed — good for a first pass)
  --source bert   run the fine-tuned model over the corpus and use ITS
                  predictions (the analysis you report; BERT generalizes past
                  the dictionary)

Usage
-----
    python src/compare.py --source weak
    python src/compare.py --source bert --model models/bert-skills-ner
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .taxonomy import Category, surface_to_skill
from .utils import ROOT, read_jsonl as _read_jsonl

PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
DISCIPLINES = ("me", "ee", "se")
DISCIPLINE_NAMES = {"me": "Mechanical", "ee": "Electrical", "se": "Software"}


# --------------------------------------------------------------------------- #
# Decode BIO -> skill spans
# --------------------------------------------------------------------------- #
def decode_spans(tokens: list[str], tags: list[str]) -> list[tuple[str, str]]:
    """Turn a BIO-tagged token sequence into (skill_surface, category) pairs."""
    spans: list[tuple[str, str]] = []
    cur_tokens: list[str] = []
    cur_cat: str | None = None

    def flush():
        nonlocal cur_tokens, cur_cat
        if cur_tokens and cur_cat:
            spans.append((_detok(cur_tokens), cur_cat))
        cur_tokens, cur_cat = [], None

    for tok, tag in zip(tokens, tags):
        if tag == "O" or tag in ("[CLS]", "[SEP]", "[PAD]"):
            flush()
            continue
        prefix, _, cat = tag.partition("-")
        if prefix == "B":
            flush()
            cur_tokens, cur_cat = [tok], cat
        elif prefix == "I" and cur_cat == cat:
            cur_tokens.append(tok)
        else:  # stray I- without matching B-: start fresh
            flush()
            cur_tokens, cur_cat = [tok], cat
    flush()
    return spans


def _detok(tokens: list[str]) -> str:
    out: list[str] = []
    for t in tokens:
        if t.startswith("##") and out:
            out[-1] = out[-1] + t[2:]
        else:
            out.append(t)
    return " ".join(out)


def canonicalize(surface: str) -> str:
    """Map a decoded surface form to its canonical skill name when known.

    Unknown surfaces (skills BERT found that aren't in the taxonomy) pass
    through as title-cased text — surfacing these is itself interesting.
    """
    skill = surface_to_skill().get(surface.lower())
    return skill.canonical if skill else surface.strip().title()


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
def aggregate(records: list[dict], tag_key: str = "tags") -> dict:
    """Return per-discipline skill + category counts and posting totals."""
    skill_counts: dict[str, Counter] = {d: Counter() for d in DISCIPLINES}
    cat_counts: dict[str, Counter] = {d: Counter() for d in DISCIPLINES}
    postings: Counter = Counter()

    for rec in records:
        disc = rec.get("discipline")
        if disc not in DISCIPLINES:
            continue
        postings[disc] += 1
        spans = decode_spans(rec["tokens"], rec[tag_key])
        seen_in_posting: set[str] = set()
        for surface, cat in spans:
            name = canonicalize(surface)
            # count each skill at most once per posting (demand = % of postings)
            if name not in seen_in_posting:
                skill_counts[disc][name] += 1
                seen_in_posting.add(name)
            cat_counts[disc][cat] += 1

    return {
        "skill_counts": skill_counts,
        "cat_counts": cat_counts,
        "postings": postings,
    }


def top_skills(agg: dict, disc: str, k: int = 15) -> list[tuple[str, float]]:
    """Top-k skills in a discipline as (name, % of postings mentioning it)."""
    total = max(agg["postings"][disc], 1)
    return [(name, 100 * cnt / total)
            for name, cnt in agg["skill_counts"][disc].most_common(k)]


def shared_and_unique(agg: dict, k: int = 30) -> dict:
    """Skills shared across all three disciplines vs unique to one."""
    top_sets = {
        d: {name for name, _ in agg["skill_counts"][d].most_common(k)}
        for d in DISCIPLINES
    }
    shared = set.intersection(*top_sets.values())
    unique = {
        d: top_sets[d] - set.union(*(top_sets[o] for o in DISCIPLINES if o != d))
        for d in DISCIPLINES
    }
    return {"shared": shared, "unique": unique}


# --------------------------------------------------------------------------- #
# Predictions from BERT (optional source)
# --------------------------------------------------------------------------- #
def add_bert_tags(records: list[dict], model_dir: Path) -> None:
    """Run the fine-tuned model and attach predicted tags as 'bert_tags'."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from .taxonomy import ID2LABEL

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    model.eval()
    with torch.no_grad():
        for rec in records:
            ids = tok.convert_tokens_to_ids(rec["tokens"])
            input_ids = torch.tensor([ids])
            logits = model(input_ids=input_ids,
                           attention_mask=torch.ones_like(input_ids)).logits
            rec["bert_tags"] = [ID2LABEL[p] for p in logits.argmax(-1)[0].tolist()]


# --------------------------------------------------------------------------- #
# Outputs: tables + figures
# --------------------------------------------------------------------------- #
def write_tables(agg: dict) -> None:
    import pandas as pd

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # top skills per discipline
    frames = []
    for d in DISCIPLINES:
        rows = top_skills(agg, d, k=20)
        df = pd.DataFrame(rows, columns=["skill", "pct_postings"])
        df.insert(0, "discipline", DISCIPLINE_NAMES[d])
        frames.append(df)
    pd.concat(frames).to_csv(RESULTS_DIR / "top_skills_by_discipline.csv",
                             index=False)

    # category mix
    cat_rows = []
    for d in DISCIPLINES:
        total = sum(agg["cat_counts"][d].values()) or 1
        for cat in Category:
            cat_rows.append({
                "discipline": DISCIPLINE_NAMES[d],
                "category": cat.value,
                "pct": 100 * agg["cat_counts"][d][cat.value] / total,
            })
    pd.DataFrame(cat_rows).to_csv(RESULTS_DIR / "category_mix.csv", index=False)
    print(f"Wrote tables to {RESULTS_DIR}/")


def plot_top_skills(agg: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    colors = {"me": "#1f6f8b", "ee": "#e08a3c", "se": "#2a9d8f"}
    for ax, d in zip(axes, DISCIPLINES):
        rows = top_skills(agg, d, k=12)[::-1]
        names = [r[0] for r in rows]
        pcts = [r[1] for r in rows]
        ax.barh(names, pcts, color=colors[d])
        ax.set_title(f"{DISCIPLINE_NAMES[d]} — top skills")
        ax.set_xlabel("% of postings")
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("In-demand skills by engineering discipline", fontsize=14)
    fig.tight_layout()
    out = RESULTS_DIR / "top_skills_by_discipline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_category_mix(agg: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cats = [c.value for c in Category]
    cat_colors = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]
    data = np.array([
        [agg["cat_counts"][d][c] for c in cats] for d in DISCIPLINES
    ], dtype=float)
    row_tot = data.sum(axis=1, keepdims=True)
    row_tot[row_tot == 0] = 1
    pct = 100 * data / row_tot

    fig, ax = plt.subplots(figsize=(9, 5))
    left = np.zeros(len(DISCIPLINES))
    y = np.arange(len(DISCIPLINES))
    for j, cat in enumerate(cats):
        ax.barh(y, pct[:, j], left=left, label=cat, color=cat_colors[j])
        left += pct[:, j]
    ax.set_yticks(y)
    ax.set_yticklabels([DISCIPLINE_NAMES[d] for d in DISCIPLINES])
    ax.set_xlabel("% of skill mentions")
    ax.set_title("Skill-category mix by discipline")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    out = RESULTS_DIR / "category_mix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_heatmap(agg: dict, k: int = 15) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # union of each discipline's top-k skills
    skills: list[str] = []
    for d in DISCIPLINES:
        for name, _ in agg["skill_counts"][d].most_common(k):
            if name not in skills:
                skills.append(name)

    mat = np.zeros((len(skills), len(DISCIPLINES)))
    for j, d in enumerate(DISCIPLINES):
        total = max(agg["postings"][d], 1)
        for i, s in enumerate(skills):
            mat[i, j] = 100 * agg["skill_counts"][d][s] / total

    fig, ax = plt.subplots(figsize=(7, max(6, len(skills) * 0.32)))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(DISCIPLINES)))
    ax.set_xticklabels([DISCIPLINE_NAMES[d] for d in DISCIPLINES])
    ax.set_yticks(range(len(skills)))
    ax.set_yticklabels(skills, fontsize=8)
    ax.set_title(f"Skill demand across disciplines (% of postings)")
    fig.colorbar(im, ax=ax, label="% of postings")
    fig.tight_layout()
    out = RESULTS_DIR / "skill_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-discipline skill analysis.")
    ap.add_argument("--source", choices=["weak", "bert"], default="weak",
                    help="Use matcher weak tags, or fine-tuned BERT predictions.")
    ap.add_argument("--model", type=str, default=None,
                    help="Model dir (required for --source bert).")
    ap.add_argument("--corpus", type=str,
                    default=str(PROC_DIR / "corpus.jsonl"))
    args = ap.parse_args()

    records = _read_jsonl(Path(args.corpus))
    print(f"Loaded {len(records)} postings.")

    tag_key = "tags"
    if args.source == "bert":
        if not args.model:
            raise SystemExit("--source bert requires --model.")
        # corpus uses 'ner_tags' (ids); BERT writes 'bert_tags' (strings)
        print("Running BERT inference over corpus ...")
        add_bert_tags(records, Path(args.model))
        tag_key = "bert_tags"
    else:
        # convert stored ner_tags (ids) to BIO strings under key 'tags'
        from .taxonomy import ID2LABEL
        for r in records:
            r["tags"] = [ID2LABEL[t] for t in r["ner_tags"]]

    agg = aggregate(records, tag_key=tag_key)

    print("\nPostings per discipline:")
    for d in DISCIPLINES:
        print(f"  {DISCIPLINE_NAMES[d]:12s} {agg['postings'][d]}")

    su = shared_and_unique(agg)
    print("\nShared across all three (top-30 overlap):")
    print("  " + (", ".join(sorted(su["shared"])) or "(none)"))
    for d in DISCIPLINES:
        uniq = sorted(su["unique"][d])
        print(f"\nUnique to {DISCIPLINE_NAMES[d]}:")
        print("  " + (", ".join(uniq) or "(none)"))

    write_tables(agg)
    plot_top_skills(agg)
    plot_category_mix(agg)
    plot_heatmap(agg)
    print("\nDone.")


if __name__ == "__main__":
    main()
