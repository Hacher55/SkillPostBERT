"""
Evaluate fine-tuned BERT against the keyword baseline — fairly.

The catch
---------
The corpus labels are produced BY the keyword matcher (weak supervision). So
scoring the baseline against them is circular: it reproduces its own labels and
"wins" ~100%. The only valid head-to-head needs an INDEPENDENT gold set whose
labels a human has checked. This module provides the tooling to build one
cheaply (correct, don't annotate from scratch) and then scores both systems
against it with entity-level seqeval.

Three-step workflow
-------------------
1. Export a sample of the test split, pre-filled with the matcher's guesses:

       python src/evaluate.py --export-gold --n 60

   Writes gold.jsonl (full metadata) + gold.conll (token<TAB>tag, easy to edit).

2. Open gold.conll in any text editor and FIX the tags — add skills the matcher
   missed, remove false hits, fix category/boundaries. Leave [CLS]/[SEP] rows
   as-is. This is correction, not from-scratch labeling, so it's fast.

3. Fold corrections back in and evaluate:

       python src/evaluate.py --apply-conll gold.conll     # updates gold.jsonl
       python src/evaluate.py --gold gold.jsonl --model models/bert-skills-ner

The final step scores BERT and the baseline on the same gold tokens and writes
a comparison table to results/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baseline import KeywordMatcher
from .model import load_dataset
from .taxonomy import ID2LABEL, LABEL2ID
from .utils import ROOT, read_jsonl as _read_jsonl

PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
DEFAULT_GOLD = PROC_DIR / "gold.jsonl"


# --------------------------------------------------------------------------- #
# Gold-set construction
# --------------------------------------------------------------------------- #
def export_gold(n: int, model_name: str, seed: int) -> None:
    """Sample n records from the test split; write editable gold templates."""
    ds, _ = load_dataset(model_name=model_name, seed=seed)
    test = ds["test"]
    n = min(n, len(test))

    # We need tokens + the raw text + offsets to score the baseline later, but
    # the encoded dataset dropped them. Re-read the corpus to recover them by
    # matching on the token sequence.
    corpus = _read_jsonl(PROC_DIR / "corpus.jsonl")
    by_tokens = {tuple(r["tokens"]): r for r in corpus}

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)

    gold_records = []
    for i in range(n):
        ex = test[i]
        tokens = tok.convert_ids_to_tokens(ex["input_ids"])
        src = by_tokens.get(tuple(tokens))
        # tags: start from the (un-masked) weak labels for easy correction
        tags = [ID2LABEL[t] if t != -100 else "O" for t in ex["labels"]]
        gold_records.append({
            "id": f"{i:04d}",
            "discipline": ex["discipline"],
            "tokens": tokens,
            "tags": tags,
            # text/offsets recovered from corpus if available (for baseline)
            "text": src.get("title", "") if src else "",
        })

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    with DEFAULT_GOLD.open("w", encoding="utf-8") as f:
        for r in gold_records:
            f.write(json.dumps(r) + "\n")

    conll_path = PROC_DIR / "gold.conll"
    with conll_path.open("w", encoding="utf-8") as f:
        for r in gold_records:
            f.write(f"# id: {r['id']}  ({r['discipline']})\n")
            for token, tag in zip(r["tokens"], r["tags"]):
                f.write(f"{token}\t{tag}\n")
            f.write("\n")

    print(f"Wrote {len(gold_records)} records:")
    print(f"  {DEFAULT_GOLD}   (metadata — do not edit by hand)")
    print(f"  {conll_path}   (EDIT THIS: fix tags, then --apply-conll)")
    print("\nValid BIO tags:", ", ".join(LABEL2ID))


def parse_conll(path: Path) -> dict[str, list[str]]:
    """Read a (possibly hand-edited) CoNLL file -> {id: [tags]}."""
    docs: dict[str, list[str]] = {}
    current_id: str | None = None
    tags: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# id:"):
            if current_id is not None:
                docs[current_id] = tags
            current_id = line.split("# id:")[1].strip().split()[0]
            tags = []
        elif line.strip() == "":
            continue
        else:
            parts = line.split("\t")
            if len(parts) >= 2:
                tags.append(parts[1].strip())
    if current_id is not None:
        docs[current_id] = tags
    return docs


def apply_conll(conll_path: Path) -> None:
    """Fold corrected tags from CoNLL back into gold.jsonl (matched by id)."""
    if not DEFAULT_GOLD.exists():
        raise SystemExit(f"{DEFAULT_GOLD} not found. Run --export-gold first.")
    corrected = parse_conll(conll_path)
    records = _read_jsonl(DEFAULT_GOLD)

    updated = 0
    bad = []
    for rec in records:
        new_tags = corrected.get(rec["id"])
        if new_tags is None:
            continue
        if len(new_tags) != len(rec["tokens"]):
            bad.append((rec["id"], len(new_tags), len(rec["tokens"])))
            continue
        invalid = [t for t in new_tags if t not in LABEL2ID]
        if invalid:
            bad.append((rec["id"], f"invalid tags: {set(invalid)}", ""))
            continue
        rec["tags"] = new_tags
        updated += 1

    if bad:
        print("WARNING — some docs not applied:")
        for b in bad:
            print(f"  id {b[0]}: {b[1]} vs {b[2]}")

    with DEFAULT_GOLD.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Applied corrections to {updated} records in {DEFAULT_GOLD}.")


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def predict_baseline(records: list[dict]) -> list[list[str]]:
    """Run the keyword matcher to produce BIO tags per gold record.

    We re-tokenize the stored text to get offsets, then align. If a record has
    no usable text (older export), fall back to detokenizing the gold tokens.
    """
    matcher = KeywordMatcher()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")

    preds = []
    for rec in records:
        text = rec.get("text") or _detokenize(rec["tokens"])
        enc = tok(text, truncation=True, max_length=512,
                  return_offsets_mapping=True)
        toks = tok.convert_ids_to_tokens(enc["input_ids"])
        bio = matcher.bio_tags(toks, enc["offset_mapping"], text)
        # pad/trim to gold length so seqeval aligns
        bio = _fit(bio, len(rec["tokens"]))
        preds.append(bio)
    return preds


def predict_bert(records: list[dict], model_dir: Path) -> list[list[str]]:
    """Load the fine-tuned model and predict BIO tags for each gold record."""
    import torch
    from transformers import (AutoModelForTokenClassification,
                              AutoTokenizer)

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    model.eval()

    preds = []
    with torch.no_grad():
        for rec in records:
            ids = tok.convert_tokens_to_ids(rec["tokens"])
            input_ids = torch.tensor([ids])
            attn = torch.ones_like(input_ids)
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            pred_ids = logits.argmax(-1)[0].tolist()
            preds.append([ID2LABEL[p] for p in pred_ids])
    return preds


def _detokenize(tokens: list[str]) -> str:
    out = []
    for t in tokens:
        if t in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        if t.startswith("##"):
            out[-1] = out[-1] + t[2:]
        else:
            out.append(t)
    return " ".join(out)


def _fit(seq: list[str], n: int) -> list[str]:
    if len(seq) >= n:
        return seq[:n]
    return seq + ["O"] * (n - len(seq))


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score(gold: list[list[str]], pred: list[list[str]]) -> dict:
    import evaluate as hf_evaluate

    seqeval = hf_evaluate.load("seqeval")
    res = seqeval.compute(predictions=pred, references=gold)
    out = {
        "precision": res["overall_precision"],
        "recall": res["overall_recall"],
        "f1": res["overall_f1"],
        "accuracy": res["overall_accuracy"],
    }
    for key, val in res.items():
        if isinstance(val, dict):
            out[f"{key}_f1"] = round(val["f1"], 4)
    return out


def _strip_special(tags_list: list[list[str]],
                   token_lists: list[list[str]]) -> list[list[str]]:
    """Drop [CLS]/[SEP]/[PAD] positions so they don't affect scoring."""
    cleaned = []
    for tags, toks in zip(tags_list, token_lists):
        cleaned.append([
            tag for tag, t in zip(tags, toks)
            if t not in ("[CLS]", "[SEP]", "[PAD]")
        ])
    return cleaned


def run_eval(gold_path: Path, model_dir: Path | None) -> None:
    records = _read_jsonl(gold_path)
    token_lists = [r["tokens"] for r in records]
    gold = _strip_special([r["tags"] for r in records], token_lists)

    print(f"Evaluating on {len(records)} gold records.\n")

    base_pred = _strip_special(predict_baseline(records), token_lists)
    base_scores = score(gold, base_pred)

    rows = {"keyword baseline": base_scores}
    if model_dir is not None:
        bert_pred = _strip_special(predict_bert(records, model_dir), token_lists)
        rows["BERT (fine-tuned)"] = score(gold, bert_pred)

    # print a compact comparison table
    metrics = ["precision", "recall", "f1", "accuracy",
               "TECHNICAL_f1", "TOOLS_f1", "SOFT_f1", "CERT_f1"]
    header = f"{'metric':<14}" + "".join(f"{name:>22}" for name in rows)
    print(header)
    print("-" * len(header))
    for m in metrics:
        line = f"{m:<14}"
        for name in rows:
            v = rows[name].get(m, float("nan"))
            line += f"{v:>22.4f}" if isinstance(v, (int, float)) else f"{'-':>22}"
        print(line)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "comparison.json"
    with out.open("w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved -> {out}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate BERT vs keyword baseline.")
    ap.add_argument("--export-gold", action="store_true",
                    help="Sample test docs into editable gold templates.")
    ap.add_argument("--apply-conll", type=str, default=None,
                    help="Fold corrected tags from a CoNLL file into gold.jsonl.")
    ap.add_argument("--gold", type=str, default=None,
                    help="Path to gold.jsonl to evaluate against.")
    ap.add_argument("--model", type=str, default=None,
                    help="Fine-tuned model dir. Omit to score baseline only.")
    ap.add_argument("--n", type=int, default=60, help="Sample size for --export-gold.")
    ap.add_argument("--model-name", type=str, default="bert-base-uncased",
                    help="Base tokenizer for sampling/baseline alignment.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.export_gold:
        export_gold(args.n, args.model_name, args.seed)
    elif args.apply_conll:
        apply_conll(Path(args.apply_conll))
    elif args.gold:
        model_dir = Path(args.model) if args.model else None
        run_eval(Path(args.gold), model_dir)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
