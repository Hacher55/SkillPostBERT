"""
Model + dataset plumbing for BERT token-classification (NER).

The corpus written by preprocess.py already carries subword-aligned BIO labels
(one label id per wordpiece token, special tokens labeled "O"). Here we:

  1. Load that JSONL.
  2. Convert stored token strings back to input_ids (deterministic for BERT
     wordpiece) and build attention masks.
  3. Mask special tokens ([CLS]/[SEP]) with -100 so they contribute to neither
     the loss nor the metrics — standard practice for token classification.
  4. Split into train / validation / test, stratified by discipline so each
     split has comparable ME/EE/SE proportions.
  5. Build the model with the taxonomy's label scheme baked in (so a reloaded
     checkpoint reports human-readable tags).

`DataCollatorForTokenClassification` handles padding at batch time, including
padding the labels with -100.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

from .taxonomy import ID2LABEL, LABEL2ID, LABELS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "processed" / "corpus.jsonl"
DEFAULT_MODEL = "bert-base-uncased"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _encode_record(rec: dict, tokenizer) -> dict:
    """Token strings -> input_ids; mask special tokens in the labels."""
    input_ids = tokenizer.convert_tokens_to_ids(rec["tokens"])
    labels = list(rec["ner_tags"])
    special = set(tokenizer.all_special_ids)
    labels = [(-100 if tid in special else lab)
              for tid, lab in zip(input_ids, labels)]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "discipline": rec.get("discipline", "other"),
    }


def _stratified_split(
    records: list[dict],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split preserving discipline proportions across train/val/test."""
    import random

    rng = random.Random(seed)
    by_disc: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_disc[r["discipline"]].append(r)

    train, val, test = [], [], []
    for disc, items in by_disc.items():
        rng.shuffle(items)
        n = len(items)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        test.extend(items[:n_test])
        val.extend(items[n_test:n_test + n_val])
        train.extend(items[n_test + n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def load_dataset(
    corpus_path: Path = DEFAULT_CORPUS,
    model_name: str = DEFAULT_MODEL,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[DatasetDict, "AutoTokenizer"]:
    """Return a DatasetDict (train/validation/test) plus the tokenizer."""
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"{corpus_path} not found. Run preprocess.py first."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    raw = _read_jsonl(corpus_path)
    encoded = [_encode_record(r, tokenizer) for r in raw]

    train, val, test = _stratified_split(encoded, val_frac, test_frac, seed)
    ds = DatasetDict({
        "train": Dataset.from_list(train),
        "validation": Dataset.from_list(val),
        "test": Dataset.from_list(test),
    })
    return ds, tokenizer


def build_model(model_name: str = DEFAULT_MODEL):
    """Token-classification model with the taxonomy label scheme attached."""
    return AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )


def build_collator(tokenizer):
    return DataCollatorForTokenClassification(tokenizer)


if __name__ == "__main__":
    # Quick sanity check — requires a built corpus.
    ds, tok = load_dataset()
    print(ds)
    for split in ds:
        discs = ds[split]["discipline"]
        from collections import Counter
        print(f"  {split}: {Counter(discs)}")
