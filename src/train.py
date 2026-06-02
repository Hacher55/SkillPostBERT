"""
Fine-tune BERT for skill-extraction NER.

Reads hyperparameters from a YAML config (configs/bert_base.yaml by default);
any field can be overridden on the command line. Trains with the HuggingFace
Trainer, evaluates each epoch with seqeval (entity-level P/R/F1), keeps the best
checkpoint by validation F1, then reports final test-set numbers.

Usage
-----
    python src/train.py                              # use default config
    python src/train.py --config configs/bert_base.yaml
    python src/train.py --epochs 5 --learning-rate 3e-5
    python src/train.py --model distilbert-base-uncased \
                        --output-dir models/distilbert-skills-ner
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .model import build_collator, build_model, load_dataset
from .taxonomy import ID2LABEL
from .utils import ROOT, get_hardware_profile


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_compute_metrics():
    """Return a compute_metrics fn using seqeval entity-level scoring."""
    import evaluate

    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        # drop positions masked with -100 (special tokens, padding)
        true_preds, true_labels = [], []
        for pred_row, lab_row in zip(preds, labels):
            p_seq, l_seq = [], []
            for p, l in zip(pred_row, lab_row):
                if l == -100:
                    continue
                p_seq.append(ID2LABEL[int(p)])
                l_seq.append(ID2LABEL[int(l)])
            true_preds.append(p_seq)
            true_labels.append(l_seq)

        results = seqeval.compute(predictions=true_preds, references=true_labels)
        out = {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
        # per-category breakdown (TECHNICAL, TOOLS, SOFT, CERT)
        for key, val in results.items():
            if isinstance(val, dict):
                out[f"{key}_f1"] = val["f1"]
        return out

    return compute_metrics


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fine-tune BERT for skill NER.")
    ap.add_argument("--config", type=str,
                    default=str(ROOT / "configs" / "bert_base.yaml"))
    # optional overrides (None => fall back to config)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--corpus", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--train-batch-size", type=int, default=None)
    return ap.parse_args()


def resolve(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply CLI overrides onto the YAML config."""
    overrides = {
        "model_name": args.model,
        "corpus": args.corpus,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
    }
    for key, val in overrides.items():
        if val is not None:
            cfg[key] = val
    return cfg


def main() -> None:
    # Trainer is imported lazily so that --help works without torch installed.
    from transformers import Trainer, TrainingArguments

    args = parse_args()
    cfg = resolve(load_config(Path(args.config)), args)
    print("Config:\n" + json.dumps(cfg, indent=2))

    hw = get_hardware_profile()
    print(f"\nHardware detected: {hw['device_name']}")
    print(f"  fp16={hw['fp16']}  bf16={hw['bf16']}")

    corpus = ROOT / cfg["corpus"] if not Path(cfg["corpus"]).is_absolute() \
        else Path(cfg["corpus"])
    ds, tokenizer = load_dataset(
        corpus_path=corpus,
        model_name=cfg["model_name"],
        val_frac=cfg["val_frac"],
        test_frac=cfg["test_frac"],
        seed=cfg["seed"],
    )
    print(f"\nSplits — train: {len(ds['train'])}  "
          f"val: {len(ds['validation'])}  test: {len(ds['test'])}")

    model = build_model(cfg["model_name"])
    collator = build_collator(tokenizer)

    out_dir = ROOT / cfg["output_dir"] if not Path(cfg["output_dir"]).is_absolute() \
        else Path(cfg["output_dir"])

    # Cap batch size on CPU to avoid running out of RAM.
    train_bs = cfg["train_batch_size"]
    if hw["batch_size_cap"] and train_bs > hw["batch_size_cap"]:
        print(f"  CPU detected — capping train batch size "
              f"{train_bs} -> {hw['batch_size_cap']}")
        train_bs = hw["batch_size_cap"]

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=cfg["eval_batch_size"],
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        eval_strategy=cfg["eval_strategy"],
        save_strategy=cfg["save_strategy"],
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model=cfg["metric_for_best_model"],
        fp16=hw["fp16"],
        bf16=hw["bf16"],
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=make_compute_metrics(),
    )

    print("\nTraining ...")
    trainer.train()

    print("\nEvaluating on held-out test set ...")
    test_metrics = trainer.evaluate(ds["test"], metric_key_prefix="test")
    print(json.dumps(test_metrics, indent=2))

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    with (out_dir / "test_metrics.json").open("w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\nSaved model + tokenizer + metrics to {out_dir}")


if __name__ == "__main__":
    main()
