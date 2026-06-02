"""
Lightweight shared helpers — deliberately free of heavy imports (torch,
transformers, datasets) so that analysis modules can use them without pulling
in the training stack.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def get_hardware_profile() -> dict:
    """
    Detect available compute hardware and return settings for optimal training.

    Priority: Apple Silicon MPS > NVIDIA CUDA > CPU.

    Returned dict fields
    --------------------
    device_type        "mps" | "cuda" | "cpu"
    device_name        Human-readable label (includes GPU name for CUDA)
    fp16               True when fp16 mixed-precision is safe (CUDA only)
    bf16               True when bf16 mixed-precision is safe (Ampere+ CUDA)
    batch_size_cap     Max recommended per-device train batch size, or None
                       to leave the config value unchanged
    """
    import torch

    if torch.backends.mps.is_available():
        return {
            "device_type": "mps",
            "device_name": "Apple Silicon GPU (MPS)",
            "fp16": False,   # fp16 is not supported on MPS
            "bf16": False,   # bf16 on MPS is experimental; fp32 is safest
            "batch_size_cap": None,
        }

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        major, _ = torch.cuda.get_device_capability(0)
        return {
            "device_type": "cuda",
            "device_name": f"CUDA GPU — {name}",
            "fp16": True,
            "bf16": major >= 8,   # bf16 requires Ampere (sm_80) or newer
            "batch_size_cap": None,
        }

    return {
        "device_type": "cpu",
        "device_name": "CPU (no GPU detected)",
        "fp16": False,
        "bf16": False,
        "batch_size_cap": 8,   # keep RAM pressure reasonable on CPU-only machines
    }


def get_device():
    """Return the best available torch device: MPS (Apple Silicon) > CUDA > CPU."""
    import torch
    profile = get_hardware_profile()
    return torch.device(profile["device_type"])
