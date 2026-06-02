"""
Download the raw Kaggle datasets into data/raw/.

This must run on YOUR machine — Kaggle requires authentication tied to your
account, so it can't be done from a sandbox.

Setup (one time)
----------------
1. Create a free account at https://www.kaggle.com
2. Account -> Settings -> API -> "Create New Token". This downloads
   `kaggle.json` (your API credentials).
3. Place it where the Kaggle client looks for it:
       Linux/macOS:  ~/.kaggle/kaggle.json   (then: chmod 600 ~/.kaggle/kaggle.json)
       Windows:      C:\\Users\\<you>\\.kaggle\\kaggle.json
4. pip install kaggle      (already in requirements.txt)

Then
----
    python src/download_data.py

The datasets are large (the LinkedIn one is ~1 GB unzipped; the 1.3M skills
set is several GB). Comment out whichever you don't want before running.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# (kaggle dataset slug, human label). Comment out any you don't want.
DATASETS: list[tuple[str, str]] = [
    ("arshkon/linkedin-job-postings", "LinkedIn Job Postings (2023-2024)"),
    # The 1.3M set is very large — uncomment only if you want the extra volume:
    # ("asaniczka/1-3m-linkedin-jobs-and-skills-2024", "1.3M LinkedIn Jobs & Skills"),
]


def _check_kaggle_available() -> None:
    try:
        import kaggle  # noqa: F401
    except ImportError:
        sys.exit(
            "The 'kaggle' package isn't installed.\n"
            "  pip install kaggle\n"
            "Then create an API token (see this file's docstring)."
        )
    except (OSError, ValueError) as exc:
        sys.exit(
            f"Kaggle credentials invalid or missing ({exc}).\n"
            "Create kaggle.json and place it in C:\\Users\\<you>\\.kaggle\\\n"
            "  or set KAGGLE_USERNAME / KAGGLE_KEY env vars.\n"
            "(see this file's docstring for full setup steps)"
        )


def _marker(slug: str) -> Path:
    safe = slug.replace("/", "_")
    return RAW_DIR / f".{safe}.done"


def download(slug: str, label: str) -> None:
    print(f"\n=== {label} ===")
    marker = _marker(slug)
    if marker.exists():
        print(f"    already cached — skipping download.")
        return
    print(f"    {slug} -> {RAW_DIR}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle", "datasets", "download",
        "-d", slug,
        "-p", str(RAW_DIR),
        "--unzip",
    ]
    try:
        subprocess.run(cmd, check=True)
        marker.touch()
        print(f"    done.")
    except subprocess.CalledProcessError as exc:
        print(f"    FAILED ({exc.returncode}). "
              f"Check the slug is current and your token is valid.")


def main() -> None:
    _check_kaggle_available()
    for slug, label in DATASETS:
        download(slug, label)
    print("\nAll requested datasets processed. Contents of data/raw/:")
    for p in sorted(RAW_DIR.glob("*")):
        size_mb = p.stat().st_size / 1e6 if p.is_file() else 0
        print(f"  {p.name}  ({size_mb:.1f} MB)" if p.is_file() else f"  {p.name}/")


if __name__ == "__main__":
    main()
