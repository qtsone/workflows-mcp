#!/usr/bin/env python3
"""Fetch public benchmark datasets used by workflows-mcp benchmarks.

Datasets:
- LongMemEval: longmemeval_s_cleaned.json
- LoCoMo: locomo10.json
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import urllib.request
from pathlib import Path

LONGMEMEVAL_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"
    "longmemeval_s_cleaned.json"
)
LOCOMO_REPO_URL = "https://github.com/snap-research/locomo.git"


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    destination.write_bytes(data)


def _clone_or_update_locomo(repo_url: str, repo_dir: Path) -> None:
    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
            check=True,
            text=True,
        )
        return

    subprocess.run(["git", "-C", str(repo_dir), "fetch", "origin", "main"], check=True, text=True)
    subprocess.run(
        ["git", "-C", str(repo_dir), "reset", "--hard", "origin/main"],
        check=True,
        text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch LongMemEval and LoCoMo benchmark datasets")
    parser.add_argument(
        "--output-dir",
        default="benchmarks/data",
        help="Directory where datasets are stored.",
    )
    parser.add_argument(
        "--longmemeval-url",
        default=LONGMEMEVAL_URL,
        help="Download URL for longmemeval_s_cleaned.json",
    )
    parser.add_argument(
        "--locomo-repo-url",
        default=LOCOMO_REPO_URL,
        help="Git repository URL for LoCoMo.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download datasets even when files already exist.",
    )
    parser.add_argument(
        "--skip-locomo-update",
        action="store_true",
        help="Skip git fetch/reset for an existing LoCoMo repo clone.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    longmemeval_path = output_dir / "longmemeval_s_cleaned.json"
    locomo_path = output_dir / "locomo10.json"
    locomo_repo_dir = output_dir / "_locomo_repo"

    if args.force or not longmemeval_path.exists():
        print(f"Downloading LongMemEval -> {longmemeval_path}")
        _download_file(args.longmemeval_url, longmemeval_path)
    else:
        print(f"LongMemEval already exists -> {longmemeval_path}")

    if args.force and locomo_repo_dir.exists():
        shutil.rmtree(locomo_repo_dir)

    if not locomo_repo_dir.exists() or not args.skip_locomo_update:
        print(f"Syncing LoCoMo repo -> {locomo_repo_dir}")
        _clone_or_update_locomo(args.locomo_repo_url, locomo_repo_dir)
    else:
        print(f"Using existing LoCoMo repo clone -> {locomo_repo_dir}")

    source_locomo_file = locomo_repo_dir / "data" / "locomo10.json"
    if not source_locomo_file.exists():
        raise FileNotFoundError(f"Expected LoCoMo file missing: {source_locomo_file}")

    if args.force or not locomo_path.exists():
        shutil.copy2(source_locomo_file, locomo_path)
        print(f"Copied LoCoMo dataset -> {locomo_path}")
    else:
        print(f"LoCoMo dataset already exists -> {locomo_path}")

    print("Done.")


if __name__ == "__main__":
    main()
