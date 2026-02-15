#!/usr/bin/env python3
"""Spot-check training data quality.

Usage:
    python scripts/spot_check.py --sample 20 --data data/finetuning/master_train.jsonl
    python scripts/spot_check.py --report data/finetuning/annotations.jsonl
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Spot-check training data quality")
    parser.add_argument("--sample", type=int, default=20, help="Number of examples to sample")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/finetuning/master_train.jsonl"),
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Generate report from annotations JSONL file",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable stratified sampling by source",
    )
    args = parser.parse_args()

    from incite.finetuning.spot_check import (
        format_for_review,
        load_annotations,
        quality_report,
        sample_for_review,
    )

    if args.report:
        annotations = load_annotations(args.report)
        print(quality_report(annotations))
        return

    if not args.data.exists():
        print(f"Error: {args.data} not found")
        sys.exit(1)

    samples = sample_for_review(
        args.data,
        n=args.sample,
        stratify_by_source=not args.no_stratify,
        seed=args.seed,
    )
    print(format_for_review(samples))
    print(f"\nSampled {len(samples)} examples from {args.data}")


if __name__ == "__main__":
    main()
