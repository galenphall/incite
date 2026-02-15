#!/usr/bin/env python3
"""Merge all training data sources into master train/dev files.

Reads source-specific JSONL files from data/finetuning/, normalizes to a
common TrainingExample schema, filters out junk (LLM refusals, too-short
text), deduplicates by query text, and writes:

    data/finetuning/master_train.jsonl
    data/finetuning/master_dev.jsonl

Source files are NOT modified or deleted.

Usage:
    python scripts/merge_training_data.py
    python scripts/merge_training_data.py --dry-run     # show stats only
    python scripts/merge_training_data.py --force        # overwrite existing master files
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "finetuning"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from incite.finetuning.quality import (
    dedup_by_query,
    filter_low_similarity,
    normalize_training_example,
    remove_train_dev_leakage,
)

# --- Paper-level training/dev sources (4 core sources) ---
TRAIN_SOURCES = [
    (
        "../finetuning_v2/train.jsonl",
        "v2_pipeline",
    ),  # ~87K (s2orc_abstract + synthetic + paper_level)
    ("s2orc_citation/train.jsonl", "s2orc_citation"),  # ~14K (real citation contexts)
    ("unarxiv_expanded/train.jsonl", "paper_level_expanded"),  # ~13K
    ("unarxiv_expanded_v2/train.jsonl", "paper_level_expanded_v2"),  # additional unarxiv
    ("train.jsonl", "paper_level"),  # ~3K (original paper-level)
    ("failure_mining/train.jsonl", "failure_mining"),  # ~7K (eval failure cases)
    (
        "s2_contexts/train_augmented.jsonl",
        "s2_contexts",
    ),  # ~20K (S2 API citation contexts, neural+BM25 hard negs)
]

DEV_SOURCES = [
    ("../finetuning_v2/dev.jsonl", "v2_pipeline"),
    ("s2orc_citation/dev.jsonl", "s2orc_citation"),
    ("unarxiv_expanded/dev.jsonl", "paper_level_expanded"),
    ("unarxiv_expanded_v2/dev.jsonl", "paper_level_expanded_v2"),
    ("dev.jsonl", "paper_level"),
    (
        "s2_contexts/dev_augmented.jsonl",
        "s2_contexts",
    ),  # ~5K (S2 API citation contexts, neural+BM25 hard negs)
]

# Sources to drop after normalization (post-remap source names).
# s2orc_abstract: 33% wrong pairs, 54% format mismatch, 0 hard negatives
# (100-example spot-check audit, Feb 2026)
DROP_SOURCES = {"s2orc_abstract"}


def load_and_normalize(
    sources: list[tuple[str, str]],
) -> tuple[list[dict], dict]:
    """Load source files, normalize, and return examples + stats."""
    examples = []
    stats: dict[str, dict] = {}

    for filename, source_tag in sources:
        path = DATA_DIR / filename
        if not path.exists():
            stats[filename] = {"status": "not found"}
            continue

        raw_count = 0
        kept = 0
        filtered = 0
        dropped = 0

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw_count += 1
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    filtered += 1
                    continue

                normalized = normalize_training_example(raw, source_tag)
                if normalized is None:
                    filtered += 1
                    continue

                if normalized.get("source") in DROP_SOURCES:
                    dropped += 1
                    continue

                examples.append(normalized)
                kept += 1

        stats[filename] = {
            "raw": raw_count,
            "kept": kept,
            "filtered": filtered,
            "dropped": dropped,
            "source_tag": source_tag,
        }

    return examples, stats


def main():
    parser = argparse.ArgumentParser(description="Merge training data into master files")
    parser.add_argument("--dry-run", action="store_true", help="Show stats without writing")
    parser.add_argument("--force", action="store_true", help="Overwrite existing master files")
    parser.add_argument(
        "--cap-source",
        nargs=2,
        metavar=("SOURCE", "N"),
        action="append",
        default=[],
        help="Cap a source to N examples (e.g. --cap-source s2orc_abstract 30000)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(2, (os.cpu_count() or 4) // 2),
        help="Max threads for numpy/torch (default: half of available cores)",
    )
    args = parser.parse_args()

    # Limit CPU threads to prevent thermal throttling / OOM kills on Mac
    threads = str(args.threads)
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["MKL_NUM_THREADS"] = threads
    os.environ["OPENBLAS_NUM_THREADS"] = threads
    try:
        import torch

        torch.set_num_threads(args.threads)
    except ImportError:
        pass
    print(f"Thread limit: {threads}")

    # --- Train ---
    print("=" * 60)
    print("TRAIN DATA")
    print("=" * 60)
    train_examples, train_stats = load_and_normalize(TRAIN_SOURCES)
    for fname, s in train_stats.items():
        if s.get("status") == "not found":
            print(f"  {fname:<45} NOT FOUND")
        else:
            drop_note = f", {s['dropped']} dropped" if s.get("dropped") else ""
            print(
                f"  {fname:<45} {s['raw']:>6} raw -> "
                f"{s['kept']:>6} kept  ({s['filtered']} filtered{drop_note})"
            )

    train_filtered, train_sim_removed = filter_low_similarity(train_examples)
    train_deduped, train_dupes = dedup_by_query(train_filtered)
    print(f"\n  Total after normalization: {len(train_examples):,}")
    print(f"  Similarity filtered:      {train_sim_removed:,}")
    print(f"  Duplicates removed:       {train_dupes:,}")
    print(f"  Final train count:        {len(train_deduped):,}")

    # Source distribution
    source_counts = Counter(ex.get("source", "unknown") for ex in train_deduped)
    print("\n  By source:")
    for src, cnt in source_counts.most_common():
        print(f"    {src:<25} {cnt:>6,}")

    has_negs = sum(1 for ex in train_deduped if ex.get("hard_negatives"))
    print(
        f"\n  Has hard_negatives:   {has_negs:,} "
        f"({100 * has_negs / max(1, len(train_deduped)):.1f}%)"
    )

    # --- Dev ---
    print("\n" + "=" * 60)
    print("DEV DATA")
    print("=" * 60)
    dev_examples, dev_stats = load_and_normalize(DEV_SOURCES)
    for fname, s in dev_stats.items():
        if s.get("status") == "not found":
            print(f"  {fname:<45} NOT FOUND")
        else:
            drop_note = f", {s['dropped']} dropped" if s.get("dropped") else ""
            print(
                f"  {fname:<45} {s['raw']:>6} raw -> "
                f"{s['kept']:>6} kept  ({s['filtered']} filtered{drop_note})"
            )

    dev_filtered, dev_sim_removed = filter_low_similarity(dev_examples)
    dev_deduped, dev_dupes = dedup_by_query(dev_filtered)
    print(f"\n  Total after normalization: {len(dev_examples):,}")
    print(f"  Similarity filtered:      {dev_sim_removed:,}")
    print(f"  Duplicates removed:       {dev_dupes:,}")
    print(f"  Final dev count:          {len(dev_deduped):,}")

    # --- Source capping ---
    cap_removed = 0
    if args.cap_source:
        print("\n" + "=" * 60)
        print("SOURCE CAPPING")
        print("=" * 60)
        import random as _rng

        _rng.seed(42)
        for source_name, cap_str in args.cap_source:
            cap_n = int(cap_str)
            matching = [ex for ex in train_deduped if ex.get("source") == source_name]
            others = [ex for ex in train_deduped if ex.get("source") != source_name]
            before = len(matching)
            if before > cap_n:
                _rng.shuffle(matching)
                matching = matching[:cap_n]
                removed = before - len(matching)
                cap_removed += removed
                print(f"  {source_name}: {before:,} -> {len(matching):,} (removed {removed:,})")
            else:
                print(f"  {source_name}: {before:,} (under cap {cap_n:,}, no change)")
            train_deduped = others + matching
        print(f"  Train count after cap: {len(train_deduped):,}")

    # --- Cross-set leakage removal ---
    print("\n" + "=" * 60)
    print("CROSS-SET LEAKAGE")
    print("=" * 60)
    train_deduped, leakage_removed = remove_train_dev_leakage(train_deduped, dev_deduped)
    print(f"  Train queries also in dev: {leakage_removed:,} (removed from train)")
    print(f"  Final train count:         {len(train_deduped):,}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  master_train.jsonl:  {len(train_deduped):>6,} examples")
    print(f"  master_dev.jsonl:    {len(dev_deduped):>6,} examples")
    print(f"  Total:               {len(train_deduped) + len(dev_deduped):>6,} examples")

    if args.dry_run:
        print("\nDry run complete. Use without --dry-run to write files.")
        return

    # --- Write ---
    train_path = DATA_DIR / "master_train.jsonl"
    dev_path = DATA_DIR / "master_dev.jsonl"

    for p in [train_path, dev_path]:
        if p.exists() and not args.force:
            print(f"\nERROR: {p} already exists. Use --force to overwrite.")
            return

    with open(train_path, "w") as f:
        for ex in train_deduped:
            f.write(json.dumps(ex) + "\n")
    print(f"\nWrote {train_path} ({len(train_deduped):,} examples)")

    with open(dev_path, "w") as f:
        for ex in dev_deduped:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {dev_path} ({len(dev_deduped):,} examples)")

    # --- Pipeline info ---
    train_source_counts = dict(
        Counter(ex.get("source", "unknown") for ex in train_deduped).most_common()
    )
    dev_source_counts = dict(
        Counter(ex.get("source", "unknown") for ex in dev_deduped).most_common()
    )
    pipeline_info = {
        "created": datetime.now(timezone.utc).isoformat(),
        "train": {
            "total": len(train_deduped),
            "by_source": train_source_counts,
            "duplicates_removed": train_dupes,
            "similarity_filtered": train_sim_removed,
            "leakage_removed": leakage_removed,
            "source_capped": cap_removed,
        },
        "dev": {
            "total": len(dev_deduped),
            "by_source": dev_source_counts,
            "duplicates_removed": dev_dupes,
            "similarity_filtered": dev_sim_removed,
        },
    }
    info_path = DATA_DIR / "pipeline_info.json"
    with open(info_path, "w") as f:
        json.dump(pipeline_info, f, indent=2)
    print(f"Wrote {info_path}")

    print("\nTo train:")
    print("  incite finetune train --data-dir data/finetuning \\")
    print("      --train master_train.jsonl --dev master_dev.jsonl")


if __name__ == "__main__":
    main()
