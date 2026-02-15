"""Spot-checking tools for training data quality review.

Supports stratified sampling, human/LLM-readable formatting, and
annotation persistence for tracking data quality over time.
"""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from incite.finetuning.types import TrainingExample, load_training_data


@dataclass
class SpotCheckResult:
    """Annotation for a single reviewed example."""

    example_index: int
    source: str
    query_quality: int  # 1-5
    positive_quality: int  # 1-5
    notes: str = ""
    reviewer: str = "human"  # "human" or "claude"


def sample_for_review(
    data_path: Path,
    n: int = 20,
    stratify_by_source: bool = True,
    seed: int = 42,
) -> list[tuple[int, TrainingExample]]:
    """Sample examples for review, optionally stratified by source.

    Returns list of (original_index, example) tuples.
    """
    examples = load_training_data(data_path)
    rng = random.Random(seed)

    if not stratify_by_source or n >= len(examples):
        indices = list(range(len(examples)))
        rng.shuffle(indices)
        selected = indices[:n]
        return [(i, examples[i]) for i in sorted(selected)]

    # Stratified: proportional sampling by source
    by_source: dict[str, list[int]] = {}
    for i, ex in enumerate(examples):
        src = ex.source or "unknown"
        by_source.setdefault(src, []).append(i)

    # Allocate samples proportionally, at least 1 per source
    total = len(examples)
    sampled: list[int] = []
    remaining = n

    sources = sorted(by_source.keys())
    for src in sources:
        src_indices = by_source[src]
        alloc = max(1, round(n * len(src_indices) / total))
        alloc = min(alloc, remaining, len(src_indices))
        rng.shuffle(src_indices)
        sampled.extend(src_indices[:alloc])
        remaining -= alloc
        if remaining <= 0:
            break

    # Fill remaining from all sources
    if remaining > 0:
        sampled_set = set(sampled)
        all_indices = [i for i in range(len(examples)) if i not in sampled_set]
        rng.shuffle(all_indices)
        sampled.extend(all_indices[:remaining])

    return [(i, examples[i]) for i in sorted(sampled)]


def format_for_review(examples: list[tuple[int, TrainingExample]]) -> str:
    """Format sampled examples as human/LLM-readable text."""
    lines: list[str] = []
    for idx, ex in examples:
        lines.append("=" * 70)
        lines.append(
            f"Example #{idx} | source={ex.source or 'unknown'} | scale={ex.scale or 'N/A'}"
        )
        lines.append("=" * 70)
        lines.append(f"QUERY:\n{ex.query}\n")
        lines.append(f"POSITIVE:\n{ex.positive}\n")
        if ex.hard_negatives:
            lines.append(f"HARD NEGATIVES ({len(ex.hard_negatives)}):")
            for i, neg in enumerate(ex.hard_negatives[:3]):
                lines.append(f"  [{i + 1}] {neg[:200]}...")
            if len(ex.hard_negatives) > 3:
                lines.append(f"  ... and {len(ex.hard_negatives) - 3} more")
        lines.append("")
    return "\n".join(lines)


def save_annotations(annotations: list[SpotCheckResult], path: Path) -> None:
    """Append annotations to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for ann in annotations:
            f.write(json.dumps(asdict(ann)) + "\n")


def load_annotations(path: Path) -> list[SpotCheckResult]:
    """Load annotations from JSONL file."""
    if not path.exists():
        return []
    results: list[SpotCheckResult] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                results.append(SpotCheckResult(**d))
    return results


def quality_report(annotations: list[SpotCheckResult]) -> str:
    """Generate per-source quality summary from annotations."""
    if not annotations:
        return "No annotations to report."

    lines = ["Training Data Quality Report", "=" * 40]

    by_source: dict[str, list[SpotCheckResult]] = {}
    for ann in annotations:
        by_source.setdefault(ann.source, []).append(ann)

    for source in sorted(by_source.keys()):
        anns = by_source[source]
        q_scores = [a.query_quality for a in anns]
        p_scores = [a.positive_quality for a in anns]
        lines.append(f"\n{source} (n={len(anns)}):")
        lines.append(f"  Query quality:    {sum(q_scores) / len(q_scores):.1f}/5")
        lines.append(f"  Positive quality: {sum(p_scores) / len(p_scores):.1f}/5")
        low_quality = [a for a in anns if a.query_quality <= 2 or a.positive_quality <= 2]
        if low_quality:
            lines.append(f"  Low quality:      {len(low_quality)}/{len(anns)}")

    # Overall
    all_q = [a.query_quality for a in annotations]
    all_p = [a.positive_quality for a in annotations]
    lines.append(f"\nOverall (n={len(annotations)}):")
    lines.append(f"  Query quality:    {sum(all_q) / len(all_q):.1f}/5")
    lines.append(f"  Positive quality: {sum(all_p) / len(all_p):.1f}/5")

    return "\n".join(lines)
