"""Dataset validation for training data JSONL files.

Checks format, quality constraints, and cross-split integrity.
Used by both `incite finetune validate` and pytest tests.
"""

from pathlib import Path

from incite.finetuning.data_preparation import TrainingExample, load_training_data

# Must match data_sources.py MIN_QUERY_LENGTH
MIN_QUERY_LENGTH = 50
MIN_POSITIVE_LENGTH = 30


def validate_example(ex: TrainingExample, index: int = 0) -> list[str]:
    """Validate a single TrainingExample. Returns list of issues."""
    issues = []

    if len(ex.query) < MIN_QUERY_LENGTH:
        issues.append(f"Line {index}: query too short ({len(ex.query)} chars)")

    if len(ex.positive) < MIN_POSITIVE_LENGTH:
        issues.append(f"Line {index}: positive too short ({len(ex.positive)} chars)")

    if not ex.query.strip():
        issues.append(f"Line {index}: query is empty after stripping")

    if not ex.positive.strip():
        issues.append(f"Line {index}: positive is empty after stripping")

    if ex.positive in ex.hard_negatives:
        issues.append(f"Line {index}: positive appears in hard_negatives")

    if len(ex.hard_negatives) != len(set(ex.hard_negatives)):
        issues.append(f"Line {index}: duplicate hard_negatives")

    # Note: passage_positive without intent is informational, not an error.
    # After passage data normalization, intent is useful metadata but not required.

    return issues


def validate_dataset(path: Path) -> list[str]:
    """Validate a training dataset JSONL file. Returns list of issues."""
    issues = []
    try:
        examples = load_training_data(path)
    except Exception as e:
        return [f"Failed to load {path}: {e}"]

    for i, ex in enumerate(examples):
        issues.extend(validate_example(ex, index=i))

    return issues


def validate_split_integrity(
    train_path: Path,
    dev_path: Path,
) -> list[str]:
    """Validate train/dev split has no source paper overlap or query leakage."""
    issues = []

    try:
        train = load_training_data(train_path)
        dev = load_training_data(dev_path)
    except Exception as e:
        return [f"Failed to load splits: {e}"]

    # Check source paper overlap
    train_sources = {ex.source_paper_id for ex in train if ex.source_paper_id}
    dev_sources = {ex.source_paper_id for ex in dev if ex.source_paper_id}
    overlap = train_sources & dev_sources
    if overlap:
        issues.append(
            f"Source paper overlap between train and dev: {len(overlap)} papers "
            f"(e.g., {list(overlap)[:3]})"
        )

    # Check query overlap
    train_queries = {ex.query for ex in train}
    dev_queries = {ex.query for ex in dev}
    query_overlap = train_queries & dev_queries
    if query_overlap:
        issues.append(f"Query overlap between train and dev: {len(query_overlap)} queries")

    # Check dev fraction is reasonable (5-25%)
    total = len(train) + len(dev)
    if total > 0:
        dev_frac = len(dev) / total
        if dev_frac < 0.05 or dev_frac > 0.25:
            issues.append(f"Dev fraction {dev_frac:.1%} outside expected range (5-25%)")

    return issues
