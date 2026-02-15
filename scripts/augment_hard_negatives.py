#!/usr/bin/env python3
"""Augment training data with hard negatives from the current model.

Three complementary mining approaches:
  Phase A -- Neural: Uses Granite-FT v6 embeddings to find semantically similar
      but incorrect papers as hard negatives.  Reuses mine_hard_negatives()
      from quality.py.
  Phase B -- BM25: Builds a BM25 index over all unique positives and finds
      lexically similar confounders.
  Phase C -- Random: Samples random positives from other examples as easy
      negatives to prevent overspecialization on hard cases.

Together, these bring hard negative coverage from ~73% to ~99%.

Usage:
    # Neural-only (recommended — uses Granite-FT v6 with asymmetric prefixes):
    python scripts/augment_hard_negatives.py \\
        --input data/finetuning/master_train.jsonl \\
        --output data/finetuning/master_train_augmented.jsonl \\
        --skip-bm25

    # All three (neural + BM25 + random):
    python scripts/augment_hard_negatives.py \\
        --input data/finetuning/master_train.jsonl \\
        --output data/finetuning/master_train_augmented.jsonl \\
        --neural-negatives 5 --bm25-negatives 3 --random-negatives 2

    # BM25-only (no GPU needed):
    python scripts/augment_hard_negatives.py \\
        --input data/finetuning/master_train.jsonl \\
        --output data/finetuning/master_train_augmented.jsonl \\
        --skip-neural --bm25-negatives 5
"""

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def mine_bm25_hard_negatives(
    examples: list,
    num_negatives: int = 3,
    show_progress: bool = True,
    batch_size: int = 5000,
) -> dict:
    """Mine BM25-based hard negatives using vectorized TF-IDF sparse ops.

    Uses scikit-learn's TfidfVectorizer with stemming to build sparse matrices
    for queries and positives, then performs batch matrix multiplication for
    fast scoring (~3 min instead of ~45 hours for 170K queries x 125K docs).

    Args:
        examples: List of TrainingExample objects (modified in place)
        num_negatives: Number of BM25 negatives to add per example
        show_progress: Show progress bars
        batch_size: Number of queries to score per batch

    Returns:
        Stats dict
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from tqdm import tqdm

    from incite.models import strip_metadata_prefix
    from incite.retrieval.bm25 import tokenize_with_stemming

    # Build corpus of unique positives
    positive_texts: list[str] = []
    positive_indices: dict[str, int] = {}
    for ex in examples:
        key = ex.positive.strip()
        if key not in positive_indices:
            positive_indices[key] = len(positive_texts)
            positive_texts.append(key)

    print(f"  Unique positives for TF-IDF: {len(positive_texts):,}")

    # Custom tokenizer that uses the existing stemming pipeline
    def stemming_tokenizer(text):
        return tokenize_with_stemming(text)

    # Build TF-IDF vectorizer over the positives corpus
    print("  Building TF-IDF index over positives...")
    vectorizer = TfidfVectorizer(
        tokenizer=stemming_tokenizer,
        token_pattern=None,
        sublinear_tf=True,
        norm="l2",
        max_features=200_000,
    )
    corpus_matrix = vectorizer.fit_transform(
        tqdm(positive_texts, desc="  Vectorizing corpus") if show_progress else positive_texts
    )
    print(f"  Corpus matrix: {corpus_matrix.shape}, nnz={corpus_matrix.nnz:,}")

    # Pre-tokenize all queries into strings for the vectorizer
    print("  Vectorizing queries...")
    query_texts = [ex.query for ex in examples]
    query_matrix = vectorizer.transform(
        tqdm(query_texts, desc="  Vectorizing queries") if show_progress else query_texts
    )
    print(f"  Query matrix: {query_matrix.shape}, nnz={query_matrix.nnz:,}")

    # Mine negatives in batches using sparse matrix multiplication
    stats = {
        "examples_augmented": 0,
        "negatives_added": 0,
    }

    n_examples = len(examples)
    n_batches = (n_examples + batch_size - 1) // batch_size
    top_k = 20  # retrieve top 20 candidates per query

    print(f"  Scoring in {n_batches} batches of {batch_size}...")
    for batch_idx in (
        tqdm(range(n_batches), desc="  Batch scoring") if show_progress else range(n_batches)
    ):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_examples)
        batch_queries = query_matrix[start:end]

        # Sparse matrix multiply: (batch, vocab) @ (vocab, corpus) = (batch, corpus)
        scores = (batch_queries @ corpus_matrix.T).toarray()

        for i in range(end - start):
            ex = examples[start + i]
            existing_neg_set = {n.strip() for n in ex.hard_negatives}
            existing_neg_cores = {strip_metadata_prefix(n.strip()) for n in ex.hard_negatives}
            actual_positive = ex.positive.strip()
            actual_positive_core = strip_metadata_prefix(actual_positive)

            # Get top-k candidates
            row_scores = scores[i]
            top_indices = np.argpartition(row_scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(row_scores[top_indices])][::-1]

            mined: list[str] = []
            for idx in top_indices:
                if len(mined) >= num_negatives:
                    break
                candidate = positive_texts[idx]
                candidate_stripped = candidate.strip()
                candidate_core = strip_metadata_prefix(candidate_stripped)
                if candidate_stripped == actual_positive or candidate_core == actual_positive_core:
                    continue
                if candidate_stripped in existing_neg_set or candidate_core in existing_neg_cores:
                    continue
                mined.append(candidate)

            if mined:
                ex.hard_negatives = list(ex.hard_negatives) + mined
                stats["examples_augmented"] += 1
                stats["negatives_added"] += len(mined)

    return stats


def add_random_negatives(
    examples: list,
    num_negatives: int = 2,
    seed: int = 42,
) -> dict:
    """Add random negatives sampled from the pool of all unique positives.

    Random (easy) negatives prevent overspecialization on hard cases and
    improve generalization when the reranker sees retriever candidates that
    are far from the query.

    Args:
        examples: List of TrainingExample objects (modified in place)
        num_negatives: Number of random negatives to add per example
        seed: Random seed for reproducibility

    Returns:
        Stats dict
    """
    import random as _random

    from incite.models import strip_metadata_prefix

    rng = _random.Random(seed)

    # Build pool of all unique positives
    pool: list[str] = []
    pool_cores: dict[str, str] = {}
    seen: set[str] = set()
    for ex in examples:
        key = ex.positive.strip()
        if key not in seen:
            seen.add(key)
            pool.append(key)
            pool_cores[key] = strip_metadata_prefix(key)

    print(f"  Random negative pool: {len(pool):,} unique positives")

    stats = {"examples_augmented": 0, "negatives_added": 0}

    # Oversample 3x to handle dedup filtering
    sample_size = num_negatives * 3

    for ex in examples:
        actual_positive = ex.positive.strip()
        actual_positive_core = strip_metadata_prefix(actual_positive)
        existing_neg_set = {n.strip() for n in ex.hard_negatives}
        existing_neg_cores = {strip_metadata_prefix(n.strip()) for n in ex.hard_negatives}

        candidates = rng.sample(pool, min(sample_size, len(pool)))
        mined: list[str] = []
        for candidate in candidates:
            if len(mined) >= num_negatives:
                break
            candidate_core = pool_cores[candidate]
            if candidate == actual_positive or candidate_core == actual_positive_core:
                continue
            if candidate in existing_neg_set or candidate_core in existing_neg_cores:
                continue
            mined.append(candidate)

        if mined:
            ex.hard_negatives = list(ex.hard_negatives) + mined
            stats["examples_augmented"] += 1
            stats["negatives_added"] += len(mined)

    return stats


def cap_hard_negatives(examples: list, max_total: int = 10) -> int:
    """Cap total hard negatives per example.

    Args:
        examples: List of TrainingExample objects (modified in place)
        max_total: Maximum hard negatives to keep per example

    Returns:
        Number of examples that were capped
    """
    capped = 0
    for ex in examples:
        if len(ex.hard_negatives) > max_total:
            ex.hard_negatives = ex.hard_negatives[:max_total]
            capped += 1
    return capped


def main():
    parser = argparse.ArgumentParser(description="Augment training data with hard negatives")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL training file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file with augmented hard negatives",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/granite-citation-v6/final",
        help="Path to trained model for neural mining (default: Granite-FT v6)",
    )
    parser.add_argument(
        "--query-prefix",
        type=str,
        default="query: ",
        help="Prefix for queries (default: 'query: ' for Granite)",
    )
    parser.add_argument(
        "--passage-prefix",
        type=str,
        default="passage: ",
        help="Prefix for passages (default: 'passage: ' for Granite)",
    )
    parser.add_argument(
        "--neural-negatives",
        type=int,
        default=5,
        help="Number of neural hard negatives to add (default: 5)",
    )
    parser.add_argument(
        "--bm25-negatives",
        type=int,
        default=3,
        help="Number of BM25 hard negatives to add (default: 3)",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=10,
        help="Maximum total hard negatives per example (default: 10)",
    )
    parser.add_argument(
        "--skip-neural",
        action="store_true",
        help="Skip neural mining (BM25 only)",
    )
    parser.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Skip BM25 mining (neural only)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for neural encoding (default: cpu — MPS risks OOM on laptops)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Encode batch size for neural mining (default: 64, lower if OOM)",
    )
    parser.add_argument(
        "--random-negatives",
        type=int,
        default=0,
        help="Number of random negatives to add per example (default: 0, disabled)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(2, (os.cpu_count() or 4) // 2),
        help="Max threads for numpy/torch/OpenBLAS (default: half of available cores)",
    )
    args = parser.parse_args()

    # Limit CPU threads to prevent thermal throttling / OOM kills
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

    from incite.finetuning.types import load_training_data

    # Load examples
    print(f"Loading {args.input}...")
    examples = load_training_data(args.input)
    print(f"Loaded {len(examples):,} examples")

    has_negs_before = sum(1 for ex in examples if ex.hard_negatives)
    avg_negs_before = sum(len(ex.hard_negatives) for ex in examples) / max(1, len(examples))
    print(
        f"Before: {has_negs_before:,} ({100 * has_negs_before / max(1, len(examples)):.1f}%) "
        f"have hard negatives, avg {avg_negs_before:.1f}/example"
    )

    # Phase A: Neural mining
    if not args.skip_neural:
        model_path = Path(args.model)
        # Accept both local paths and HuggingFace model IDs (contain '/')
        is_hf_model = "/" in args.model and not model_path.exists()
        if not model_path.exists() and not is_hf_model:
            print(f"\nWARNING: Model not found at {model_path}")
            print("Skipping neural mining. Use --skip-neural to suppress this warning.")
        else:
            print(f"\n{'=' * 60}")
            print("Phase A: Neural Hard Negative Mining")
            print(f"{'=' * 60}")
            # Write to temp file, then read back augmented
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as tmp:
                tmp_path = Path(tmp.name)
                for ex in examples:
                    tmp.write(json.dumps(ex.to_dict()) + "\n")

            from incite.finetuning.quality import mine_hard_negatives

            neural_stats = mine_hard_negatives(
                input_path=tmp_path,
                output_path=tmp_path,  # overwrite in-place
                model_path=str(args.model),
                num_negatives=args.neural_negatives,
                batch_size=args.batch_size,
                query_prefix=args.query_prefix,
                passage_prefix=args.passage_prefix,
                device=args.device,
            )
            print(f"  Neural stats: {neural_stats}")

            # Reload augmented examples
            examples = load_training_data(tmp_path)
            tmp_path.unlink()

    # Phase B: BM25 mining
    if not args.skip_bm25:
        print(f"\n{'=' * 60}")
        print("Phase B: BM25 Hard Negative Mining")
        print(f"{'=' * 60}")
        bm25_stats = mine_bm25_hard_negatives(examples, num_negatives=args.bm25_negatives)
        print(f"  BM25 augmented: {bm25_stats['examples_augmented']:,}")
        print(f"  BM25 negatives added: {bm25_stats['negatives_added']:,}")

    # Phase C: Random negatives
    if args.random_negatives > 0:
        print(f"\n{'=' * 60}")
        print("Phase C: Random Negative Sampling")
        print(f"{'=' * 60}")
        random_stats = add_random_negatives(examples, num_negatives=args.random_negatives)
        print(f"  Random augmented: {random_stats['examples_augmented']:,}")
        print(f"  Random negatives added: {random_stats['negatives_added']:,}")

    # Remove contradictory negatives (positive text appearing in hard_negatives).
    # Also detects format-variant duplicates: the same paper text with vs without
    # metadata prefix (title/authors/year/journal), which BM25 mining introduces
    # when positives from different sources use different formats.
    from incite.models import strip_metadata_prefix as _strip_metadata_prefix

    contradictions_removed = 0
    for ex in examples:
        positive_stripped = ex.positive.strip()
        positive_core = _strip_metadata_prefix(positive_stripped)
        filtered = [
            n
            for n in ex.hard_negatives
            if n.strip() != positive_stripped and _strip_metadata_prefix(n.strip()) != positive_core
        ]
        if len(filtered) < len(ex.hard_negatives):
            contradictions_removed += len(ex.hard_negatives) - len(filtered)
            ex.hard_negatives = filtered
    if contradictions_removed:
        print(
            f"\nRemoved {contradictions_removed:,} contradictory negatives "
            f"(exact match or format-variant duplicate)"
        )

    # Cap total hard negatives
    capped = cap_hard_negatives(examples, max_total=args.max_total)
    if capped:
        print(f"\nCapped {capped:,} examples to {args.max_total} hard negatives")

    # Summary
    has_negs_after = sum(1 for ex in examples if ex.hard_negatives)
    avg_negs_after = sum(len(ex.hard_negatives) for ex in examples) / max(1, len(examples))
    print(
        f"\nAfter: {has_negs_after:,} ({100 * has_negs_after / max(1, len(examples)):.1f}%) "
        f"have hard negatives, avg {avg_negs_after:.1f}/example"
    )

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict()) + "\n")
    print(f"\nWrote {len(examples):,} examples to {args.output}")


if __name__ == "__main__":
    main()
