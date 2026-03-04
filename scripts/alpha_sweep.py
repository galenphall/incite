#!/usr/bin/env python3
"""Sweep alpha parameter for two-stage retrieval.

Finds the optimal alpha (paper vs chunk score blend weight) by evaluating
across a range from 0.0 to 1.0.

Usage:
    python scripts/alpha_sweep.py --embedder minilm-ft
    python scripts/alpha_sweep.py --embedder minilm-ft --steps 20 --output results/alpha_sweep.json

Requires:
    - Built corpus index (data/processed/index/)
    - Built chunk index (~/.incite/zotero_chunks_{embedder}/)
    - Test set (data/processed/test_set.jsonl)
"""

import argparse
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Sweep alpha for two-stage retrieval")
    parser.add_argument(
        "--embedder", type=str, default="minilm-ft", help="Embedder type (default: minilm-ft)"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/processed/corpus.jsonl",
        help="Path to corpus JSONL",
    )
    parser.add_argument(
        "--test-set",
        type=str,
        default="data/processed/test_set.jsonl",
        help="Path to test set JSONL",
    )
    parser.add_argument(
        "--index", type=str, default="data/processed/index", help="Path to paper FAISS index"
    )
    parser.add_argument(
        "--steps", type=int, default=11, help="Number of alpha values (default: 11)"
    )
    parser.add_argument(
        "--scale", type=str, default="narrow", help="Context scale (default: narrow)"
    )
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON results")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N queries")
    args = parser.parse_args()

    from incite.corpus.loader import load_chunks, load_corpus, load_test_set
    from incite.embeddings.chunk_store import ChunkStore
    from incite.evaluation.metrics import clean_test_set, evaluate_retrieval
    from incite.retrieval.factory import create_two_stage_retriever

    # Load data
    print(f"Loading corpus from {args.corpus}...")
    papers = load_corpus(args.corpus)
    paper_dict = {p.id: p for p in papers}
    print(f"Loaded {len(papers)} papers")

    print(f"Loading test set from {args.test_set}...")
    test_set = load_test_set(args.test_set)
    if args.limit:
        test_set = test_set[: args.limit]
    test_set, cleaning_stats = clean_test_set(test_set, paper_dict)
    if cleaning_stats.total_removed > 0:
        print(cleaning_stats)
    print(f"Using {len(test_set)} queries")

    # Load chunk store
    cache_dir = Path.home() / ".incite"
    chunk_index_path = cache_dir / f"zotero_chunks_{args.embedder}"
    if not (chunk_index_path / "index.faiss").exists():
        print(f"Error: Chunk index not found at {chunk_index_path}")
        print("Build with: incite index-chunks --embedder", args.embedder)
        sys.exit(1)

    print(f"Loading chunk store from {chunk_index_path}...")
    chunk_store = ChunkStore()
    chunk_store.load(chunk_index_path)

    # Load chunk texts
    chunks_jsonl = cache_dir / "zotero_chunks_paragraph.jsonl"
    chunk_dict = {}
    if chunks_jsonl.exists():
        raw_chunks = load_chunks(str(chunks_jsonl))
        chunk_dict = {c.id: c for c in raw_chunks}
        print(f"Loaded {len(chunk_dict)} chunks")

    # Also run paper-only baseline
    from incite.retrieval.factory import create_retriever

    print("\n--- Paper-only baseline ---")
    baseline_retriever = create_retriever(
        papers=papers,
        method="hybrid",
        embedder_type=args.embedder,
        index_path=Path(args.index) if Path(args.index).exists() else None,
        show_progress=True,
    )
    baseline_result = evaluate_retrieval(baseline_retriever, test_set, k=50, scale=args.scale)
    print(
        f"  MRR={baseline_result.mrr:.4f}  R@1={baseline_result.recall_at_1:.4f}  "
        f"R@10={baseline_result.recall_at_10:.4f}"
    )

    # Sweep alpha
    alphas = [i / (args.steps - 1) for i in range(args.steps)]
    results = []

    for alpha in alphas:
        print(f"\n--- alpha={alpha:.2f} ---")
        start = time.perf_counter()

        retriever = create_two_stage_retriever(
            papers=papers,
            chunk_store=chunk_store,
            chunks=chunk_dict,
            embedder_type=args.embedder,
            index_path=Path(args.index) if Path(args.index).exists() else None,
            alpha=alpha,
            show_progress=False,
        )

        result = evaluate_retrieval(retriever, test_set, k=50, scale=args.scale)
        elapsed = time.perf_counter() - start

        entry = {
            "alpha": alpha,
            "mrr": result.mrr,
            "recall_at_1": result.recall_at_1,
            "recall_at_5": result.recall_at_5,
            "recall_at_10": result.recall_at_10,
            "recall_at_20": result.recall_at_20,
            "concordance": result.concordance,
            "skill_mrr": result.skill_mrr,
            "elapsed_s": elapsed,
        }
        results.append(entry)
        print(
            f"  MRR={result.mrr:.4f}  R@1={result.recall_at_1:.4f}  "
            f"R@10={result.recall_at_10:.4f}  ({elapsed:.1f}s)"
        )

    # Find best alpha
    best = max(results, key=lambda r: r["mrr"])
    print(f"\n=== Best alpha: {best['alpha']:.2f} (MRR={best['mrr']:.4f}) ===")

    # Print comparison table
    print(f"\n{'Alpha':<8}{'MRR':<10}{'R@1':<10}{'R@5':<10}{'R@10':<10}{'R@20':<10}")
    print("-" * 58)
    # Baseline row
    print(
        f"{'base':<8}{baseline_result.mrr:<10.4f}{baseline_result.recall_at_1:<10.4f}"
        f"{baseline_result.recall_at_5:<10.4f}{baseline_result.recall_at_10:<10.4f}"
        f"{baseline_result.recall_at_20:<10.4f}"
    )
    for r in results:
        marker = " *" if r["alpha"] == best["alpha"] else ""
        print(
            f"{r['alpha']:<8.2f}{r['mrr']:<10.4f}{r['recall_at_1']:<10.4f}"
            f"{r['recall_at_5']:<10.4f}{r['recall_at_10']:<10.4f}"
            f"{r['recall_at_20']:<10.4f}{marker}"
        )

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        full_output = {
            "embedder": args.embedder,
            "scale": args.scale,
            "num_queries": len(test_set),
            "baseline": baseline_result.to_dict(),
            "best_alpha": best["alpha"],
            "sweep": results,
        }
        output_path.write_text(json.dumps(full_output, indent=2))
        print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
