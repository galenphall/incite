#!/usr/bin/env python
"""Sweep hybrid retrieval configurations for any embedder.

Loads corpus/test set once, builds retrievers once, then evaluates many
hybrid configurations in a loop to find the best settings.

Usage:
    python scripts/hybrid_sweep.py
    python scripts/hybrid_sweep.py --quick                    # subset of configs
    python scripts/hybrid_sweep.py --scale narrow             # use narrow context
    python scripts/hybrid_sweep.py --embedder granite-ft      # sweep for Granite
"""

import argparse
import time

from incite.corpus.loader import load_corpus, load_test_set
from incite.evaluation import clean_test_set, evaluate_retrieval
from incite.retrieval.bm25 import BM25Retriever, tokenize_with_stemming
from incite.retrieval.factory import get_embedder
from incite.retrieval.hybrid import HybridRetriever
from incite.retrieval.neural import NeuralRetriever

CORPUS_PATH = "data/processed/corpus.jsonl"
TEST_SET_PATH = "data/processed/test_set.jsonl"


def run_eval(retriever, test_set, scale):
    """Run evaluation and return result dict."""
    result = evaluate_retrieval(retriever, test_set, k=50, scale=scale)
    return result


def main():
    parser = argparse.ArgumentParser(description="Sweep hybrid configurations")
    parser.add_argument("--quick", action="store_true", help="Run reduced config set")
    parser.add_argument("--scale", default="local", help="Context scale (default: local)")
    parser.add_argument("--embedder", default="minilm-ft", help="Embedder key (default: minilm-ft)")
    args = parser.parse_args()

    # --- Load data once ---
    print("Loading corpus...")
    papers = load_corpus(CORPUS_PATH)
    paper_dict = {p.id: p for p in papers}
    print(f"  {len(papers)} papers")

    print("Loading test set...")
    test_set = load_test_set(TEST_SET_PATH)
    test_set, stats = clean_test_set(test_set, paper_dict)
    if stats.total_removed > 0:
        print(f"  {stats}")
    print(f"  {len(test_set)} queries after cleaning")

    # --- Build retrievers once ---
    # Build neural index from corpus papers (not pre-built Zotero index, which
    # uses different paper IDs and would produce zero matches after ref-set filtering)
    print("\nBuilding neural retriever (embedding corpus papers)...")
    embedder = get_embedder(args.embedder)
    neural = NeuralRetriever.from_papers(papers, embedder, show_progress=True)

    print("Building BM25 retriever...")
    bm25 = BM25Retriever.from_papers(papers)

    print("Building BM25 retriever (stemmed)...")
    bm25_stemmed = BM25Retriever.from_papers(papers, tokenizer=tokenize_with_stemming)

    # --- Define configurations ---
    configs = []

    # 1. Neural-only baseline
    configs.append(("neural-only", neural, None))

    # 2. RRF k sweep (equal weights, BM25)
    rrf_ks = [5, 10, 15, 20, 30, 60]
    if args.quick:
        rrf_ks = [5, 10, 30]
    for k in rrf_ks:
        hybrid = HybridRetriever(retrievers=[(neural, 1.0), (bm25, 1.0)], fusion="rrf", rrf_k=k)
        configs.append((f"rrf k={k} (1:1)", hybrid, None))

    # 3. Weight ratio sweep (RRF, k=10 — current default)
    ratios = [(2.0, 1.0), (1.5, 1.0), (1.0, 1.0), (1.0, 1.5), (1.0, 2.0)]
    if args.quick:
        ratios = [(2.0, 1.0), (1.0, 1.0), (1.0, 2.0)]
    for nw, bw in ratios:
        hybrid = HybridRetriever(retrievers=[(neural, nw), (bm25, bw)], fusion="rrf", rrf_k=10)
        configs.append((f"rrf k=10 ({nw}:{bw})", hybrid, None))

    # 4. Weighted fusion (not RRF)
    for nw, bw in ratios:
        hybrid = HybridRetriever(retrievers=[(neural, nw), (bm25, bw)], fusion="weighted")
        configs.append((f"weighted ({nw}:{bw})", hybrid, None))

    # 5. BM25 with stemming (best RRF configs)
    stem_ks = [5, 10, 20] if not args.quick else [10]
    for k in stem_ks:
        hybrid = HybridRetriever(
            retrievers=[(neural, 1.0), (bm25_stemmed, 1.0)], fusion="rrf", rrf_k=k
        )
        configs.append((f"rrf k={k} stemmed (1:1)", hybrid, None))

    # Stemmed with best weight ratios
    for nw, bw in [(2.0, 1.0), (1.5, 1.0)]:
        hybrid = HybridRetriever(
            retrievers=[(neural, nw), (bm25_stemmed, bw)], fusion="rrf", rrf_k=10
        )
        configs.append((f"rrf k=10 stemmed ({nw}:{bw})", hybrid, None))

    # --- Run sweep ---
    print(f"\n{'=' * 80}")
    print(f"Running {len(configs)} configurations (scale={args.scale})")
    print(f"{'=' * 80}\n")

    results = []
    for i, (name, retriever, _) in enumerate(configs):
        t0 = time.time()
        if name == "neural-only":
            result = run_eval(neural, test_set, args.scale)
        else:
            result = run_eval(retriever, test_set, args.scale)
        elapsed = time.time() - t0

        d = result.to_dict()
        results.append((name, d, elapsed))
        print(
            f"[{i + 1}/{len(configs)}] {name:<35} "
            f"MRR={d['mrr']:.4f}  R@1={d['recall@1']:.4f}  "
            f"R@10={d['recall@10']:.4f}  R@20={d['recall@20']:.4f}  "
            f"({elapsed:.1f}s)"
        )

    # --- Summary table sorted by MRR ---
    print(f"\n{'=' * 80}")
    print("RESULTS SORTED BY MRR")
    print(f"{'=' * 80}")
    results.sort(key=lambda x: x[1]["mrr"], reverse=True)

    header = f"{'Config':<40} {'MRR':>7} {'R@1':>7} {'R@10':>7} {'R@20':>7} {'NDCG@10':>7}"
    print(header)
    print("-" * len(header))
    for name, d, _ in results:
        print(
            f"{name:<40} {d['mrr']:>7.4f} {d['recall@1']:>7.4f} "
            f"{d['recall@10']:>7.4f} {d['recall@20']:>7.4f} {d['ndcg@10']:>7.4f}"
        )

    # Highlight best
    best_name, best_d, _ = results[0]
    baseline_d = next(d for name, d, _ in results if name == "neural-only")
    delta_mrr = best_d["mrr"] - baseline_d["mrr"]
    delta_r1 = best_d["recall@1"] - baseline_d["recall@1"]
    print(f"\nBest: {best_name}")
    print(f"  MRR  {best_d['mrr']:.4f}  (delta vs neural-only: {delta_mrr:+.4f})")
    print(f"  R@1  {best_d['recall@1']:.4f}  (delta vs neural-only: {delta_r1:+.4f})")


if __name__ == "__main__":
    main()
