"""Evaluation metrics for retrieval."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from incite.evaluation.failure_analysis import DiagnosisResult

import numpy as np

from incite.embeddings.base import BaseEmbedder
from incite.interfaces import Reranker, Retriever
from incite.models import CitationContext, EvaluationResult, Paper, QueryResult, RetrievalResult

logger = logging.getLogger(__name__)

# Minimum abstract length to consider a GT paper findable by embedding.
_MIN_ABSTRACT_LENGTH = 30


@dataclass
class CleaningStats:
    """Statistics from test set cleaning."""

    original_queries: int = 0
    cleaned_queries: int = 0
    removed_degenerate_gt: int = 0
    removed_domain_mismatch: int = 0
    removed_duplicate_queries: int = 0
    degenerate_paper_ids: list[str] = field(default_factory=list)
    mismatch_source_ids: list[str] = field(default_factory=list)

    @property
    def total_removed(self) -> int:
        return self.original_queries - self.cleaned_queries

    def __str__(self) -> str:
        lines = [
            f"Cleaned {self.original_queries} → {self.cleaned_queries} queries "
            f"({self.total_removed} removed):"
        ]
        if self.removed_degenerate_gt:
            lines.append(
                f"  Degenerate GT papers: {self.removed_degenerate_gt} queries "
                f"({len(self.degenerate_paper_ids)} papers)"
            )
        if self.removed_domain_mismatch:
            lines.append(
                f"  Domain mismatch: {self.removed_domain_mismatch} queries "
                f"({len(self.mismatch_source_ids)} source papers)"
            )
        if self.removed_duplicate_queries:
            lines.append(f"  Duplicate queries: {self.removed_duplicate_queries}")
        return "\n".join(lines)


def clean_test_set(
    test_set: list[CitationContext],
    papers: dict[str, Paper],
    remove_degenerate: bool = True,
    remove_duplicates: bool = True,
    min_abstract_length: int = _MIN_ABSTRACT_LENGTH,
) -> tuple[list[CitationContext], CleaningStats]:
    """Remove anomalous queries from a test set for cleaner evaluation.

    Applies three filters:
    1. Degenerate GT papers: Removes queries whose ground truth papers have
       abstracts shorter than min_abstract_length (placeholder/stub metadata
       from OpenAlex that makes embedding-based retrieval impossible).
    2. Domain mismatch: Removes queries from source papers whose GT papers
       are in a completely unrelated domain (detected by checking if none of
       the GT papers appear in any other source paper's reference set).
    3. Duplicate queries: Removes exact duplicates where the same
       (local_context, ground_truth_ids) pair appears more than once.

    Args:
        test_set: List of CitationContext objects
        papers: Dict mapping paper_id to Paper objects (corpus)
        remove_degenerate: If True, remove queries with degenerate GT papers
        remove_duplicates: If True, remove exact duplicate queries
        min_abstract_length: Minimum abstract length for GT papers

    Returns:
        Tuple of (cleaned_test_set, stats)
    """
    stats = CleaningStats(original_queries=len(test_set))
    result = list(test_set)

    if remove_degenerate:
        result, stats = _remove_degenerate_gt(result, papers, stats, min_abstract_length)

    # Domain mismatch: detect source papers whose GT papers share zero overlap
    # with any other source paper's reference set. This catches cases like
    # 0807.4043 (cosmology paper mapped to bioinformatics GT papers).
    result, stats = _remove_domain_mismatches(result, stats)

    if remove_duplicates:
        result, stats = _remove_duplicate_queries(result, stats)

    stats.cleaned_queries = len(result)
    return result, stats


def _remove_degenerate_gt(
    test_set: list[CitationContext],
    papers: dict[str, Paper],
    stats: CleaningStats,
    min_abstract_length: int,
) -> tuple[list[CitationContext], CleaningStats]:
    """Remove queries whose GT papers have degenerate metadata."""
    # Build set of degenerate paper IDs
    degenerate_ids: set[str] = set()
    for paper in papers.values():
        abstract = (paper.abstract or "").strip()
        if len(abstract) < min_abstract_length:
            degenerate_ids.add(paper.id)

    cleaned = []
    removed = 0
    flagged_papers: set[str] = set()
    for ctx in test_set:
        # Remove if ALL GT papers are degenerate (query is impossible)
        gt_set = set(ctx.ground_truth_ids)
        if gt_set and gt_set.issubset(degenerate_ids):
            removed += 1
            flagged_papers.update(gt_set)
        else:
            cleaned.append(ctx)

    stats.removed_degenerate_gt = removed
    stats.degenerate_paper_ids = sorted(flagged_papers)
    return cleaned, stats


def _remove_domain_mismatches(
    test_set: list[CitationContext],
    stats: CleaningStats,
) -> tuple[list[CitationContext], CleaningStats]:
    """Remove source papers whose GT papers don't appear in any other source's refs.

    A source paper is flagged if the union of its GT paper IDs has zero overlap
    with the union of all other source papers' reference sets. This detects
    catastrophic ID resolution errors (e.g., cosmology paper mapped to
    bioinformatics GT papers).
    """
    # Build per-source-paper sets
    from collections import defaultdict as _dd

    source_gt: dict[str, set[str]] = _dd(set)
    source_refs: dict[str, set[str]] = _dd(set)

    for ctx in test_set:
        sp = ctx.source_paper_id
        if sp:
            source_gt[sp].update(ctx.ground_truth_ids)
            source_refs[sp].update(ctx.reference_set_ids)

    # Need at least 2 source papers to detect cross-paper anomalies
    if len(source_gt) < 2:
        return test_set, stats

    # Find source papers whose GT papers are completely absent from all
    # OTHER source papers' reference sets
    mismatch_sources: set[str] = set()
    for sp, gt_ids in source_gt.items():
        if not gt_ids:
            continue
        # Build union of ALL other source papers' reference sets
        other_refs: set[str] = set()
        for other_sp, other_ref_set in source_refs.items():
            if other_sp != sp:
                other_refs.update(other_ref_set)
        if not other_refs:
            continue  # Can't compare if no other refs exist
        if not gt_ids.intersection(other_refs):
            # Only flag if ref set is very small — large ref sets with unique
            # GT may just be specialized, not erroneous
            if len(source_refs[sp]) <= 5:
                mismatch_sources.add(sp)

    if not mismatch_sources:
        return test_set, stats

    cleaned = [ctx for ctx in test_set if ctx.source_paper_id not in mismatch_sources]
    stats.removed_domain_mismatch = len(test_set) - len(cleaned)
    stats.mismatch_source_ids = sorted(mismatch_sources)
    return cleaned, stats


def _remove_duplicate_queries(
    test_set: list[CitationContext],
    stats: CleaningStats,
) -> tuple[list[CitationContext], CleaningStats]:
    """Remove exact duplicate queries (same local_context + ground_truth_ids)."""
    seen: set[tuple[str, frozenset[str]]] = set()
    cleaned = []
    removed = 0

    for ctx in test_set:
        key = (ctx.local_context, frozenset(ctx.ground_truth_ids))
        if key in seen:
            removed += 1
        else:
            seen.add(key)
            cleaned.append(ctx)

    stats.removed_duplicate_queries = removed
    return cleaned, stats


def _get_embedder(retriever: Retriever) -> Optional[BaseEmbedder]:
    """Extract embedder from a retriever chain (Neural, Hybrid, TwoStage, or Paragraph)."""
    if hasattr(retriever, "embedder"):
        return retriever.embedder
    # TwoStageRetriever wraps a paper_retriever
    if hasattr(retriever, "paper_retriever"):
        return _get_embedder(retriever.paper_retriever)
    if hasattr(retriever, "paragraph_retriever"):
        return _get_embedder(retriever.paragraph_retriever)
    if hasattr(retriever, "retrievers"):
        for r, _ in retriever.retrievers:
            emb = _get_embedder(r)
            if emb is not None:
                return emb
    return None


def _harmonic_number(n: int) -> float:
    """Compute the n-th harmonic number H(n) = sum(1/k for k=1..n)."""
    return sum(1.0 / k for k in range(1, n + 1))


def concordance_index(rank: Optional[int], reference_set_size: int) -> float:
    """Compute concordance index (C-index) for a single query.

    Measures the probability that the correct paper is ranked above a
    randomly-chosen incorrect paper. Normalizes for reference set size,
    making scores comparable across queries with different corpus sizes.

    Args:
        rank: 1-indexed rank of first relevant result, or None if not found
        reference_set_size: Total number of candidates (N)

    Returns:
        C-index in [0, 1]. 1.0 = perfect (rank 1), 0.5 = random, 0.0 = worst.
    """
    if reference_set_size <= 1:
        return 1.0
    if rank is None:
        return 0.0
    rank = min(rank, reference_set_size)
    return (reference_set_size - rank) / (reference_set_size - 1)


def skill_adjusted_mrr(rank: Optional[int], reference_set_size: int) -> float:
    """Compute skill-adjusted MRR for a single query.

    Normalizes MRR relative to the random baseline for the given reference
    set size. A skill of 0 means random performance, 1 means perfect.

    The random expected MRR for a uniformly-random ranking over N candidates
    is H(N)/N, where H(N) is the N-th harmonic number.

    Args:
        rank: 1-indexed rank of first relevant result, or None if not found
        reference_set_size: Total number of candidates (N)

    Returns:
        Skill score. 0.0 = random, 1.0 = perfect, negative = worse than random.
    """
    if reference_set_size <= 1:
        return 1.0
    n = reference_set_size
    expected_mrr = _harmonic_number(n) / n
    observed_mrr = 0.0 if rank is None else 1.0 / rank
    denominator = 1.0 - expected_mrr
    if denominator <= 0:
        return 0.0
    return (observed_mrr - expected_mrr) / denominator


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Calculate Recall@k.

    Args:
        retrieved_ids: List of retrieved document IDs (in rank order)
        relevant_ids: List of relevant document IDs
        k: Cutoff rank

    Returns:
        Recall@k score (0-1)
    """
    if not relevant_ids:
        return 0.0

    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    return len(retrieved_set & relevant_set) / len(relevant_set)


def mean_reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Calculate Mean Reciprocal Rank (MRR).

    Args:
        retrieved_ids: List of retrieved document IDs (in rank order)
        relevant_ids: List of relevant document IDs

    Returns:
        MRR score (0-1)
    """
    relevant_set = set(relevant_ids)

    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k.

    Args:
        retrieved_ids: List of retrieved document IDs (in rank order)
        relevant_ids: List of relevant document IDs (binary relevance)
        k: Cutoff rank

    Returns:
        NDCG@k score (0-1)
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)

    # Calculate DCG
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], 1):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    # Calculate ideal DCG
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant_ids))))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_single(
    retrieved: list[RetrievalResult],
    ground_truth_ids: list[str],
    reference_set_size: Optional[int] = None,
) -> dict[str, float]:
    """Evaluate a single query.

    Args:
        retrieved: List of retrieval results
        ground_truth_ids: List of relevant document IDs
        reference_set_size: Total candidates for this query (enables
            corpus-size-adjusted metrics when provided)

    Returns:
        Dict of metric scores
    """
    retrieved_ids = [r.paper_id for r in retrieved]

    scores = {
        "recall@1": recall_at_k(retrieved_ids, ground_truth_ids, 1),
        "recall@5": recall_at_k(retrieved_ids, ground_truth_ids, 5),
        "recall@10": recall_at_k(retrieved_ids, ground_truth_ids, 10),
        "recall@20": recall_at_k(retrieved_ids, ground_truth_ids, 20),
        "recall@50": recall_at_k(retrieved_ids, ground_truth_ids, 50),
        "mrr": mean_reciprocal_rank(retrieved_ids, ground_truth_ids),
        "ndcg@10": ndcg_at_k(retrieved_ids, ground_truth_ids, 10),
    }

    if reference_set_size is not None and reference_set_size > 1:
        relevant_set = set(ground_truth_ids)
        first_rank = None
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                first_rank = i
                break
        scores["concordance"] = concordance_index(first_rank, reference_set_size)
        scores["skill_mrr"] = skill_adjusted_mrr(first_rank, reference_set_size)

    return scores


def _compute_first_relevant_rank(
    results: list[RetrievalResult],
    ground_truth_ids: list[str],
) -> Optional[int]:
    """Find the 1-indexed rank of the first relevant result, or None."""
    gt_set = set(ground_truth_ids)
    for rank, r in enumerate(results, 1):
        if r.paper_id in gt_set:
            return rank
    return None


def evaluate_retrieval(
    retriever: Retriever,
    test_set: list[CitationContext],
    k: int = 50,
    scale: str = "local",
    use_reference_sets: bool = True,
    prefix_section: bool = False,
    macro_average: bool = False,
) -> EvaluationResult:
    """Evaluate retriever on a test set.

    Args:
        retriever: Retriever to evaluate
        test_set: List of CitationContext objects with ground_truth_ids
        k: Number of results to retrieve per query
        scale: Context scale to use for queries
        use_reference_sets: If True, filter results to only include papers
            in each context's reference_set_ids. This makes evaluation more
            realistic by only searching among papers the source actually cites.
        prefix_section: If True, prepend section heading to queries
        macro_average: If True, average within each source paper first, then
            average across papers. This corrects for source-paper skew.

    Returns:
        Aggregated EvaluationResult (with per_query list attached)
    """
    # Pre-compute all query embeddings in batch (major speedup)
    embedder = _get_embedder(retriever)
    if embedder is not None:
        valid_contexts = [c for c in test_set if c.ground_truth_ids]
        queries = [
            c.get_query(scale, clean=True, prefix_section=prefix_section) for c in valid_contexts
        ]
        print(f"Pre-computing {len(queries)} query embeddings in batch...")
        embedder.precompute_queries(queries, show_progress=True)

    query_results: list[QueryResult] = []
    # Collect per-query retrieval results for two-stage metric computation
    all_retrieval_results: list[tuple[list[str], list[RetrievalResult]]] = []

    for context in test_set:
        if not context.ground_truth_ids:
            continue

        # Retrieve more results if we'll be filtering
        retrieve_k = k
        if use_reference_sets and context.reference_set_ids:
            # Retrieve extra to ensure we have k after filtering
            retrieve_k = min(k * 3, 200)

        results = retriever.retrieve_for_context(
            context, k=retrieve_k, scale=scale, prefix_section=prefix_section
        )

        # Filter to reference set if enabled
        if use_reference_sets and context.reference_set_ids:
            ref_set = set(context.reference_set_ids)
            results = [r for r in results if r.paper_id in ref_set]

        ref_set_size = len(context.reference_set_ids) if context.reference_set_ids else None
        scores = evaluate_single(results, context.ground_truth_ids, ref_set_size)
        first_rank = _compute_first_relevant_rank(results, context.ground_truth_ids)

        all_retrieval_results.append((context.ground_truth_ids, results))

        query_results.append(
            QueryResult(
                context_id=context.id,
                source_paper_id=context.source_paper_id,
                ground_truth_ids=context.ground_truth_ids,
                scores=scores,
                first_relevant_rank=first_rank,
            )
        )

    # Clean up query cache
    if embedder is not None:
        embedder.clear_query_cache()

    if not query_results:
        return EvaluationResult(num_queries=0)

    # Compute aggregates
    metric_keys = [
        "recall@1",
        "recall@5",
        "recall@10",
        "recall@20",
        "recall@50",
        "mrr",
        "ndcg@10",
        "concordance",
        "skill_mrr",
    ]

    if macro_average:
        # Group by source_paper_id, average within, then average across
        by_paper: dict[Optional[str], list[QueryResult]] = defaultdict(list)
        for qr in query_results:
            by_paper[qr.source_paper_id].append(qr)

        paper_means: dict[str, list[float]] = {k: [] for k in metric_keys}
        for paper_qrs in by_paper.values():
            for key in metric_keys:
                paper_mean = sum(qr.scores.get(key, 0.0) for qr in paper_qrs) / len(paper_qrs)
                paper_means[key].append(paper_mean)

        avg = {k: sum(v) / len(v) for k, v in paper_means.items()}
    else:
        n = len(query_results)
        avg = {k: sum(qr.scores.get(k, 0.0) for qr in query_results) / n for k in metric_keys}

    # Compute two-stage metrics if results contain score_breakdown data
    evidence_coverage = 0.0
    mean_best_chunk_score = 0.0
    two_stage_detected = False

    correct_papers_total = 0
    correct_with_evidence = 0
    chunk_score_sum = 0.0
    chunk_score_count = 0

    for gt_ids, results in all_retrieval_results:
        gt_set = set(gt_ids)
        for r in results:
            if r.paper_id not in gt_set:
                continue
            # Check for two-stage score_breakdown keys
            if "best_chunk_score" not in r.score_breakdown:
                continue
            two_stage_detected = True
            correct_papers_total += 1
            if r.matched_paragraphs:
                correct_with_evidence += 1
            bcs = r.score_breakdown["best_chunk_score"]
            chunk_score_sum += bcs
            chunk_score_count += 1

    if two_stage_detected and correct_papers_total > 0:
        evidence_coverage = correct_with_evidence / correct_papers_total
        mean_best_chunk_score = chunk_score_sum / chunk_score_count

    return EvaluationResult(
        recall_at_1=avg["recall@1"],
        recall_at_5=avg["recall@5"],
        recall_at_10=avg["recall@10"],
        recall_at_20=avg["recall@20"],
        recall_at_50=avg["recall@50"],
        mrr=avg["mrr"],
        ndcg_at_10=avg["ndcg@10"],
        concordance=avg.get("concordance", 0.0),
        skill_mrr=avg.get("skill_mrr", 0.0),
        num_queries=len(query_results),
        per_query=query_results,
        evidence_coverage=evidence_coverage,
        mean_best_chunk_score=mean_best_chunk_score,
    )


def evaluate_retrieval_stratified(
    retriever: Retriever,
    test_set: list[CitationContext],
    papers: dict[str, Paper],
    k: int = 50,
    scale: str = "local",
    use_reference_sets: bool = True,
    prefix_section: bool = False,
) -> dict[str, EvaluationResult]:
    """Evaluate retriever with results stratified by ground truth paper type.

    Splits queries into two subsets based on whether the ground truth paper
    has full text (multiple chunks) or is abstract-only (1 chunk). Reports
    metrics for each subset separately to measure fairness gap.

    Args:
        retriever: Retriever to evaluate
        test_set: List of CitationContext objects with ground_truth_ids
        papers: Dict mapping paper_id -> Paper (used to check has_full_text)
        k: Number of results to retrieve per query
        scale: Context scale to use for queries
        use_reference_sets: If True, filter results to reference_set_ids
        prefix_section: If True, prepend section heading to queries

    Returns:
        Dict with keys "overall", "full_text", "abstract_only", each an EvaluationResult.
    """
    # Pre-compute query embeddings in batch
    embedder = _get_embedder(retriever)
    if embedder is not None:
        valid_contexts = [c for c in test_set if c.ground_truth_ids]
        queries = [
            c.get_query(scale, clean=True, prefix_section=prefix_section) for c in valid_contexts
        ]
        print(f"Pre-computing {len(queries)} query embeddings in batch...")
        embedder.precompute_queries(queries, show_progress=True)

    # Accumulators for each stratum
    metrics_list = [
        "recall@1",
        "recall@5",
        "recall@10",
        "recall@20",
        "recall@50",
        "mrr",
        "ndcg@10",
        "concordance",
        "skill_mrr",
    ]
    strata = {
        "overall": {"sum": {m: 0.0 for m in metrics_list}, "n": 0},
        "full_text": {"sum": {m: 0.0 for m in metrics_list}, "n": 0},
        "abstract_only": {"sum": {m: 0.0 for m in metrics_list}, "n": 0},
    }

    for context in test_set:
        if not context.ground_truth_ids:
            continue

        # Determine stratum: check if ground truth paper has full text
        gt_id = context.ground_truth_ids[0]
        gt_paper = papers.get(gt_id)
        if gt_paper and gt_paper.has_full_text:
            stratum = "full_text"
        else:
            stratum = "abstract_only"

        # Retrieve
        retrieve_k = k
        if use_reference_sets and context.reference_set_ids:
            retrieve_k = min(k * 3, 200)

        results = retriever.retrieve_for_context(
            context, k=retrieve_k, scale=scale, prefix_section=prefix_section
        )

        if use_reference_sets and context.reference_set_ids:
            ref_set = set(context.reference_set_ids)
            results = [r for r in results if r.paper_id in ref_set]

        ref_set_size = len(context.reference_set_ids) if context.reference_set_ids else None
        scores = evaluate_single(results, context.ground_truth_ids, ref_set_size)

        for metric, value in scores.items():
            if metric in strata["overall"]["sum"]:
                strata["overall"]["sum"][metric] += value
                strata[stratum]["sum"][metric] += value
        strata["overall"]["n"] += 1
        strata[stratum]["n"] += 1

    # Clean up query cache
    if embedder is not None:
        embedder.clear_query_cache()

    # Build results
    results = {}
    for name, data in strata.items():
        n = data["n"]
        if n == 0:
            results[name] = EvaluationResult(num_queries=0)
        else:
            s = data["sum"]
            results[name] = EvaluationResult(
                recall_at_1=s["recall@1"] / n,
                recall_at_5=s["recall@5"] / n,
                recall_at_10=s["recall@10"] / n,
                recall_at_20=s["recall@20"] / n,
                recall_at_50=s["recall@50"] / n,
                mrr=s["mrr"] / n,
                ndcg_at_10=s["ndcg@10"] / n,
                concordance=s["concordance"] / n,
                skill_mrr=s["skill_mrr"] / n,
                num_queries=n,
            )

    print("\nStratified results:")
    print(
        f"  Overall:       {results['overall'].num_queries} queries, "
        f"R@10={results['overall'].recall_at_10:.1%}, "
        f"MRR={results['overall'].mrr:.3f}"
    )
    print(
        f"  Full-text GT:  {results['full_text'].num_queries} queries, "
        f"R@10={results['full_text'].recall_at_10:.1%}, "
        f"MRR={results['full_text'].mrr:.3f}"
    )
    print(
        f"  Abstract-only: {results['abstract_only'].num_queries} queries, "
        f"R@10={results['abstract_only'].recall_at_10:.1%}, "
        f"MRR={results['abstract_only'].mrr:.3f}"
    )

    ft_r10 = results["full_text"].recall_at_10
    ao_r10 = results["abstract_only"].recall_at_10
    gap = ft_r10 - ao_r10
    print(f"  Gap (full_text - abstract_only): {gap:+.1%} R@10")

    return results


def evaluate_with_reranking(
    retriever: Retriever,
    reranker: Reranker,
    papers: dict[str, Paper],
    test_set: list[CitationContext],
    initial_k: int = 100,
    final_k: int = 50,
    scale: str = "local",
    use_reference_sets: bool = True,
    show_progress: bool = False,
    prefix_section: bool = False,
    blend_alpha: float = 0.0,
    use_full_text: bool = False,
) -> EvaluationResult:
    """Evaluate retrieval with cross-encoder reranking.

    This function performs two-stage retrieval:
    1. Initial retrieval to get top-initial_k candidates
    2. Cross-encoder reranking to produce final top-final_k results

    Args:
        retriever: Initial retriever for candidate generation
        reranker: Cross-encoder reranker for second-stage scoring
        papers: Dict mapping paper_id to Paper objects
        test_set: List of CitationContext objects with ground_truth_ids
        initial_k: Number of candidates from initial retrieval
        final_k: Number of final results after reranking
        scale: Context scale to use for queries
        use_reference_sets: If True, filter results to reference_set_ids
        show_progress: Whether to show progress bar
        prefix_section: If True, prepend section heading to queries

    Returns:
        Aggregated EvaluationResult
    """
    from tqdm import tqdm

    metrics_sum: dict[str, float] = {
        "recall@1": 0.0,
        "recall@5": 0.0,
        "recall@10": 0.0,
        "recall@20": 0.0,
        "recall@50": 0.0,
        "mrr": 0.0,
        "ndcg@10": 0.0,
        "concordance": 0.0,
        "skill_mrr": 0.0,
    }

    valid_queries = 0

    iterator = test_set
    if show_progress:
        iterator = tqdm(test_set, desc="Evaluating with reranking")

    for context in iterator:
        if not context.ground_truth_ids:
            continue

        # Initial retrieval (get more candidates if filtering by reference set)
        retrieve_k = initial_k
        if use_reference_sets and context.reference_set_ids:
            retrieve_k = min(initial_k * 3, 300)

        candidates = retriever.retrieve_for_context(
            context, k=retrieve_k, scale=scale, prefix_section=prefix_section
        )

        # Filter to reference set if enabled
        if use_reference_sets and context.reference_set_ids:
            ref_set = set(context.reference_set_ids)
            candidates = [r for r in candidates if r.paper_id in ref_set]

        # Limit to initial_k after filtering
        candidates = candidates[:initial_k]

        # Rerank
        query = context.get_query(scale, clean=True, prefix_section=prefix_section)
        reranked = reranker.rerank(
            query=query,
            candidates=candidates,
            papers=papers,
            k=final_k,
            blend_alpha=blend_alpha,
            use_full_text=use_full_text,
        )

        ref_set_size = len(context.reference_set_ids) if context.reference_set_ids else None
        scores = evaluate_single(reranked, context.ground_truth_ids, ref_set_size)

        for metric, value in scores.items():
            if metric in metrics_sum:
                metrics_sum[metric] += value

        valid_queries += 1

    if valid_queries == 0:
        return EvaluationResult(num_queries=0)

    return EvaluationResult(
        recall_at_1=metrics_sum["recall@1"] / valid_queries,
        recall_at_5=metrics_sum["recall@5"] / valid_queries,
        recall_at_10=metrics_sum["recall@10"] / valid_queries,
        recall_at_20=metrics_sum["recall@20"] / valid_queries,
        recall_at_50=metrics_sum["recall@50"] / valid_queries,
        mrr=metrics_sum["mrr"] / valid_queries,
        ndcg_at_10=metrics_sum["ndcg@10"] / valid_queries,
        concordance=metrics_sum["concordance"] / valid_queries,
        skill_mrr=metrics_sum["skill_mrr"] / valid_queries,
        num_queries=valid_queries,
    )


def bootstrap_ci(
    scores: Sequence[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        scores: Per-query metric scores
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g. 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        (lower, upper) bounds of the confidence interval
    """
    scores_arr = np.array(scores, dtype=np.float64)
    if len(scores_arr) == 0:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    n = len(scores_arr)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = scores_arr[indices].mean(axis=1)

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lower, upper)


def paired_bootstrap_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Paired bootstrap significance test between two systems.

    Args:
        scores_a: Per-query scores from system A
        scores_b: Per-query scores from system B (same queries, same order)
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for reproducibility

    Returns:
        (delta, p_value, effect_size) where:
        - delta: mean(scores_b) - mean(scores_a) (positive = B is better)
        - p_value: Two-sided p-value for the null hypothesis delta=0
        - effect_size: Cohen's d
    """
    a = np.array(scores_a, dtype=np.float64)
    b = np.array(scores_b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError(f"Score arrays must have same length: {len(a)} vs {len(b)}")
    if len(a) == 0:
        return (0.0, 1.0, 0.0)

    diffs = b - a
    observed_delta = float(diffs.mean())

    # Bootstrap the paired differences
    rng = np.random.default_rng(seed)
    n = len(diffs)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_deltas = diffs[indices].mean(axis=1)

    # Two-sided p-value: proportion of bootstrap deltas on the opposite side of zero
    # (or more extreme than observed)
    p_value = float(np.mean(np.abs(boot_deltas - observed_delta) >= abs(observed_delta)))

    # Cohen's d
    pooled_std = float(diffs.std(ddof=1)) if n > 1 else 0.0
    effect_size = observed_delta / pooled_std if pooled_std > 0 else 0.0

    return (observed_delta, p_value, effect_size)


def evaluate_retrieval_by_intent(
    query_results: list[QueryResult],
    diagnoses: list["DiagnosisResult"],
) -> dict[str, EvaluationResult]:
    """Compute metrics broken down by citation intent.

    Args:
        query_results: Per-query evaluation results (from experiment run).
        diagnoses: LLM diagnosis results with intent labels.

    Returns:
        Dict mapping intent label to EvaluationResult for queries with that intent.
        Includes an "all" key with overall metrics and "unknown" for unmatched queries.
    """

    # Map context_id -> diagnosis
    diag_by_id: dict[str, DiagnosisResult] = {d.context_id: d for d in diagnoses}

    # Group query results by intent
    by_intent: dict[str, list[QueryResult]] = defaultdict(list)
    for qr in query_results:
        diag = diag_by_id.get(qr.context_id)
        intent = diag.intent if diag else "unknown"
        by_intent[intent].append(qr)

    # Also add "all" group
    by_intent["all"] = list(query_results)

    metric_keys = [
        "recall@1",
        "recall@5",
        "recall@10",
        "recall@20",
        "recall@50",
        "mrr",
        "ndcg@10",
    ]

    results: dict[str, EvaluationResult] = {}
    for intent, qrs in by_intent.items():
        n = len(qrs)
        if n == 0:
            continue
        avg = {k: sum(qr.scores.get(k, 0.0) for qr in qrs) / n for k in metric_keys}
        results[intent] = EvaluationResult(
            recall_at_1=avg["recall@1"],
            recall_at_5=avg["recall@5"],
            recall_at_10=avg["recall@10"],
            recall_at_20=avg["recall@20"],
            recall_at_50=avg["recall@50"],
            mrr=avg["mrr"],
            ndcg_at_10=avg["ndcg@10"],
            num_queries=n,
            per_query=qrs,
        )

    return results
