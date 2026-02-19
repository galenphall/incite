"""Hybrid retrieval combining neural and BM25."""

import time
from collections import defaultdict
from typing import Optional, Union

from incite.interfaces import Retriever
from incite.models import Paper, RetrievalResult
from incite.utils import (
    apply_author_boost,
    apply_graph_boost,
    compute_confidence,
    deduplicate_results,
)


def rrf_fuse(
    result_sets: list[tuple[list[RetrievalResult], float, str]],
    rrf_k: int = 60,
) -> dict[str, dict]:
    """Fuse multiple retrieval result sets using Reciprocal Rank Fusion.

    Args:
        result_sets: List of (results, weight, tier_name) tuples.
            Each tier contributes weight / (rrf_k + rank) per paper.
        rrf_k: RRF smoothing constant (smaller = more top-heavy)

    Returns:
        Dict mapping paper_id to:
        - "total_score": float — accumulated RRF score
        - "scores": dict — per-tier breakdown ({tier}_rank, {tier}_score)
        - "best_result": dict[str, RetrievalResult] — original result per tier
    """
    fused: dict[str, dict] = defaultdict(
        lambda: {"total_score": 0.0, "scores": {}, "best_result": {}}
    )

    for results, weight, tier_name in result_sets:
        for result in results:
            pid = result.paper_id
            rrf_score = weight / (rrf_k + result.rank)
            fused[pid]["total_score"] += rrf_score
            fused[pid]["scores"][f"{tier_name}_rank"] = result.rank
            fused[pid]["scores"][f"{tier_name}_score"] = result.score
            # Keep the best (first-seen, highest-ranked) result per tier
            if tier_name not in fused[pid]["best_result"]:
                fused[pid]["best_result"][tier_name] = result

    return dict(fused)


def rrf_sort(fused: dict[str, dict]) -> list[str]:
    """Sort paper IDs by their fused total_score descending."""
    return sorted(fused.keys(), key=lambda pid: fused[pid]["total_score"], reverse=True)


class HybridRetriever(Retriever):
    """Hybrid retriever combining multiple retrieval methods."""

    def __init__(
        self,
        retrievers: list[tuple[Retriever, float]],
        fusion: str = "rrf",
        rrf_k: int = 10,
    ):
        """Initialize hybrid retriever.

        Args:
            retrievers: List of (retriever, weight) tuples
            fusion: Fusion method ('rrf' or 'weighted')
            rrf_k: RRF constant (only used when fusion='rrf')
        """
        self.retrievers = retrievers
        self.fusion = fusion
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        k: int = 10,
        papers: Optional[dict[str, Paper]] = None,
        author_boost: float = 1.2,
        graph_metrics: Optional[dict[str, dict[str, float]]] = None,
        doi_to_s2: Optional[dict[str, str]] = None,
        return_timing: bool = False,
        deduplicate: bool = False,
        **kwargs,
    ) -> Union[list[RetrievalResult], tuple[list[RetrievalResult], dict]]:
        """Retrieve top-k papers using hybrid fusion with optional author boosting.

        Args:
            query: Query text
            k: Number of results to return
            papers: Optional dict mapping paper IDs to Paper objects. If provided
                    and author_boost > 1.0, papers whose author lastnames appear
                    in the query will have their scores boosted.
            author_boost: Multiplier for author match boosting (default 1.2 = 20% boost).
                          Set to 1.0 to disable boosting.
            graph_metrics: Optional dict with "pagerank" and/or "cocitation" sub-dicts
                           mapping S2 IDs to scores. Used for graph-based boosting.
            doi_to_s2: Optional mapping from DOI to S2 ID, required for graph boosting.
            return_timing: If True, return (results, timing_dict)
            deduplicate: If True and papers provided, remove results with duplicate titles

        Returns:
            List of RetrievalResult objects sorted by fused score.
            If return_timing=True, returns (results, timing_dict).
        """
        timing = {}

        # Get more results if we'll be re-ranking with author boosting or deduplicating
        initial_k = k * 2 if (papers and author_boost > 1.0) or deduplicate else k

        # Pass query_embedding to neural sub-retrievers (BM25 ignores it)
        sub_kwargs = {}
        if "query_embedding" in kwargs:
            sub_kwargs["query_embedding"] = kwargs["query_embedding"]

        # Collect results from all sub-retrievers
        result_sets = []
        for retriever, weight in self.retrievers:
            retriever_name = type(retriever).__name__.lower().replace("retriever", "")
            if return_timing:
                results, sub_timing = retriever.retrieve(
                    query, k=initial_k * 3, return_timing=True, **sub_kwargs
                )
                for key, value in sub_timing.items():
                    timing[f"{retriever_name}_{key}"] = value
            else:
                results = retriever.retrieve(query, k=initial_k * 3, **sub_kwargs)

            if self.fusion == "rrf":
                result_sets.append((results, weight, retriever_name))
            else:
                # Weighted score fusion: accumulate directly
                result_sets.append((results, weight, retriever_name))

        # Fuse results
        fusion_start = time.perf_counter()
        if self.fusion == "rrf":
            all_results = rrf_fuse(result_sets, rrf_k=self.rrf_k)
            # Merge original score_breakdowns for backward compatibility
            for pid, data in all_results.items():
                for tier_result in data["best_result"].values():
                    data["scores"].update(tier_result.score_breakdown)
        else:
            # Weighted score fusion fallback
            all_results: dict[str, dict] = defaultdict(lambda: {"scores": {}, "total_score": 0.0})
            for results, weight, _ in result_sets:
                for result in results:
                    all_results[result.paper_id]["total_score"] += weight * result.score
                    all_results[result.paper_id]["scores"].update(result.score_breakdown)

        # Apply author boosting if papers dict provided
        if papers and author_boost > 1.0:
            apply_author_boost(all_results, query, papers, author_boost)

        # Apply graph-based boosting if metrics provided
        if papers and graph_metrics and doi_to_s2:
            apply_graph_boost(all_results, papers, graph_metrics, doi_to_s2)

        # Sort and build results
        sorted_ids = rrf_sort(all_results)
        fetch_k = k * 2 if deduplicate else k
        results = []
        for rank, paper_id in enumerate(sorted_ids[:fetch_k]):
            data = all_results[paper_id]
            results.append(
                RetrievalResult(
                    paper_id=paper_id,
                    score=data["total_score"],
                    rank=rank + 1,
                    score_breakdown=data["scores"],
                    confidence=compute_confidence(data["scores"], mode="hybrid"),
                )
            )

        # Deduplicate by title if requested
        if deduplicate and papers:
            results = deduplicate_results(results, papers)
            results = results[:k]
            for i, result in enumerate(results):
                result.rank = i + 1

        if return_timing:
            timing["fusion_ms"] = (time.perf_counter() - fusion_start) * 1000
            return results, timing
        return results
