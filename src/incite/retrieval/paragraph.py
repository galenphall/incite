"""Paragraph-level retrieval with paper aggregation.

This module provides retrievers that search at the chunk/paragraph level
and aggregate results to paper-level recommendations.
"""

import math
import time
from collections import defaultdict
from typing import Optional, Union

from incite.embeddings.base import BaseEmbedder
from incite.embeddings.chunk_store import ChunkStore
from incite.interfaces import Retriever
from incite.models import Chunk, Paper, RetrievalResult
from incite.retrieval.hybrid import rrf_fuse, rrf_sort
from incite.utils import apply_author_boost, compute_confidence, deduplicate_results


def _highlight_sentence_in_parent(chunk: Chunk) -> str:
    """Get display text with focal sentence highlighted within parent paragraph.

    If parent_text is available, returns the parent with the focal sentence
    wrapped in **bold** markdown. Falls back to just the sentence text.

    Args:
        chunk: The matched chunk

    Returns:
        Display text with highlighting
    """
    if not chunk.parent_text:
        return chunk.text

    # Try to find the focal sentence in the parent and highlight it
    focal = chunk.text.strip()
    parent = chunk.parent_text

    # Find the sentence in the parent (exact match)
    idx = parent.find(focal)
    if idx != -1:
        # Highlight with markdown bold
        before = parent[:idx]
        after = parent[idx + len(focal) :]
        return f"{before}**{focal}**{after}"

    # Fallback: sentence not found in parent (shouldn't happen)
    return chunk.text


class ParagraphRetriever(Retriever):
    """Retriever that searches chunk-level embeddings and aggregates to papers.

    Searches the chunk index for similar paragraphs, then aggregates scores
    to paper level using max pooling (best chunk score represents paper).
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        chunk_store: ChunkStore,
        chunks: dict[str, Chunk],
        aggregation: str = "max",
        top_k_for_mean: int = 3,
        weighted_max_alpha: float = 0.7,
        coverage_bonus_per_chunk: float = 0.02,
        log_norm_weight: float = 0.02,
    ):
        """Initialize paragraph retriever.

        Args:
            embedder: Embedder for query encoding
            chunk_store: ChunkStore with indexed chunks
            chunks: Dict mapping chunk_id -> Chunk for matched text lookup
            aggregation: How to aggregate chunk scores to paper level.
                "max" = best chunk score (default)
                "mean" = average chunk score
                "sum" = sum of chunk scores
                "weighted_max" = max * alpha + mean * (1-alpha)
                "top_k_mean" = average of top-k chunks per paper
                "coverage" = max + bonus for each additional chunk
                "log_normalized" = max + small log bonus (dampens multi-chunk advantage)
            top_k_for_mean: Number of top chunks to average for "top_k_mean" (default 3)
            weighted_max_alpha: Weight for max in "weighted_max" (default 0.7)
            coverage_bonus_per_chunk: Bonus per additional chunk for "coverage" (default 0.02)
            log_norm_weight: Weight for log bonus in "log_normalized" (default 0.02)
        """
        valid_aggregations = {
            "max",
            "mean",
            "sum",
            "weighted_max",
            "top_k_mean",
            "coverage",
            "log_normalized",
        }
        if aggregation not in valid_aggregations:
            raise ValueError(
                f"Unknown aggregation: {aggregation!r}. Available: {sorted(valid_aggregations)}"
            )

        self.embedder = embedder
        self.chunk_store = chunk_store
        self.chunks = chunks
        self.aggregation = aggregation
        self.top_k_for_mean = top_k_for_mean
        self.weighted_max_alpha = weighted_max_alpha
        self.coverage_bonus_per_chunk = coverage_bonus_per_chunk
        self.log_norm_weight = log_norm_weight

    def retrieve(
        self,
        query: str,
        k: int = 10,
        initial_k_multiplier: int = 3,
        return_timing: bool = False,
        **kwargs,
    ) -> Union[list[RetrievalResult], tuple[list[RetrievalResult], dict]]:
        """Retrieve top-k papers based on paragraph similarity.

        Args:
            query: Query text
            k: Number of papers to return
            initial_k_multiplier: How many more chunks to retrieve than papers
                                 (to ensure good paper coverage)
            return_timing: If True, return (results, timing_dict)
            **kwargs: Ignored (for API compatibility)

        Returns:
            List of RetrievalResult with paper_id and matched_paragraph.
            If return_timing=True, returns (results, timing_dict).
        """
        timing = {}

        # Use pre-computed embedding if provided, otherwise embed the query
        embed_start = time.perf_counter()
        query_embedding = kwargs.get("query_embedding")
        if query_embedding is None:
            query_embedding = self.embedder.embed_query(query)
        timing["embed_query_ms"] = (time.perf_counter() - embed_start) * 1000

        # Search for more chunks than papers needed
        search_start = time.perf_counter()
        initial_k = k * initial_k_multiplier
        chunk_results = self.chunk_store.search_with_papers(query_embedding, k=initial_k)
        timing["vector_search_ms"] = (time.perf_counter() - search_start) * 1000

        # Aggregate to paper level
        paper_scores: dict[str, dict] = defaultdict(
            lambda: {"scores": [], "best_chunk_id": None, "best_score": -1.0}
        )

        for chunk_id, paper_id, score in chunk_results:
            paper_data = paper_scores[paper_id]
            paper_data["scores"].append(score)

            if score > paper_data["best_score"]:
                paper_data["best_score"] = score
                paper_data["best_chunk_id"] = chunk_id

        # Compute aggregated scores
        paper_final_scores: dict[str, float] = {}
        for paper_id, data in paper_scores.items():
            scores = data["scores"]
            max_score = max(scores)
            mean_score = sum(scores) / len(scores)

            if self.aggregation == "max":
                paper_final_scores[paper_id] = max_score
            elif self.aggregation == "mean":
                paper_final_scores[paper_id] = mean_score
            elif self.aggregation == "sum":
                paper_final_scores[paper_id] = sum(scores)
            elif self.aggregation == "weighted_max":
                # Blend max and mean to penalize one-hit wonders
                alpha = self.weighted_max_alpha
                paper_final_scores[paper_id] = alpha * max_score + (1 - alpha) * mean_score
            elif self.aggregation == "top_k_mean":
                # Average of top-k chunks (more stable than single max)
                sorted_scores = sorted(scores, reverse=True)
                top_k = sorted_scores[: self.top_k_for_mean]
                paper_final_scores[paper_id] = sum(top_k) / len(top_k)
            elif self.aggregation == "coverage":
                # Bonus for papers with multiple relevant chunks
                extra_chunks = max(0, len(scores) - 1)
                bonus = extra_chunks * self.coverage_bonus_per_chunk
                paper_final_scores[paper_id] = max_score + bonus
            elif self.aggregation == "log_normalized":
                # Dampen multi-chunk advantage: max + small log bonus
                extra_chunks = max(0, len(scores) - 1)
                bonus = self.log_norm_weight * math.log1p(extra_chunks)
                paper_final_scores[paper_id] = max_score + bonus
            else:
                paper_final_scores[paper_id] = max_score

        # Sort by aggregated score
        sorted_papers = sorted(
            paper_final_scores.keys(),
            key=lambda x: paper_final_scores[x],
            reverse=True,
        )

        # Build results with matched paragraphs
        results = []
        for rank, paper_id in enumerate(sorted_papers[:k]):
            data = paper_scores[paper_id]
            best_chunk_id = data["best_chunk_id"]

            # Get matched paragraph text (with highlighting if parent_text available)
            matched_paragraph = None
            if best_chunk_id and best_chunk_id in self.chunks:
                matched_paragraph = _highlight_sentence_in_parent(self.chunks[best_chunk_id])

            breakdown = {
                "best_chunk_score": data["best_score"],
                "num_chunks_matched": len(data["scores"]),
            }
            results.append(
                RetrievalResult(
                    paper_id=paper_id,
                    score=paper_final_scores[paper_id],
                    rank=rank + 1,
                    score_breakdown=breakdown,
                    matched_paragraph=matched_paragraph,
                    confidence=compute_confidence(breakdown, mode="paragraph"),
                )
            )

        if return_timing:
            return results, timing
        return results

    @classmethod
    def from_chunks(
        cls,
        chunks: list[Chunk],
        embedder: BaseEmbedder,
        aggregation: str = "max",
        show_progress: bool = True,
        top_k_for_mean: int = 3,
        weighted_max_alpha: float = 0.7,
        coverage_bonus_per_chunk: float = 0.02,
        log_norm_weight: float = 0.02,
    ) -> "ParagraphRetriever":
        """Build a ParagraphRetriever from chunks.

        Args:
            chunks: List of Chunk objects
            embedder: Embedder for encoding
            aggregation: Aggregation method ("max", "mean", "sum", "weighted_max",
                        "top_k_mean", "coverage", "log_normalized")
            show_progress: Whether to show progress bar
            top_k_for_mean: Number of top chunks to average for "top_k_mean"
            weighted_max_alpha: Weight for max in "weighted_max"
            coverage_bonus_per_chunk: Bonus per additional chunk for "coverage"
            log_norm_weight: Weight for log bonus in "log_normalized"

        Returns:
            Configured ParagraphRetriever
        """
        from incite.embeddings.chunk_store import build_chunk_index

        # Build chunk dict for lookup
        chunk_dict = {c.id: c for c in chunks}

        # Build index
        chunk_store = build_chunk_index(chunks, embedder, show_progress=show_progress)

        return cls(
            embedder=embedder,
            chunk_store=chunk_store,
            chunks=chunk_dict,
            aggregation=aggregation,
            top_k_for_mean=top_k_for_mean,
            weighted_max_alpha=weighted_max_alpha,
            coverage_bonus_per_chunk=coverage_bonus_per_chunk,
            log_norm_weight=log_norm_weight,
        )


class HybridParagraphRetriever(Retriever):
    """Hybrid retriever combining paragraph neural search with paper-level BM25.

    Uses RRF fusion to combine:
    1. Paragraph-level neural retrieval (aggregated to paper)
    2. Paper-level BM25 (title + abstract)

    This gives the precision of paragraph matching with the recall of
    keyword search on full paper metadata.
    """

    def __init__(
        self,
        paragraph_retriever: ParagraphRetriever,
        bm25_retriever: Retriever,
        neural_weight: float = 1.0,
        bm25_weight: float = 1.0,
        rrf_k: int = 10,
    ):
        """Initialize hybrid paragraph retriever.

        Args:
            paragraph_retriever: Paragraph-level neural retriever
            bm25_retriever: Paper-level BM25 retriever
            neural_weight: Weight for neural results in fusion
            bm25_weight: Weight for BM25 results in fusion
            rrf_k: RRF constant (smaller = more top-heavy ranking)
        """
        self.paragraph_retriever = paragraph_retriever
        self.bm25_retriever = bm25_retriever
        self.neural_weight = neural_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        k: int = 10,
        papers: Optional[dict[str, Paper]] = None,
        author_boost: float = 1.0,
        return_timing: bool = False,
        deduplicate: bool = False,
        **kwargs,
    ) -> Union[list[RetrievalResult], tuple[list[RetrievalResult], dict]]:
        """Retrieve top-k papers using hybrid paragraph + BM25 fusion.

        Args:
            query: Query text
            k: Number of results to return
            papers: Optional dict for author boosting
            author_boost: Multiplier for author match boosting
            return_timing: If True, return (results, timing_dict)
            deduplicate: If True and papers provided, remove results with duplicate titles

        Returns:
            List of RetrievalResult with matched_paragraph from neural.
            If return_timing=True, returns (results, timing_dict).
        """
        timing = {}

        # Get more results for fusion
        initial_k = k * 3

        # Pass query_embedding to paragraph (neural) sub-retriever
        sub_kwargs = {}
        if "query_embedding" in kwargs:
            sub_kwargs["query_embedding"] = kwargs["query_embedding"]

        # Get paragraph-level neural results
        if return_timing:
            para_results, para_timing = self.paragraph_retriever.retrieve(
                query, k=initial_k, return_timing=True, **sub_kwargs
            )
            timing.update(para_timing)
        else:
            para_results = self.paragraph_retriever.retrieve(query, k=initial_k, **sub_kwargs)

        # Get paper-level BM25 results (no query_embedding needed)
        if return_timing:
            bm25_results, bm25_timing = self.bm25_retriever.retrieve(
                query, k=initial_k, return_timing=True
            )
            timing.update(bm25_timing)
        else:
            bm25_results = self.bm25_retriever.retrieve(query, k=initial_k)

        # Fuse with RRF
        fusion_start = time.perf_counter()
        fused = rrf_fuse(
            [
                (para_results, self.neural_weight, "neural"),
                (bm25_results, self.bm25_weight, "bm25"),
            ],
            rrf_k=self.rrf_k,
        )

        # Propagate chunk metadata from neural tier
        for pid, data in fused.items():
            neural_result = data["best_result"].get("neural")
            if neural_result:
                data["matched_paragraph"] = neural_result.matched_paragraph
                for key in ("best_chunk_score", "num_chunks_matched"):
                    if key in neural_result.score_breakdown:
                        data["scores"][key] = neural_result.score_breakdown[key]
            else:
                data["matched_paragraph"] = None

        # Apply author boosting if papers dict provided
        if papers and author_boost > 1.0:
            apply_author_boost(fused, query, papers, author_boost)

        # Sort and build results
        sorted_ids = rrf_sort(fused)
        fetch_k = k * 2 if deduplicate else k
        results = []
        for rank, paper_id in enumerate(sorted_ids[:fetch_k]):
            data = fused[paper_id]
            results.append(
                RetrievalResult(
                    paper_id=paper_id,
                    score=data["total_score"],
                    rank=rank + 1,
                    score_breakdown=data["scores"],
                    matched_paragraph=data["matched_paragraph"],
                    confidence=compute_confidence(data["scores"], mode="paragraph"),
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

    @classmethod
    def from_chunks_and_papers(
        cls,
        chunks: list[Chunk],
        papers: list[Paper],
        embedder: BaseEmbedder,
        neural_weight: float = 1.0,
        bm25_weight: float = 1.0,
        rrf_k: int = 10,
        show_progress: bool = True,
    ) -> "HybridParagraphRetriever":
        """Build a HybridParagraphRetriever from chunks and papers.

        Args:
            chunks: List of Chunk objects
            papers: List of Paper objects
            embedder: Embedder for encoding
            neural_weight: Weight for neural in fusion
            bm25_weight: Weight for BM25 in fusion
            rrf_k: RRF constant
            show_progress: Whether to show progress bars

        Returns:
            Configured HybridParagraphRetriever
        """
        from incite.retrieval.bm25 import BM25Retriever

        # Build paragraph retriever
        para_retriever = ParagraphRetriever.from_chunks(
            chunks, embedder, show_progress=show_progress
        )

        # Build BM25 retriever from papers
        bm25_retriever = BM25Retriever.from_papers(papers, include_metadata=True)

        return cls(
            paragraph_retriever=para_retriever,
            bm25_retriever=bm25_retriever,
            neural_weight=neural_weight,
            bm25_weight=bm25_weight,
            rrf_k=rrf_k,
        )
