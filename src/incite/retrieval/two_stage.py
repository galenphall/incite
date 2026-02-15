"""Two-stage retrieval: paper-level ranking + paragraph-level reranking.

Stage 1: Find candidate papers via hybrid retrieval (neural + BM25).
Stage 2: For each candidate, search only its chunks and blend scores:
         final_score = alpha * paper_score_norm + (1 - alpha) * best_chunk_score

Papers with no chunks (abstract-only) get final_score = alpha * paper_score_norm,
naturally penalized but not excluded.
"""

import time
from typing import Optional, Union

from incite.embeddings.base import BaseEmbedder
from incite.embeddings.chunk_store import ChunkStore
from incite.interfaces import Retriever
from incite.models import Chunk, Paper, RetrievalResult
from incite.retrieval.paragraph import _highlight_sentence_in_parent
from incite.utils import compute_confidence, deduplicate_results


class TwoStageRetriever(Retriever):
    """Two-stage retriever: paper-level candidates + paragraph reranking.

    Wraps an existing paper-level retriever (stage 1) with a ChunkStore
    for scoped paragraph search (stage 2). The final score blends
    paper-level and paragraph-level signals.
    """

    def __init__(
        self,
        paper_retriever: Retriever,
        chunk_store: ChunkStore,
        chunks: dict[str, Chunk],
        embedder: BaseEmbedder,
        alpha: float = 0.6,
        stage1_k: int = 50,
        evidence_threshold: float = 0.35,
        max_evidence_per_paper: int = 3,
    ):
        """Initialize two-stage retriever.

        Args:
            paper_retriever: Stage 1 retriever (typically HybridRetriever)
            chunk_store: ChunkStore with indexed paragraph embeddings
            chunks: Dict mapping chunk_id -> Chunk for text extraction
            embedder: Embedder for computing query embeddings
            alpha: Weight for paper score in the blend (0-1).
                1.0 = paper score only, 0.0 = chunk score only.
            stage1_k: Number of papers to pass from stage 1 to stage 2
            evidence_threshold: Minimum chunk score to attach as evidence
            max_evidence_per_paper: Maximum evidence snippets per paper
        """
        self.paper_retriever = paper_retriever
        self.chunk_store = chunk_store
        self.chunks = chunks
        self.embedder = embedder
        self.alpha = alpha
        self.stage1_k = stage1_k
        self.evidence_threshold = evidence_threshold
        self.max_evidence_per_paper = max_evidence_per_paper

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
        """Retrieve top-k papers using two-stage pipeline.

        Args:
            query: Query text
            k: Number of final results to return
            papers: Optional dict for author boosting and deduplication
            author_boost: Multiplier for author match boosting
            return_timing: If True, return (results, timing_dict)
            deduplicate: If True and papers provided, remove duplicate titles
            **kwargs: query_embedding (pre-computed), passed to stage 1

        Returns:
            List of RetrievalResult with evidence attached.
            If return_timing=True, returns (results, timing_dict).
        """
        timing = {}

        # --- Stage 1: Paper-level retrieval ---
        stage1_start = time.perf_counter()
        stage1_results, stage1_timing = self.paper_retriever.retrieve(
            query,
            k=self.stage1_k,
            papers=papers,
            author_boost=author_boost,
            return_timing=True,
            **kwargs,
        )
        timing["stage1_ms"] = (time.perf_counter() - stage1_start) * 1000
        timing.update({f"stage1_{k}": v for k, v in stage1_timing.items()})

        if not stage1_results:
            if return_timing:
                return [], timing
            return []

        # Normalize stage 1 scores to [0, 1] via min-max
        scores = [r.score for r in stage1_results]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        if score_range > 0:
            paper_scores_norm = {
                r.paper_id: (r.score - min_score) / score_range for r in stage1_results
            }
        else:
            paper_scores_norm = {r.paper_id: 1.0 for r in stage1_results}

        # Preserve original breakdowns for output
        paper_breakdowns = {r.paper_id: r.score_breakdown for r in stage1_results}

        # --- Stage 2: Scoped chunk search ---
        stage2_start = time.perf_counter()

        # Get query embedding (reuse if provided, else compute)
        query_embedding = kwargs.get("query_embedding")
        if query_embedding is None:
            query_embedding = self.embedder.embed_query(query)

        paper_ids = [r.paper_id for r in stage1_results]
        chunk_results = self.chunk_store.search_within_papers(
            query_embedding,
            paper_ids,
            top_per_paper=self.max_evidence_per_paper,
        )
        timing["stage2_ms"] = (time.perf_counter() - stage2_start) * 1000

        # --- Rerank: blend paper + chunk scores ---
        rerank_start = time.perf_counter()

        final_results = []
        for paper_id in paper_ids:
            paper_score_norm = paper_scores_norm[paper_id]
            breakdown = dict(paper_breakdowns.get(paper_id, {}))
            breakdown["paper_score_norm"] = paper_score_norm

            # Get best chunk score for this paper
            paper_chunk_results = chunk_results.get(paper_id, [])
            if paper_chunk_results:
                best_chunk_score = paper_chunk_results[0][1]  # Already sorted desc
                final_score = self.alpha * paper_score_norm + (1 - self.alpha) * best_chunk_score
                breakdown["best_chunk_score"] = best_chunk_score
                breakdown["num_chunks_matched"] = len(paper_chunk_results)
            else:
                # No chunks — paper score only, naturally penalized
                final_score = self.alpha * paper_score_norm
                breakdown["best_chunk_score"] = 0.0
                breakdown["num_chunks_matched"] = 0

            breakdown["alpha"] = self.alpha

            # Build evidence snippets
            matched_paragraph = None
            matched_paragraphs = []
            for chunk_id, score in paper_chunk_results:
                if score >= self.evidence_threshold - 1e-6 and chunk_id in self.chunks:
                    chunk = self.chunks[chunk_id]
                    text = _highlight_sentence_in_parent(chunk)
                    matched_paragraphs.append(
                        {
                            "text": text,
                            "score": score,
                            "section": chunk.section,
                            "page": chunk.page_number,
                        }
                    )
            if matched_paragraphs:
                matched_paragraph = matched_paragraphs[0]["text"]

            final_results.append(
                RetrievalResult(
                    paper_id=paper_id,
                    score=final_score,
                    rank=0,  # Will be set after sorting
                    score_breakdown=breakdown,
                    matched_paragraph=matched_paragraph,
                    matched_paragraphs=matched_paragraphs,
                    confidence=compute_confidence(breakdown, mode="two_stage"),
                )
            )

        # Sort by final score descending
        final_results.sort(key=lambda r: r.score, reverse=True)

        # Apply deduplication before assigning ranks
        if deduplicate and papers:
            final_results = deduplicate_results(final_results, papers)

        # Assign ranks and truncate
        final_results = final_results[:k]
        for i, result in enumerate(final_results):
            result.rank = i + 1

        timing["rerank_ms"] = (time.perf_counter() - rerank_start) * 1000

        if return_timing:
            return final_results, timing
        return final_results
