"""Cross-encoder rerankers for improving retrieval precision."""

import time
from typing import Optional, Union

import numpy as np

from incite.interfaces import Reranker, Retriever
from incite.models import Paper, RetrievalResult
from incite.utils import get_best_device

# Available cross-encoder configurations
RERANKERS = {
    "bge": {
        "name": "BGE Reranker v2 M3 (best quality, long context)",
        "model": "BAAI/bge-reranker-v2-m3",
        "max_length": 8192,
        "size_mb": 2300,
    },
    "bge-base": {
        "name": "BGE Reranker Base (faster, shorter context)",
        "model": "BAAI/bge-reranker-base",
        "max_length": 512,
        "size_mb": 440,
    },
    "ms-marco": {
        "name": "MS-MARCO MiniLM L6 (fast baseline)",
        "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "max_length": 512,
        "size_mb": 90,
    },
    "ms-marco-l12": {
        "name": "MS-MARCO MiniLM L12 (slightly better)",
        "model": "cross-encoder/ms-marco-MiniLM-L12-v2",
        "max_length": 512,
        "size_mb": 134,
    },
    "jina": {
        "name": "Jina Reranker v2 (multilingual, balanced)",
        "model": "jinaai/jina-reranker-v2-base-multilingual",
        "max_length": 1024,
        "size_mb": 1100,
    },
    "gte": {
        "name": "GTE Reranker (long context, lightweight)",
        "model": "Alibaba-NLP/gte-multilingual-reranker-base",
        "max_length": 8192,
        "size_mb": 440,
    },
    "citation-ft": {
        "name": "Citation-FT MiniLM v3 (listwise LambdaLoss, co-citation negatives)",
        "model": "models/reranker-v3/final",
        "max_length": 512,
        "size_mb": 90,
    },
    "citation-ft-v4": {
        "name": "Citation-FT BGE v4 (BGE-reranker-base, cleaned data, cosine LR)",
        "model": "models/reranker-v4/final",
        "max_length": 512,
        "size_mb": 440,
    },
    "citation-ft-v5": {
        "name": "Citation-FT BGE v5 (full-text input, retriever-aware dev, random negatives)",
        "model": "models/reranker-v5/final",
        "max_length": 512,
        "size_mb": 440,
    },
}

DEFAULT_RERANKER = "bge"


def _blend_scores(
    retrieval_scores: np.ndarray,
    ce_scores: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend retrieval and cross-encoder scores via min-max normalization.

    Args:
        retrieval_scores: Scores from initial retrieval (e.g. RRF)
        ce_scores: Scores from cross-encoder
        alpha: Blend weight. 0.0 = pure CE, 1.0 = pure retrieval.

    Returns:
        Blended scores in [0, 1] range
    """

    def _minmax(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-12:
            return np.full_like(arr, 0.5)
        return (arr - lo) / (hi - lo)

    return alpha * _minmax(retrieval_scores) + (1 - alpha) * _minmax(ce_scores)


class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker using sentence-transformers CrossEncoder.

    Cross-encoders score query-document pairs together, allowing them to
    capture fine-grained interactions that bi-encoders miss. Use as a
    second stage after initial retrieval to improve precision at top ranks.

    Uses sentence-transformers CrossEncoder.predict() for inference, ensuring
    the same code path as training/evaluation (CERerankingEvaluator).

    Supported models:
    - BGE Reranker v2 M3: Best quality, 8192 token context (recommended)
    - BGE Reranker Base: Faster, 512 token context
    - MS-MARCO MiniLM L6: Fast baseline, 512 tokens
    - MS-MARCO MiniLM L12: Slightly better than L6, 512 tokens
    - Jina Reranker v2: Multilingual, 1024 tokens
    - GTE Reranker: Long context (8192), lightweight
    - Citation-FT: Fine-tuned on citation data (90MB)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        """Initialize the cross-encoder reranker.

        Args:
            model_name: HuggingFace model name or path
            device: Device to use (auto-detected if None)
            batch_size: Batch size for scoring (lower for memory-constrained)
            max_length: Max sequence length (auto-detected from model if None)
            cache_dir: Directory to cache model weights
        """
        self.model_name = model_name
        self.device = device if device is not None else get_best_device()
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._cross_encoder = None

        # Determine max length from known models or default
        if max_length is not None:
            self._max_length = max_length
        else:
            # Look up in known configurations
            for config in RERANKERS.values():
                if config["model"] == model_name:
                    self._max_length = config["max_length"]
                    break
            else:
                self._max_length = 512  # Conservative default

    def _load_model(self):
        """Lazy load the CrossEncoder model."""
        if self._cross_encoder is not None:
            return

        from sentence_transformers import CrossEncoder

        print(f"Loading reranker: {self.model_name}")
        print(f"Using device: {self.device}")

        self._cross_encoder = CrossEncoder(
            self.model_name,
            max_length=self._max_length,
            device=self.device,
        )

    def _score_pairs(
        self,
        query: str,
        documents: list[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Score query-document pairs using CrossEncoder.predict().

        Uses the same code path as sentence-transformers CERerankingEvaluator
        to ensure train/inference consistency.

        Args:
            query: Query text
            documents: List of document texts
            show_progress: Whether to show progress bar

        Returns:
            Array of scores, one per document
        """
        self._load_model()

        if len(documents) == 0:
            return np.array([])

        pairs = [[query, doc] for doc in documents]
        scores = self._cross_encoder.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
        )
        return np.asarray(scores, dtype=np.float64)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        papers: dict[str, Paper],
        k: int = 10,
        max_per_paper: Optional[int] = None,
        citation_count_alpha: float = 0.0,
        blend_alpha: float = 0.0,
        use_full_text: bool = False,
        show_progress: bool = False,
    ) -> list[RetrievalResult]:
        """Rerank candidates using cross-encoder scores.

        Args:
            query: Query text (citation context)
            candidates: Initial retrieval results to rerank
            papers: Dict mapping paper_id to Paper objects
            k: Number of results to return after reranking
            max_per_paper: If set, cap candidates per paper before scoring.
            citation_count_alpha: If > 0, blend log(1+citation_count) into
                the final score.
            blend_alpha: If > 0, blend retrieval scores with CE scores.
                final = blend_alpha * norm(retrieval) + (1-blend_alpha) * norm(ce).
                0.0 = pure CE (default), 1.0 = pure retrieval.
            use_full_text: If True, pass full paper embedding text
                (title + authors + year + journal + abstract) to the
                cross-encoder instead of raw abstract only.
            show_progress: Whether to show progress bar

        Returns:
            Top-k results with updated scores and ranks
        """
        if len(candidates) == 0:
            return []

        # Deduplicate by paper_id if max_per_paper is set
        if max_per_paper is not None:
            from collections import defaultdict

            paper_counts: dict[str, int] = defaultdict(int)
            deduped = []
            for result in candidates:
                if paper_counts[result.paper_id] < max_per_paper:
                    deduped.append(result)
                    paper_counts[result.paper_id] += 1
            candidates = deduped

        # Build document texts for scoring
        documents = []
        valid_candidates = []

        for result in candidates:
            paper = papers.get(result.paper_id)
            if paper is None:
                continue

            if use_full_text:
                doc_text = paper.to_embedding_text()
            else:
                doc_text = paper.abstract if paper.abstract else paper.title
            documents.append(doc_text)
            valid_candidates.append(result)

        if len(documents) == 0:
            return []

        # Score all pairs
        ce_scores = self._score_pairs(query, documents, show_progress=show_progress)

        # Optionally blend citation count signal
        if citation_count_alpha > 0:
            import math

            for i, result in enumerate(valid_candidates):
                paper = papers.get(result.paper_id)
                if paper and hasattr(paper, "citation_count") and paper.citation_count:
                    ce_scores[i] += citation_count_alpha * math.log(1 + paper.citation_count)

        # Blend retrieval scores with CE scores via min-max normalization
        if blend_alpha > 0 and len(valid_candidates) > 1:
            retrieval_scores = np.array([r.score for r in valid_candidates], dtype=np.float64)
            scores = _blend_scores(retrieval_scores, ce_scores, blend_alpha)
        else:
            scores = ce_scores

        # Sort by final score
        scored_results = list(zip(valid_candidates, scores, ce_scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Build reranked results
        reranked = []
        for rank, (original, final_score, ce_score) in enumerate(scored_results[:k]):
            reranked.append(
                RetrievalResult(
                    paper_id=original.paper_id,
                    score=float(final_score),
                    rank=rank + 1,
                    score_breakdown={
                        "cross_encoder": float(ce_score),
                        "retrieval_score": original.score,
                        "retrieval_rank": original.rank,
                    },
                )
            )

        return reranked


def get_reranker(
    reranker_type: str = DEFAULT_RERANKER,
    device: Optional[str] = None,
    batch_size: int = 16,
) -> CrossEncoderReranker:
    """Get a reranker instance by type.

    Args:
        reranker_type: Key from RERANKERS dict ("bge", "ms-marco", etc.)
        device: Device to use (auto-detected if None)
        batch_size: Batch size for scoring

    Returns:
        Configured CrossEncoderReranker instance
    """
    if reranker_type not in RERANKERS:
        raise ValueError(f"Unknown reranker: {reranker_type}. Available: {list(RERANKERS.keys())}")

    config = RERANKERS[reranker_type]
    return CrossEncoderReranker(
        model_name=config["model"],
        device=device,
        batch_size=batch_size,
        max_length=config["max_length"],
    )


class RerankedRetriever(Retriever):
    """Wrapper that combines a retriever with a reranker.

    This class provides a convenient way to chain initial retrieval
    with cross-encoder reranking in a single step.
    """

    def __init__(
        self,
        retriever,
        reranker: Reranker,
        papers: dict[str, Paper],
        initial_k: int = 100,
    ):
        """Initialize the reranked retriever.

        Args:
            retriever: Base retriever for initial retrieval
            reranker: Reranker for second-stage scoring
            papers: Dict mapping paper_id to Paper objects
            initial_k: Number of candidates to retrieve before reranking
        """
        self.retriever = retriever
        self.reranker = reranker
        self.papers = papers
        self.initial_k = initial_k

    def retrieve(
        self,
        query: str,
        k: int = 10,
        return_timing: bool = False,
        show_progress: bool = False,
        **kwargs,
    ) -> Union[list[RetrievalResult], tuple[list[RetrievalResult], dict]]:
        """Retrieve and rerank papers for a query.

        Args:
            query: Query text
            k: Number of final results
            return_timing: If True, return (results, timing_dict)
            show_progress: Whether to show progress bar

        Returns:
            Reranked retrieval results.
            If return_timing=True, returns (results, timing_dict).
        """
        timing = {}

        # Initial retrieval
        t0 = time.perf_counter()
        candidates = self.retriever.retrieve(query, k=self.initial_k)
        timing["retrieval_ms"] = (time.perf_counter() - t0) * 1000

        # Rerank
        t0 = time.perf_counter()
        results = self.reranker.rerank(
            query=query,
            candidates=candidates,
            papers=self.papers,
            k=k,
            show_progress=show_progress,
        )
        timing["rerank_ms"] = (time.perf_counter() - t0) * 1000

        if return_timing:
            return results, timing
        return results

    def retrieve_for_context(
        self,
        context,
        k: int = 10,
        scale: str = "local",
        clean: bool = True,
        prefix_section: bool = False,
        show_progress: bool = False,
    ) -> list[RetrievalResult]:
        """Retrieve and rerank papers for a citation context.

        Args:
            context: CitationContext object
            k: Number of results
            scale: Context scale ("local", "narrow", "broad", "section", "global")
            clean: If True, remove citation markers
            prefix_section: If True, prepend section heading to query
            show_progress: Whether to show progress bar

        Returns:
            Reranked retrieval results
        """
        query = context.get_query(scale, clean=clean, prefix_section=prefix_section)
        return self.retrieve(query, k=k, show_progress=show_progress)
