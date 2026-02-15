"""Multi-scale retriever searching paper, paragraph, and sentence indexes.

This retriever searches at three levels of granularity:
1. Paper (Title + Abstract)
2. Paragraph (Chunked full text)
3. Sentence (Sentence with context)

It fuses the results using RRF and attaches the best available evidence snippet.
"""

import time
from typing import Union

from incite.interfaces import Retriever
from incite.models import RetrievalResult
from incite.retrieval.hybrid import rrf_fuse, rrf_sort
from incite.utils import compute_confidence


class MultiScaleRetriever(Retriever):
    """Retriever that fuses results from paper, paragraph, and sentence levels."""

    def __init__(
        self,
        paper_retriever: Retriever,
        paragraph_retriever: Retriever,
        sentence_retriever: Retriever,
        weights: dict[str, float] = None,
        rrf_k: int = 60,
    ):
        """Initialize multi-scale retriever.

        Args:
            paper_retriever: Searches paper-level index
            paragraph_retriever: Searches paragraph-level index
            sentence_retriever: Searches sentence-level index
            weights: Dict of weights for each level (default: 1.0 for all)
            rrf_k: RRF constant (default 60)
        """
        self.paper_retriever = paper_retriever
        self.paragraph_retriever = paragraph_retriever
        self.sentence_retriever = sentence_retriever
        self.weights = weights or {"paper": 1.0, "paragraph": 1.0, "sentence": 1.0}
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        k: int = 10,
        return_timing: bool = False,
        **kwargs,
    ) -> Union[list[RetrievalResult], tuple[list[RetrievalResult], dict]]:
        """Retrieve top-k papers by fusing multi-scale search results.

        Args:
            query: Query text
            k: Number of results to return
            return_timing: If True, return (results, timing_dict)

        Returns:
            List of RetrievalResult. matched_paragraph will contain the best
            evidence snippet found across paragraph and sentence levels.
        """
        timing = {}

        # 1. Fetch results from all tiers (fetch more than k for effective fusion)
        fetch_k = k * 3

        # Common kwargs (e.g. query_embedding if provided)
        sub_kwargs = {}
        if "query_embedding" in kwargs:
            sub_kwargs["query_embedding"] = kwargs["query_embedding"]

        # Paper Level
        t0 = time.perf_counter()
        paper_results = self.paper_retriever.retrieve(query, k=fetch_k, **sub_kwargs)
        timing["paper_ms"] = (time.perf_counter() - t0) * 1000

        # Paragraph Level
        t0 = time.perf_counter()
        para_results = self.paragraph_retriever.retrieve(query, k=fetch_k, **sub_kwargs)
        timing["paragraph_ms"] = (time.perf_counter() - t0) * 1000

        # Sentence Level
        t0 = time.perf_counter()
        sent_results = self.sentence_retriever.retrieve(query, k=fetch_k, **sub_kwargs)
        timing["sentence_ms"] = (time.perf_counter() - t0) * 1000

        # 2. Fuse Results (RRF)
        t0 = time.perf_counter()

        fused = rrf_fuse(
            [
                (paper_results, self.weights.get("paper", 1.0), "paper"),
                (para_results, self.weights.get("paragraph", 1.0), "paragraph"),
                (sent_results, self.weights.get("sentence", 1.0), "sentence"),
            ],
            rrf_k=self.rrf_k,
        )

        # Extract best evidence per paper from chunk-level tiers
        for pid, data in fused.items():
            best_evidence = None
            for tier_name in ("paragraph", "sentence"):
                tier_result = data["best_result"].get(tier_name)
                if tier_result and tier_result.matched_paragraph:
                    chunk_score = tier_result.score_breakdown.get("best_chunk_score", 0.0)
                    if best_evidence is None or chunk_score > best_evidence["score"]:
                        best_evidence = {
                            "score": chunk_score,
                            "text": tier_result.matched_paragraph,
                            "tier": tier_name,
                        }
            data["evidence"] = best_evidence

        # 3. Sort and Format
        sorted_ids = rrf_sort(fused)

        final_results = []
        for rank, pid in enumerate(sorted_ids[:k]):
            data = fused[pid]
            best_ev = data["evidence"]
            matched_para = best_ev["text"] if best_ev else None

            breakdown = data["scores"]
            if best_ev:
                breakdown["best_evidence_score"] = best_ev["score"]
                breakdown["best_evidence_tier"] = best_ev["tier"]

            final_results.append(
                RetrievalResult(
                    paper_id=pid,
                    score=data["total_score"],
                    rank=rank + 1,
                    score_breakdown=breakdown,
                    matched_paragraph=matched_para,
                    confidence=compute_confidence(breakdown, mode="multi_scale"),
                )
            )

        timing["fusion_ms"] = (time.perf_counter() - t0) * 1000

        if return_timing:
            return final_results, timing
        return final_results
