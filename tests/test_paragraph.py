"""Tests for ParagraphRetriever and HybridParagraphRetriever."""

import pytest

from incite.models import Chunk
from incite.retrieval.paragraph import (
    HybridParagraphRetriever,
    ParagraphRetriever,
    _highlight_sentence_in_parent,
)


class TestHighlightSentence:
    def test_highlight_in_parent(self):
        chunk = Chunk(
            id="p1::chunk_0",
            paper_id="p1",
            text="Sea levels are rising.",
            parent_text="Sea levels are rising. This is a major concern.",
        )
        result = _highlight_sentence_in_parent(chunk)
        assert "**Sea levels are rising.**" in result
        assert "This is a major concern." in result

    def test_highlight_fallback_no_parent(self):
        chunk = Chunk(id="p1::chunk_0", paper_id="p1", text="Some text here.")
        result = _highlight_sentence_in_parent(chunk)
        assert result == "Some text here."

    def test_highlight_not_found(self):
        chunk = Chunk(
            id="p1::chunk_0",
            paper_id="p1",
            text="This text is not in parent.",
            parent_text="Completely different parent text content.",
        )
        result = _highlight_sentence_in_parent(chunk)
        assert result == "This text is not in parent."


class TestParagraphRetriever:
    @pytest.fixture(scope="class")
    def para_retriever(self, mock_embedder, sample_chunks):
        return ParagraphRetriever.from_chunks(
            sample_chunks, mock_embedder, show_progress=False
        )

    def test_retrieve_returns_results(self, para_retriever):
        results = para_retriever.retrieve("sea level rise", k=5)
        assert len(results) > 0
        for r in results:
            assert r.paper_id

    def test_scores_descending(self, para_retriever):
        results = para_retriever.retrieve("ocean temperature", k=6)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_sequential(self, para_retriever):
        results = para_retriever.retrieve("deep learning", k=6)
        for i, r in enumerate(results):
            assert r.rank == i + 1

    def test_k_limits_output(self, para_retriever):
        results = para_retriever.retrieve("climate", k=2)
        assert len(results) <= 2

    def test_matched_paragraph_populated(self, para_retriever):
        results = para_retriever.retrieve("sea level rise", k=6)
        has_paragraph = any(r.matched_paragraph is not None for r in results)
        assert has_paragraph

    def test_score_breakdown_has_chunk_info(self, para_retriever):
        results = para_retriever.retrieve("ocean heat", k=3)
        for r in results:
            assert "best_chunk_score" in r.score_breakdown
            assert "num_chunks_matched" in r.score_breakdown

    def test_confidence_populated(self, para_retriever):
        results = para_retriever.retrieve("climate models", k=3)
        for r in results:
            assert r.confidence >= 0.0

    def test_return_timing(self, para_retriever):
        results, timing = para_retriever.retrieve(
            "sea level", k=3, return_timing=True
        )
        assert isinstance(results, list)
        assert isinstance(timing, dict)
        assert "embed_query_ms" in timing
        assert "vector_search_ms" in timing

    def test_aggregation_max(self, mock_embedder, sample_chunks):
        retriever = ParagraphRetriever.from_chunks(
            sample_chunks, mock_embedder, aggregation="max", show_progress=False
        )
        results = retriever.retrieve("climate", k=3)
        assert len(results) > 0

    def test_aggregation_mean(self, mock_embedder, sample_chunks):
        retriever = ParagraphRetriever.from_chunks(
            sample_chunks, mock_embedder, aggregation="mean", show_progress=False
        )
        results = retriever.retrieve("climate", k=3)
        assert len(results) > 0

    def test_aggregation_sum(self, mock_embedder, sample_chunks):
        retriever = ParagraphRetriever.from_chunks(
            sample_chunks, mock_embedder, aggregation="sum", show_progress=False
        )
        results = retriever.retrieve("climate", k=3)
        assert len(results) > 0

    def test_aggregation_coverage(self, mock_embedder, sample_chunks):
        retriever = ParagraphRetriever.from_chunks(
            sample_chunks, mock_embedder, aggregation="coverage", show_progress=False
        )
        results = retriever.retrieve("climate", k=3)
        assert len(results) > 0

    def test_from_chunks_builds_store(self, para_retriever, sample_chunks):
        assert para_retriever.chunk_store.size == len(sample_chunks)


class TestHybridParagraphRetriever:
    @pytest.fixture(scope="class")
    def hybrid_para_retriever(self, mock_embedder, sample_chunks, sample_papers):
        return HybridParagraphRetriever.from_chunks_and_papers(
            sample_chunks, sample_papers, mock_embedder, show_progress=False
        )

    def test_retrieve_returns_results(self, hybrid_para_retriever):
        results = hybrid_para_retriever.retrieve("sea level rise", k=5)
        assert len(results) > 0

    def test_fuses_neural_and_bm25(self, hybrid_para_retriever):
        results = hybrid_para_retriever.retrieve("sea level rise climate", k=5)
        breakdown_keys = set()
        for r in results:
            breakdown_keys.update(r.score_breakdown.keys())
        assert "neural_rank" in breakdown_keys or "neural_score" in breakdown_keys
        assert "bm25_rank" in breakdown_keys or "bm25_score" in breakdown_keys
