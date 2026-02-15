"""Tests for TwoStageRetriever."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from incite.embeddings.chunk_store import ChunkStore
from incite.models import Chunk, RetrievalResult
from incite.retrieval.two_stage import TwoStageRetriever


def _make_mock_paper_retriever(stage1_results: list[RetrievalResult]):
    """Create a mock paper retriever returning fixed results."""
    mock = MagicMock()
    mock.retrieve.return_value = (stage1_results, {"embed_query_ms": 1.0, "vector_search_ms": 2.0})
    return mock


def _make_mock_embedder(dim: int = 8):
    """Create a mock embedder."""
    mock = MagicMock()
    mock.dimension = dim
    mock.embed_query.return_value = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
    return mock


def _make_chunk_store_with_data(papers_and_scores: dict[str, list[float]], dim: int = 8):
    """Create a ChunkStore with known chunks and embeddings.

    Args:
        papers_and_scores: {paper_id: [score1, score2, ...]} — desired dot products
                          with a uniform query vector.
        dim: Embedding dimension

    Returns:
        (store, chunks_dict) tuple
    """
    query = np.ones(dim, dtype=np.float32) / np.sqrt(dim)

    chunks = []
    all_embeddings = []
    for paper_id, desired_scores in papers_and_scores.items():
        for i, target_score in enumerate(desired_scores):
            chunk_id = f"{paper_id}::chunk_{i}"
            chunks.append(
                Chunk(id=chunk_id, paper_id=paper_id, text=f"Text for chunk {i} of {paper_id}")
            )
            # Create embedding that gives desired dot product with uniform query
            # For unit query q = [1/sqrt(d), ...], dot(q, v) = sum(v) / sqrt(d)
            # So we need sum(v) = target_score * sqrt(d)
            emb = np.zeros(dim, dtype=np.float32)
            emb[0] = target_score * np.sqrt(dim)
            # Spread remaining values to avoid degenerate vectors
            all_embeddings.append(emb)

    store = ChunkStore(dimension=dim)
    if chunks:
        store.add_chunks(chunks, np.array(all_embeddings, dtype=np.float32))

    chunks_dict = {c.id: c for c in chunks}
    return store, chunks_dict


@pytest.fixture
def basic_two_stage():
    """Create a basic TwoStageRetriever with 3 papers (2 with chunks, 1 without)."""
    # Stage 1 results: paper_a (best), paper_b (middle), paper_c (worst)
    stage1_results = [
        RetrievalResult(paper_id="paper_a", score=0.9, rank=1, score_breakdown={"neural": 0.8}),
        RetrievalResult(paper_id="paper_b", score=0.7, rank=2, score_breakdown={"neural": 0.6}),
        RetrievalResult(paper_id="paper_c", score=0.5, rank=3, score_breakdown={"neural": 0.4}),
    ]

    mock_retriever = _make_mock_paper_retriever(stage1_results)
    mock_embedder = _make_mock_embedder()

    # paper_a has mediocre chunks, paper_b has excellent chunks, paper_c has no chunks
    store, chunks = _make_chunk_store_with_data(
        {
            "paper_a": [0.3, 0.2],
            "paper_b": [0.9, 0.7, 0.4],
            # paper_c: no chunks
        }
    )

    retriever = TwoStageRetriever(
        paper_retriever=mock_retriever,
        chunk_store=store,
        chunks=chunks,
        embedder=mock_embedder,
        alpha=0.6,
        stage1_k=50,
    )

    return retriever, stage1_results


class TestTwoStageRetriever:
    def test_basic_retrieve(self, basic_two_stage):
        retriever, _ = basic_two_stage
        results = retriever.retrieve("test query", k=3)
        assert len(results) == 3
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_ranks_assigned_correctly(self, basic_two_stage):
        retriever, _ = basic_two_stage
        results = retriever.retrieve("test query", k=3)
        for i, r in enumerate(results):
            assert r.rank == i + 1

    def test_scores_descending(self, basic_two_stage):
        retriever, _ = basic_two_stage
        results = retriever.retrieve("test query", k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_alpha_1_preserves_paper_order(self):
        """With alpha=1.0, chunk scores are ignored; paper order preserved."""
        stage1_results = [
            RetrievalResult(paper_id="paper_a", score=0.9, rank=1, score_breakdown={}),
            RetrievalResult(paper_id="paper_b", score=0.7, rank=2, score_breakdown={}),
        ]
        mock_retriever = _make_mock_paper_retriever(stage1_results)
        mock_embedder = _make_mock_embedder()
        store, chunks = _make_chunk_store_with_data({"paper_a": [0.1], "paper_b": [0.99]})

        retriever = TwoStageRetriever(
            paper_retriever=mock_retriever,
            chunk_store=store,
            chunks=chunks,
            embedder=mock_embedder,
            alpha=1.0,
            stage1_k=10,
        )

        results = retriever.retrieve("test", k=2)
        assert results[0].paper_id == "paper_a"
        assert results[1].paper_id == "paper_b"

    def test_alpha_0_uses_only_chunks(self):
        """With alpha=0.0, only chunk scores matter."""
        stage1_results = [
            RetrievalResult(paper_id="paper_a", score=0.9, rank=1, score_breakdown={}),
            RetrievalResult(paper_id="paper_b", score=0.1, rank=2, score_breakdown={}),
        ]
        mock_retriever = _make_mock_paper_retriever(stage1_results)
        mock_embedder = _make_mock_embedder()
        store, chunks = _make_chunk_store_with_data({"paper_a": [0.1], "paper_b": [0.99]})

        retriever = TwoStageRetriever(
            paper_retriever=mock_retriever,
            chunk_store=store,
            chunks=chunks,
            embedder=mock_embedder,
            alpha=0.0,
            stage1_k=10,
        )

        results = retriever.retrieve("test", k=2)
        # paper_b should be first because it has the highest chunk score
        assert results[0].paper_id == "paper_b"

    def test_papers_without_chunks_still_appear(self, basic_two_stage):
        retriever, _ = basic_two_stage
        results = retriever.retrieve("test query", k=3)
        paper_ids = {r.paper_id for r in results}
        assert "paper_c" in paper_ids

    def test_papers_without_chunks_penalized(self, basic_two_stage):
        """Papers without chunks get lower scores than those with good chunks."""
        retriever, _ = basic_two_stage
        results = retriever.retrieve("test query", k=3)
        result_map = {r.paper_id: r for r in results}
        # paper_c has no chunks, should have lower score
        assert result_map["paper_c"].score < result_map["paper_b"].score

    def test_evidence_attached(self, basic_two_stage):
        retriever, _ = basic_two_stage
        results = retriever.retrieve("test query", k=3)
        result_map = {r.paper_id: r for r in results}

        # paper_b has high chunk scores above threshold
        b = result_map["paper_b"]
        assert b.matched_paragraphs  # Should have evidence
        assert b.matched_paragraph is not None

        # paper_c has no chunks
        c = result_map["paper_c"]
        assert not c.matched_paragraphs
        assert c.matched_paragraph is None

    def test_score_breakdown_populated(self, basic_two_stage):
        retriever, _ = basic_two_stage
        results = retriever.retrieve("test query", k=3)
        for r in results:
            assert "paper_score_norm" in r.score_breakdown
            assert "best_chunk_score" in r.score_breakdown
            assert "alpha" in r.score_breakdown
            assert "num_chunks_matched" in r.score_breakdown

    def test_return_timing(self, basic_two_stage):
        retriever, _ = basic_two_stage
        results, timing = retriever.retrieve("test query", k=3, return_timing=True)
        assert "stage1_ms" in timing
        assert "stage2_ms" in timing
        assert "rerank_ms" in timing

    def test_k_limits_results(self, basic_two_stage):
        retriever, _ = basic_two_stage
        results = retriever.retrieve("test query", k=2)
        assert len(results) == 2

    def test_empty_stage1(self):
        """Empty stage 1 results should return empty list."""
        mock_retriever = _make_mock_paper_retriever([])
        mock_embedder = _make_mock_embedder()
        store, chunks = _make_chunk_store_with_data({})

        retriever = TwoStageRetriever(
            paper_retriever=mock_retriever,
            chunk_store=store,
            chunks=chunks,
            embedder=mock_embedder,
        )

        results = retriever.retrieve("test", k=5)
        assert results == []

    def test_pre_computed_query_embedding(self, basic_two_stage):
        """Pre-computed query embedding should be passed to stage 1."""
        retriever, _ = basic_two_stage
        pre_emb = np.ones(8, dtype=np.float32)
        retriever.retrieve("test", k=3, query_embedding=pre_emb)

        # Verify the pre-computed embedding was passed to stage 1
        call_kwargs = retriever.paper_retriever.retrieve.call_args
        assert "query_embedding" in call_kwargs.kwargs

    def test_evidence_threshold_respected(self):
        """Chunks below threshold should not appear as evidence."""
        stage1_results = [
            RetrievalResult(paper_id="paper_a", score=0.9, rank=1, score_breakdown={}),
        ]
        mock_retriever = _make_mock_paper_retriever(stage1_results)
        mock_embedder = _make_mock_embedder()

        # Chunk with very low score (below default threshold of 0.35)
        store, chunks = _make_chunk_store_with_data({"paper_a": [0.1]})

        retriever = TwoStageRetriever(
            paper_retriever=mock_retriever,
            chunk_store=store,
            chunks=chunks,
            embedder=mock_embedder,
            evidence_threshold=0.35,
        )

        results = retriever.retrieve("test", k=1)
        # Score should still be computed, but evidence not attached
        assert results[0].score_breakdown["best_chunk_score"] > 0
        assert not results[0].matched_paragraphs
