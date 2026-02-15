"""Tests for ChunkStore reverse index and scoped search."""

import tempfile

import numpy as np
import pytest

from incite.embeddings.chunk_store import ChunkStore
from incite.models import Chunk


def _make_chunks(n_papers: int = 3, chunks_per_paper: int = 4) -> list[Chunk]:
    """Create test chunks with known paper assignments."""
    chunks = []
    for p in range(n_papers):
        paper_id = f"paper_{p}"
        for c in range(chunks_per_paper):
            chunks.append(
                Chunk(
                    id=f"{paper_id}::chunk_{c}",
                    paper_id=paper_id,
                    text=f"Chunk {c} of paper {p}",
                )
            )
    return chunks


def _make_embeddings(chunks: list[Chunk], dim: int = 8) -> np.ndarray:
    """Create deterministic embeddings — each chunk gets a unique direction."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((len(chunks), dim)).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def store_with_chunks():
    """Return (store, chunks, embeddings) with 3 papers x 4 chunks."""
    chunks = _make_chunks(n_papers=3, chunks_per_paper=4)
    embs = _make_embeddings(chunks)
    store = ChunkStore(dimension=embs.shape[1])
    store.add_chunks(chunks, embs)
    return store, chunks, embs


class TestReverseIndex:
    def test_paper_to_chunks_populated(self, store_with_chunks):
        store, chunks, _ = store_with_chunks
        assert len(store._paper_to_chunks) == 3
        for p in range(3):
            paper_id = f"paper_{p}"
            assert paper_id in store._paper_to_chunks
            assert len(store._paper_to_chunks[paper_id]) == 4

    def test_chunks_for_paper_uses_reverse_index(self, store_with_chunks):
        store, _, _ = store_with_chunks
        ids = store.chunks_for_paper("paper_1")
        assert len(ids) == 4
        assert all("paper_1" in cid for cid in ids)

    def test_chunks_for_paper_empty(self, store_with_chunks):
        store, _, _ = store_with_chunks
        assert store.chunks_for_paper("nonexistent") == []

    def test_save_load_preserves_reverse_index(self, store_with_chunks):
        store, _, _ = store_with_chunks
        with tempfile.TemporaryDirectory() as tmpdir:
            store.save(tmpdir)
            loaded = ChunkStore()
            loaded.load(tmpdir)

            assert len(loaded._paper_to_chunks) == 3
            for p in range(3):
                paper_id = f"paper_{p}"
                assert len(loaded._paper_to_chunks[paper_id]) == 4
                # chunk IDs should match
                assert set(loaded.chunks_for_paper(paper_id)) == set(
                    store.chunks_for_paper(paper_id)
                )


class TestSearchWithinPapers:
    def test_returns_only_specified_papers(self, store_with_chunks):
        store, chunks, embs = store_with_chunks
        query = embs[0]  # Use first chunk's embedding as query
        results = store.search_within_papers(query, ["paper_0"], top_per_paper=3)

        assert set(results.keys()) == {"paper_0"}
        for chunk_id, _ in results["paper_0"]:
            assert "paper_0" in chunk_id

    def test_multiple_papers(self, store_with_chunks):
        store, _, embs = store_with_chunks
        query = embs[0]
        results = store.search_within_papers(query, ["paper_0", "paper_2"], top_per_paper=2)

        assert set(results.keys()) == {"paper_0", "paper_2"}
        assert len(results["paper_0"]) <= 2
        assert len(results["paper_2"]) <= 2

    def test_scores_are_descending(self, store_with_chunks):
        store, _, embs = store_with_chunks
        query = embs[0]
        results = store.search_within_papers(query, ["paper_0"], top_per_paper=4)

        scores = [s for _, s in results["paper_0"]]
        assert scores == sorted(scores, reverse=True)

    def test_missing_paper_omitted(self, store_with_chunks):
        store, _, embs = store_with_chunks
        query = embs[0]
        results = store.search_within_papers(query, ["paper_0", "nonexistent"], top_per_paper=3)

        assert "nonexistent" not in results
        assert "paper_0" in results

    def test_top_per_paper_limits_results(self, store_with_chunks):
        store, _, embs = store_with_chunks
        query = embs[0]
        results = store.search_within_papers(query, ["paper_0"], top_per_paper=1)

        assert len(results["paper_0"]) == 1

    def test_empty_store(self):
        store = ChunkStore(dimension=8)
        query = np.ones(8, dtype=np.float32)
        results = store.search_within_papers(query, ["paper_0"])
        assert results == {}

    def test_2d_query_accepted(self, store_with_chunks):
        store, _, embs = store_with_chunks
        query_2d = embs[0].reshape(1, -1)
        results = store.search_within_papers(query_2d, ["paper_0"], top_per_paper=2)
        assert "paper_0" in results

    def test_scores_are_dot_products(self, store_with_chunks):
        """Scoped scores should be dot products with the query embedding."""
        store, chunks, embs = store_with_chunks
        query = embs[0]

        # Scoped search
        scoped = store.search_within_papers(query, ["paper_0"], top_per_paper=4)

        # Verify scores match manually computed dot products
        for chunk_id, scoped_score in scoped["paper_0"]:
            idx = store._id_to_idx[chunk_id]
            chunk_emb = embs[idx]
            expected = float(np.dot(query, chunk_emb))
            np.testing.assert_almost_equal(scoped_score, expected, decimal=5)
