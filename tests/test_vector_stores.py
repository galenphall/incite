"""Tests for FAISSStore and ChunkStore."""

import numpy as np
import pytest

from incite.embeddings.chunk_store import ChunkStore
from incite.embeddings.stores import FAISSStore


class TestFAISSStoreBasic:
    @pytest.fixture
    def store_with_data(self, mock_embedder):
        store = FAISSStore(dimension=mock_embedder.dimension)
        texts = ["alpha", "beta", "gamma", "delta"]
        vecs = mock_embedder.embed(texts)
        store.add([f"d{i}" for i in range(4)], vecs)
        return store, vecs

    def test_add_and_search(self, store_with_data):
        store, vecs = store_with_data
        results = store.search(vecs[0], k=4)
        assert len(results) > 0
        ids = [r[0] for r in results]
        assert "d0" in ids
        for _, score in results:
            assert isinstance(score, float)

    def test_size_property(self, mock_embedder):
        store = FAISSStore(dimension=mock_embedder.dimension)
        assert store.size == 0
        vecs = mock_embedder.embed(["one", "two", "three", "four"])
        store.add(["a", "b", "c", "d"], vecs)
        assert store.size == 4

    def test_search_empty_store(self, mock_embedder):
        store = FAISSStore(dimension=mock_embedder.dimension)
        query = mock_embedder.embed_query("test")
        results = store.search(query, k=5)
        assert results == []

    def test_search_k_clamped(self, store_with_data):
        store, vecs = store_with_data
        results = store.search(vecs[0], k=100)
        assert len(results) == 4

    def test_search_scores_descending(self, store_with_data):
        store, vecs = store_with_data
        results = store.search(vecs[0], k=4)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_returns_best_match(self, store_with_data):
        store, vecs = store_with_data
        results = store.search(vecs[0], k=4)
        assert results[0][0] == "d0"
        assert results[0][1] == pytest.approx(1.0, abs=0.01)

    def test_dimension_set_on_first_add(self, mock_embedder):
        store = FAISSStore()
        assert store.dimension is None
        vecs = mock_embedder.embed(["test"])
        store.add(["x"], vecs)
        assert store.dimension == mock_embedder.dimension


class TestFAISSStoreIndexTypes:
    def test_flat_index_default(self, mock_embedder):
        store = FAISSStore()
        vecs = mock_embedder.embed(["a", "b"])
        store.add(["a", "b"], vecs)
        assert store.index_type == "flat"
        results = store.search(vecs[0], k=2)
        assert len(results) == 2

    def test_hnsw_index(self, mock_embedder):
        store = FAISSStore(index_type="hnsw")
        vecs = mock_embedder.embed(["a", "b", "c", "d"])
        store.add(["a", "b", "c", "d"], vecs)
        results = store.search(vecs[0], k=4)
        assert len(results) > 0
        assert results[0][0] == "a"

    @pytest.mark.skipif(
        True,  # SQ8 segfaults on ARM64 faiss-cpu with small dimensions
        reason="SQ8 index unreliable on this platform with dim=8",
    )
    def test_sq8_index(self, mock_embedder):
        store = FAISSStore(index_type="sq8")
        vecs = mock_embedder.embed(["a", "b", "c", "d"])
        store.add(["a", "b", "c", "d"], vecs)
        results = store.search(vecs[0], k=4)
        assert len(results) > 0


class TestFAISSStoreSaveLoad:
    def test_save_load_roundtrip(self, mock_embedder, tmp_path):
        store = FAISSStore(dimension=mock_embedder.dimension)
        vecs = mock_embedder.embed(["cat", "dog", "fish"])
        store.add(["c", "d", "f"], vecs)

        store.save(tmp_path / "idx")

        store2 = FAISSStore()
        store2.load(tmp_path / "idx")

        query = mock_embedder.embed_query("cat")
        r1 = store.search(query, k=3)
        r2 = store2.search(query, k=3)
        assert [x[0] for x in r1] == [x[0] for x in r2]

    def test_save_creates_files(self, mock_embedder, tmp_path):
        store = FAISSStore(dimension=mock_embedder.dimension)
        vecs = mock_embedder.embed(["x"])
        store.add(["x"], vecs)
        store.save(tmp_path / "idx")
        assert (tmp_path / "idx" / "index.faiss").exists()
        assert (tmp_path / "idx" / "id_map.json").exists()

    def test_load_nonexistent_graceful(self, tmp_path):
        store = FAISSStore()
        store.load(tmp_path / "nonexistent")
        assert store.size == 0


class TestFAISSStoreTwoStage:
    def test_fallback_without_truncated(self, mock_embedder):
        store = FAISSStore(dimension=mock_embedder.dimension)
        vecs = mock_embedder.embed(["a", "b", "c"])
        store.add(["a", "b", "c"], vecs)
        results = store.search_two_stage(vecs[0], k=3)
        assert len(results) > 0

    def test_build_and_two_stage_search(self, mock_embedder):
        store = FAISSStore(dimension=mock_embedder.dimension)
        vecs = mock_embedder.embed(["a", "b", "c", "d"])
        store.add(["a", "b", "c", "d"], vecs)
        store.build_truncated_index(truncate_dim=4, index_type="flat")
        results = store.search_two_stage(vecs[0], k=4, truncate_dim=4)
        assert len(results) > 0
        assert results[0][0] == "a"

    def test_truncated_saved_and_loaded(self, mock_embedder, tmp_path):
        store = FAISSStore(dimension=mock_embedder.dimension)
        vecs = mock_embedder.embed(["a", "b", "c"])
        store.add(["a", "b", "c"], vecs)
        store.build_truncated_index(truncate_dim=4, index_type="flat")
        store.save(tmp_path / "idx")
        assert (tmp_path / "idx" / "index_truncated.faiss").exists()

        store2 = FAISSStore()
        store2.load(tmp_path / "idx")
        results = store2.search_two_stage(vecs[0], k=3, truncate_dim=4)
        assert len(results) > 0


class TestChunkStore:
    def test_add_chunks_and_search(self, mock_embedder, sample_chunks):
        store = ChunkStore(dimension=mock_embedder.dimension)
        texts = [c.text for c in sample_chunks]
        vecs = mock_embedder.embed(texts)
        store.add_chunks(sample_chunks, vecs)
        query = mock_embedder.embed_query("sea level rise")
        results = store.search_with_papers(query, k=5)
        assert len(results) > 0
        for chunk_id, paper_id, score in results:
            assert isinstance(chunk_id, str)
            assert isinstance(paper_id, str)
            assert isinstance(score, float)

    def test_get_paper_id(self, mock_embedder, sample_chunks):
        store = ChunkStore(dimension=mock_embedder.dimension)
        vecs = mock_embedder.embed([c.text for c in sample_chunks])
        store.add_chunks(sample_chunks, vecs)
        assert store.get_paper_id("climate_1::chunk_0") == "climate_1"
        assert store.get_paper_id("nonexistent") is None

    def test_paper_ids_property(self, mock_embedder, sample_chunks):
        store = ChunkStore(dimension=mock_embedder.dimension)
        vecs = mock_embedder.embed([c.text for c in sample_chunks])
        store.add_chunks(sample_chunks, vecs)
        pids = store.paper_ids
        assert isinstance(pids, set)
        assert "climate_1" in pids
        assert "nlp_2" in pids

    def test_chunks_for_paper(self, mock_embedder, sample_chunks):
        store = ChunkStore(dimension=mock_embedder.dimension)
        vecs = mock_embedder.embed([c.text for c in sample_chunks])
        store.add_chunks(sample_chunks, vecs)
        cids = store.chunks_for_paper("climate_1")
        assert set(cids) == {"climate_1::chunk_0", "climate_1::chunk_1"}

    def test_save_load_roundtrip(self, mock_embedder, sample_chunks, tmp_path):
        store = ChunkStore(dimension=mock_embedder.dimension)
        vecs = mock_embedder.embed([c.text for c in sample_chunks])
        store.add_chunks(sample_chunks, vecs)
        store.save(tmp_path / "cs")

        store2 = ChunkStore()
        store2.load(tmp_path / "cs")
        assert (tmp_path / "cs" / "chunk_to_paper.json").exists()
        assert store2.get_paper_id("nlp_1::chunk_0") == "nlp_1"
        assert store2.size == store.size
