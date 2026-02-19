"""Tests for PostgreSQL + pgvector vector stores.

Tests the PgVectorStore and PgVectorChunkStore classes using mocked
database connections. Verifies SQL correctness, score conventions,
and multi-tenant isolation.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def mock_conn():
    """Create a mock psycopg connection."""
    conn = MagicMock()
    conn.info = MagicMock()  # Marks it as a psycopg connection
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, cursor


@pytest.fixture
def paper_store(mock_conn):
    """Create a PgVectorStore with mocked DB."""
    conn, cursor = mock_conn
    with patch("cloud.pgvector_store.PgVectorStore._get_conn", return_value=conn):
        from cloud.pgvector_store import PgVectorStore

        store = PgVectorStore(
            conninfo="postgresql://test:test@localhost/test",
            embedder_type="granite-ft",
            library_id=1,
            dimension=384,
        )
        yield store, conn, cursor


@pytest.fixture
def chunk_store(mock_conn):
    """Create a PgVectorChunkStore with mocked DB."""
    conn, cursor = mock_conn
    with patch("cloud.pgvector_store.PgVectorChunkStore._get_conn", return_value=conn):
        from cloud.pgvector_store import PgVectorChunkStore

        store = PgVectorChunkStore(
            conninfo="postgresql://test:test@localhost/test",
            embedder_type="granite-ft",
            library_id=1,
            dimension=384,
        )
        yield store, conn, cursor


class TestPgVectorStore:
    """Paper-level vector store tests."""

    def test_add_upserts_to_paper_vectors(self, paper_store):
        store, conn, cursor = paper_store

        ids = ["paper_001", "paper_002"]
        embeddings = np.random.randn(2, 384).astype(np.float32)

        store.add(ids, embeddings)

        # Should execute a single executemany call with 2 rows
        cursor.executemany.assert_called_once()

        # Verify SQL contains paper_vectors table and ON CONFLICT
        call_args = cursor.executemany.call_args
        sql = call_args[0][0]
        assert "paper_vectors" in sql
        assert "ON CONFLICT" in sql

        # Verify params list contains both papers
        params = call_args[0][1]
        assert len(params) == 2
        assert params[0][0] == "paper_001"
        assert params[0][1] == "granite-ft"
        assert params[1][0] == "paper_002"

        conn.commit.assert_called_once()

    def test_add_empty_is_noop(self, paper_store):
        store, conn, cursor = paper_store
        store.add([], np.array([]))
        cursor.execute.assert_not_called()

    def test_add_length_mismatch_raises(self, paper_store):
        store, conn, cursor = paper_store
        with pytest.raises(ValueError, match="Length mismatch"):
            store.add(["paper_001"], np.random.randn(2, 384))

    def test_search_returns_negated_scores(self, paper_store):
        store, conn, cursor = paper_store

        # pgvector <#> returns negative inner product (lower = more similar)
        # Mock: paper_001 has neg_score=-0.9 (very similar), paper_002 has -0.5
        cursor.fetchall.return_value = [
            ("paper_001", -0.9),
            ("paper_002", -0.5),
        ]

        query = np.random.randn(384).astype(np.float32)
        results = store.search(query, k=10)

        # Scores should be negated to positive (FAISS convention)
        assert results[0] == ("paper_001", 0.9)
        assert results[1] == ("paper_002", 0.5)

    def test_search_joins_library_papers(self, paper_store):
        store, conn, cursor = paper_store
        cursor.fetchall.return_value = []

        query = np.random.randn(384).astype(np.float32)
        store.search(query, k=5)

        sql = cursor.execute.call_args[0][0]
        assert "library_papers" in sql
        assert "JOIN" in sql

        # Verify library_id is scoped
        params = cursor.execute.call_args[0][1]
        assert 1 in params  # library_id
        assert "granite-ft" in params  # embedder_type

    def test_search_handles_2d_query(self, paper_store):
        """Query embedding with shape (1, 384) should work."""
        store, conn, cursor = paper_store
        cursor.fetchall.return_value = []

        query = np.random.randn(1, 384).astype(np.float32)
        store.search(query, k=5)
        # Should not raise

    def test_save_load_are_noops(self, paper_store):
        store, conn, cursor = paper_store
        store.save("/tmp/test")
        store.load("/tmp/test")
        # No exceptions, no DB calls

    def test_size_counts_library_papers(self, paper_store):
        store, conn, cursor = paper_store
        cursor.fetchone.return_value = (42,)

        assert store.size == 42

        sql = cursor.execute.call_args[0][0]
        assert "COUNT" in sql
        assert "library_papers" in sql


class TestPgVectorChunkStore:
    """Chunk-level vector store tests."""

    def test_add_chunks_upserts(self, chunk_store):
        store, conn, cursor = chunk_store

        # Create mock Chunk objects
        chunks = [
            MagicMock(id="chunk_001", paper_id="paper_001", text="Some text", section="intro"),
            MagicMock(id="chunk_002", paper_id="paper_001", text="More text", section="methods"),
        ]
        embeddings = np.random.randn(2, 384).astype(np.float32)

        store.add_chunks(chunks, embeddings)

        # Should execute a single executemany call with 2 rows
        cursor.executemany.assert_called_once()

        sql = cursor.executemany.call_args[0][0]
        assert "chunk_vectors" in sql
        assert "ON CONFLICT" in sql

        params = cursor.executemany.call_args[0][1]
        assert len(params) == 2

        conn.commit.assert_called_once()

    def test_add_chunks_empty_is_noop(self, chunk_store):
        store, conn, cursor = chunk_store
        store.add_chunks([], np.array([]))
        cursor.execute.assert_not_called()

    def test_search_with_papers_returns_triples(self, chunk_store):
        store, conn, cursor = chunk_store

        cursor.fetchall.return_value = [
            ("chunk_001", "paper_001", -0.85),
            ("chunk_002", "paper_002", -0.72),
        ]

        query = np.random.randn(384).astype(np.float32)
        results = store.search_with_papers(query, k=10)

        assert len(results) == 2
        assert results[0] == ("chunk_001", "paper_001", 0.85)
        assert results[1] == ("chunk_002", "paper_002", 0.72)

    def test_search_within_papers_groups_by_paper(self, chunk_store):
        store, conn, cursor = chunk_store

        # Return 4 chunks across 2 papers, sorted by neg_score ASC
        cursor.fetchall.return_value = [
            ("c1", "p1", -0.9),
            ("c2", "p1", -0.8),
            ("c3", "p1", -0.7),
            ("c4", "p1", -0.6),
            ("c5", "p2", -0.85),
            ("c6", "p2", -0.75),
        ]

        query = np.random.randn(384).astype(np.float32)
        results = store.search_within_papers(query, paper_ids=["p1", "p2"], top_per_paper=2)

        # Should keep top 2 per paper
        assert len(results["p1"]) == 2
        assert len(results["p2"]) == 2
        assert results["p1"][0] == ("c1", 0.9)
        assert results["p1"][1] == ("c2", 0.8)
        assert results["p2"][0] == ("c5", 0.85)

    def test_search_within_papers_empty_returns_empty(self, chunk_store):
        store, conn, cursor = chunk_store
        results = store.search_within_papers(
            np.random.randn(384).astype(np.float32),
            paper_ids=[],
        )
        assert results == {}

    def test_search_within_papers_uses_any(self, chunk_store):
        store, conn, cursor = chunk_store
        cursor.fetchall.return_value = []

        query = np.random.randn(384).astype(np.float32)
        store.search_within_papers(query, paper_ids=["p1", "p2"], top_per_paper=3)

        sql = cursor.execute.call_args[0][0]
        assert "ANY" in sql

    def test_get_paper_id(self, chunk_store):
        store, conn, cursor = chunk_store
        cursor.fetchone.return_value = ("paper_001",)

        result = store.get_paper_id("chunk_001")
        assert result == "paper_001"

    def test_get_paper_id_not_found(self, chunk_store):
        store, conn, cursor = chunk_store
        cursor.fetchone.return_value = None

        result = store.get_paper_id("nonexistent")
        assert result is None

    def test_chunks_for_paper(self, chunk_store):
        store, conn, cursor = chunk_store
        cursor.fetchall.return_value = [("c1",), ("c2",), ("c3",)]

        result = store.chunks_for_paper("paper_001")
        assert result == ["c1", "c2", "c3"]

    def test_paper_ids_property(self, chunk_store):
        store, conn, cursor = chunk_store
        cursor.fetchall.return_value = [("p1",), ("p2",), ("p3",)]

        result = store.paper_ids
        assert result == {"p1", "p2", "p3"}


class TestFactoryIntegration:
    """Test that factory functions accept storage_backend parameter."""

    def test_create_retriever_accepts_storage_backend(self):
        """Verify create_retriever can use a pre-built store."""
        from incite.models import Paper
        from incite.retrieval.factory import create_retriever

        # Create minimal papers
        papers = [
            Paper(id="p1", title="Test Paper", abstract="Test abstract"),
        ]

        # Mock store that satisfies VectorStore protocol
        mock_store = MagicMock()
        mock_store.search.return_value = [("p1", 0.9)]
        mock_store.size = 1

        retriever = create_retriever(
            papers=papers,
            method="neural",
            embedder_type="minilm",
            storage_backend=mock_store,
        )

        # Should have created a NeuralRetriever with the mock store
        assert retriever.store is mock_store

    def test_create_paragraph_retriever_accepts_storage_backend(self):
        """Verify create_paragraph_retriever can use a pre-built chunk store."""
        from incite.models import Chunk, Paper
        from incite.retrieval.factory import create_paragraph_retriever

        papers = [
            Paper(id="p1", title="Test Paper", abstract="Test abstract"),
        ]
        chunks = [
            Chunk(id="c1", paper_id="p1", text="Some text"),
        ]

        mock_store = MagicMock()
        mock_store.search_with_papers.return_value = [("c1", "p1", 0.9)]

        retriever = create_paragraph_retriever(
            chunks=chunks,
            papers=papers,
            embedder_type="minilm",
            method="neural",
            storage_backend=mock_store,
        )

        assert retriever.chunk_store is mock_store
