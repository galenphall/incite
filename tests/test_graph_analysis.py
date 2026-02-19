"""Tests for citation graph analysis (PageRank + co-citation)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from scipy import sparse

from incite.discovery.graph_analysis import (
    build_subgraph,
    compute_cocitation,
    compute_pagerank,
)
from incite.discovery.graph_cache import GraphMetricsCache
from incite.discovery.models import DiscoveryCandidate

# ---------------------------------------------------------------------------
# Fixtures: synthetic graph
# ---------------------------------------------------------------------------


def _make_mock_graph() -> MagicMock:
    """Create a mock CitationGraphStore with a small synthetic graph.

    Graph structure (A -> B means A cites B):
      A -> B, C
      B -> C, D
      C -> D
      E -> A, B  (E is a citer of both A and B)
      F -> A, C  (F cites A and C)
    """
    graph = MagicMock()

    # S2 IDs (hex strings)
    ids = {
        "A": "aa" * 20,
        "B": "bb" * 20,
        "C": "cc" * 20,
        "D": "dd" * 20,
        "E": "ee" * 20,
        "F": "ff" * 20,
    }
    int_ids = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}

    # _lookup_id: s2_id -> int_id
    def lookup_id(s2_id: str) -> int | None:
        for name, sid in ids.items():
            if sid == s2_id:
                return int_ids[name]
        return None

    graph._lookup_id = lookup_id

    # _lookup_s2_id: int_id -> s2_id
    int_to_name = {v: k for k, v in int_ids.items()}

    def lookup_s2_id(int_id: int) -> str:
        name = int_to_name[int_id]
        return ids[name]

    graph._lookup_s2_id = lookup_s2_id

    # References (forward edges): who does this paper cite?
    refs = {
        0: [1, 2],  # A -> B, C
        1: [2, 3],  # B -> C, D
        2: [3],  # C -> D
        3: [],  # D (no refs)
        4: [0, 1],  # E -> A, B
        5: [0, 2],  # F -> A, C
    }
    graph.get_references_int = lambda i: refs.get(i, [])
    graph.get_references = lambda s2: [ids[int_to_name[r]] for r in refs.get(lookup_id(s2), [])]

    # Citations (backward edges): who cites this paper?
    cites = {
        0: [4, 5],  # A is cited by E, F
        1: [0, 4],  # B is cited by A, E
        2: [0, 1, 5],  # C is cited by A, B, F
        3: [1, 2],  # D is cited by B, C
        4: [],
        5: [],
    }
    graph.get_citations_int = lambda i: cites.get(i, [])
    graph.get_citations = lambda s2: [ids[int_to_name[c]] for c in cites.get(lookup_id(s2), [])]

    graph._ids = ids
    return graph


# ---------------------------------------------------------------------------
# Tests: build_subgraph
# ---------------------------------------------------------------------------


class TestBuildSubgraph:
    def test_basic_subgraph(self):
        graph = _make_mock_graph()
        seed_ids = {graph._ids["A"], graph._ids["B"]}
        int_to_s2, s2_to_int, adj = build_subgraph(graph, seed_ids, depth=1)

        # Should include seeds + their 1-hop neighbors
        assert len(int_to_s2) >= 2  # at least the seeds
        assert adj.shape[0] == adj.shape[1] == len(int_to_s2)

    def test_empty_seeds(self):
        graph = _make_mock_graph()
        int_to_s2, s2_to_int, adj = build_subgraph(graph, set(), depth=1)
        assert len(int_to_s2) == 0
        assert adj.shape == (0, 0)

    def test_unknown_seed(self):
        graph = _make_mock_graph()
        int_to_s2, s2_to_int, adj = build_subgraph(graph, {"00" * 20}, depth=1)
        # Unknown ID should be skipped gracefully
        # The mock won't find it, so no seeds
        assert len(int_to_s2) == 0

    def test_depth_2_reaches_all(self):
        graph = _make_mock_graph()
        seed_ids = {graph._ids["A"]}
        int_to_s2, s2_to_int, adj = build_subgraph(graph, seed_ids, depth=2)
        # Depth 2 from A should reach all 6 nodes
        assert len(int_to_s2) == 6


# ---------------------------------------------------------------------------
# Tests: compute_pagerank
# ---------------------------------------------------------------------------


class TestComputePagerank:
    def test_basic_pagerank(self):
        graph = _make_mock_graph()
        seed_ids = {graph._ids["A"], graph._ids["B"]}
        int_to_s2, s2_to_int, adj = build_subgraph(graph, seed_ids, depth=2)
        seed_indices = {s2_to_int[sid] for sid in seed_ids if sid in s2_to_int}

        scores = compute_pagerank(adj, seed_indices, int_to_s2)

        # Should have scores for all nodes in subgraph
        assert len(scores) == len(int_to_s2)
        # All scores in [0, 1]
        for score in scores.values():
            assert 0.0 <= score <= 1.0
        # At least one score should be 1.0 (max normalized)
        assert max(scores.values()) == pytest.approx(1.0)

    def test_seed_bias(self):
        graph = _make_mock_graph()
        seed_ids = {graph._ids["A"]}
        int_to_s2, s2_to_int, adj = build_subgraph(graph, seed_ids, depth=2)
        seed_indices = {s2_to_int[sid] for sid in seed_ids if sid in s2_to_int}

        # With high seed bias, seed papers should have higher PageRank
        scores = compute_pagerank(adj, seed_indices, int_to_s2, seed_bias=0.9)
        seed_score = scores[graph._ids["A"]]
        # Seed should have a meaningful score
        assert seed_score > 0.1

    def test_empty_graph(self):
        adj = sparse.csr_matrix((0, 0))
        scores = compute_pagerank(adj, set(), {})
        assert scores == {}


# ---------------------------------------------------------------------------
# Tests: compute_cocitation
# ---------------------------------------------------------------------------


class TestComputeCocitation:
    def test_basic_cocitation(self):
        graph = _make_mock_graph()
        # Library has A and B
        seed_ids = {graph._ids["A"], graph._ids["B"]}
        scores = compute_cocitation(graph, seed_ids)

        # Should find papers co-cited with A and B
        assert len(scores) > 0
        # All scores in [0, 1]
        for score in scores.values():
            assert 0.0 <= score <= 1.0
        # Library papers should NOT appear in results
        for seed_id in seed_ids:
            assert seed_id not in scores

    def test_empty_seeds(self):
        graph = _make_mock_graph()
        scores = compute_cocitation(graph, set())
        assert scores == {}


# ---------------------------------------------------------------------------
# Tests: GraphMetricsCache
# ---------------------------------------------------------------------------


class TestGraphMetricsCache:
    def test_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr("incite.discovery.graph_cache.CACHE_DIR", tmp_path)

        s2_ids = {"abc123", "def456"}
        cache = GraphMetricsCache(s2_ids)

        # Initially stale
        assert cache.is_stale()
        assert cache.load() is None

        # Save
        cache.save(
            pagerank={"abc123": 0.8, "def456": 0.3},
            cocitation={"xyz789": 0.5},
            seed_count=2,
            subgraph_size=100,
        )

        # Load
        data = cache.load()
        assert data is not None
        assert data["pagerank"]["abc123"] == 0.8
        assert data["cocitation"]["xyz789"] == 0.5
        assert data["seed_count"] == 2

    def test_different_libraries_different_caches(self, tmp_path, monkeypatch):
        monkeypatch.setattr("incite.discovery.graph_cache.CACHE_DIR", tmp_path)

        cache1 = GraphMetricsCache({"a", "b"})
        cache2 = GraphMetricsCache({"a", "c"})
        assert cache1.path != cache2.path

    def test_same_library_same_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("incite.discovery.graph_cache.CACHE_DIR", tmp_path)

        cache1 = GraphMetricsCache({"a", "b"})
        cache2 = GraphMetricsCache({"b", "a"})  # same set, different order
        assert cache1.path == cache2.path


# ---------------------------------------------------------------------------
# Tests: DiscoveryCandidate model updates
# ---------------------------------------------------------------------------


class TestDiscoveryCandidateGraphFields:
    def test_signal_count_includes_graph(self):
        cand = DiscoveryCandidate(
            s2_id="test",
            title="Test",
            authors=[],
            year=2024,
            doi=None,
            abstract="",
            pagerank_score=0.5,
            cocitation_score=0.3,
        )
        # pagerank > 0.1 and cocitation > 0.1 should each count
        assert cand.signal_count == 2

    def test_discovery_score_includes_graph(self):
        cand = DiscoveryCandidate(
            s2_id="test",
            title="Test",
            authors=[],
            year=2024,
            doi=None,
            abstract="",
            pagerank_score=1.0,
            cocitation_score=1.0,
        )
        # Should include graph components
        score = cand.discovery_score
        assert score > 0.0
        # 0.15 * 1.0 (pagerank) + 0.10 * 1.0 (cocitation) = 0.25
        assert score == pytest.approx(0.25, abs=0.01)

    def test_to_dict_includes_graph_fields(self):
        cand = DiscoveryCandidate(
            s2_id="test",
            title="Test",
            authors=[],
            year=2024,
            doi=None,
            abstract="",
            pagerank_score=0.7,
            cocitation_score=0.4,
        )
        d = cand.to_dict()
        assert d["pagerank_score"] == 0.7
        assert d["cocitation_score"] == 0.4

    def test_from_dict_includes_graph_fields(self):
        d = {
            "s2_id": "test",
            "title": "Test",
            "authors": [],
            "year": 2024,
            "doi": None,
            "abstract": "",
            "pagerank_score": 0.5,
            "cocitation_score": 0.3,
        }
        cand = DiscoveryCandidate.from_dict(d)
        assert cand.pagerank_score == 0.5
        assert cand.cocitation_score == 0.3

    def test_from_dict_backward_compatible(self):
        """Old dicts without graph fields should still work."""
        d = {
            "s2_id": "test",
            "title": "Test",
        }
        cand = DiscoveryCandidate.from_dict(d)
        assert cand.pagerank_score == 0.0
        assert cand.cocitation_score == 0.0
