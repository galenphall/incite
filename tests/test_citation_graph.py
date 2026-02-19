"""Unit tests for CitationGraphStore."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from incite.discovery.citation_graph import CitationGraphStore

# ------------------------------------------------------------------
# Fixture helpers
# ------------------------------------------------------------------


def _write_adjacency_bin(path: Path, paper_count: int, edges: dict[int, list[int]]) -> int:
    """Write an adjacency file (offset table + packed uint32 edges).

    Returns total edge count written.
    """
    # Build packed neighbor lists and compute offsets
    offset_table_size = (paper_count + 1) * 8  # uint64 per entry
    current_byte = offset_table_size
    offsets: list[int] = []
    all_edges: list[bytes] = []

    total = 0
    for i in range(paper_count):
        offsets.append(current_byte)
        neighbors = edges.get(i, [])
        total += len(neighbors)
        packed = struct.pack(f"<{len(neighbors)}I", *neighbors) if neighbors else b""
        all_edges.append(packed)
        current_byte += len(packed)
    offsets.append(current_byte)  # sentinel

    with open(path, "wb") as f:
        for off in offsets:
            f.write(struct.pack("<Q", off))
        for packed in all_edges:
            f.write(packed)

    return total


def build_test_graph(
    path: Path,
    papers: list[str],
    forward_edges: dict[int, list[int]],
    backward_edges: dict[int, list[int]],
) -> None:
    """Build a minimal citation graph on disk for testing."""
    path.mkdir(parents=True, exist_ok=True)
    paper_count = len(papers)

    # id_map.bin — sorted 20-byte keys
    with open(path / "id_map.bin", "wb") as f:
        for hex_id in papers:
            f.write(bytes.fromhex(hex_id))

    fwd_total = _write_adjacency_bin(path / "forward.bin", paper_count, forward_edges)
    bwd_total = _write_adjacency_bin(path / "backward.bin", paper_count, backward_edges)

    meta = {
        "release_id": "test",
        "paper_count": paper_count,
        "edge_count": fwd_total,
        "build_date": "2026-01-01",
    }
    with open(path / "metadata.json", "w") as f:
        json.dump(meta, f)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

# 10 deterministic sorted S2 IDs (40-char hex)
PAPER_IDS = sorted(f"{i:040x}" for i in range(10))

# Forward edges: paper 0 cites papers 1, 2, 3; paper 5 cites paper 9
FORWARD = {0: [1, 2, 3], 5: [9]}

# Backward edges: paper 1 cited by 0; paper 2 cited by 0; paper 3 cited by 0; paper 9 cited by 5
BACKWARD = {1: [0], 2: [0], 3: [0], 9: [5]}


@pytest.fixture()
def graph_path(tmp_path: Path) -> Path:
    p = tmp_path / "graph"
    build_test_graph(p, PAPER_IDS, FORWARD, BACKWARD)
    return p


@pytest.fixture()
def store(graph_path: Path) -> CitationGraphStore:
    s = CitationGraphStore.open(graph_path)
    assert s is not None
    yield s
    s.close()


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestOpen:
    def test_open_valid(self, graph_path: Path) -> None:
        store = CitationGraphStore.open(graph_path)
        assert store is not None
        assert store.paper_count == 10
        store.close()

    def test_open_missing_dir(self, tmp_path: Path) -> None:
        assert CitationGraphStore.open(tmp_path / "nonexistent") is None

    def test_open_missing_file(self, graph_path: Path) -> None:
        (graph_path / "forward.bin").unlink()
        assert CitationGraphStore.open(graph_path) is None


class TestContextManager:
    def test_context_manager(self, graph_path: Path) -> None:
        with CitationGraphStore.open(graph_path) as store:
            assert store is not None
            assert store.paper_count == 10


class TestLookup:
    def test_lookup_known_id(self, store: CitationGraphStore) -> None:
        for i, hex_id in enumerate(PAPER_IDS):
            assert store._lookup_id(hex_id) == i

    def test_lookup_unknown_id(self, store: CitationGraphStore) -> None:
        unknown = "ff" * 20
        assert store._lookup_id(unknown) is None

    def test_reverse_lookup(self, store: CitationGraphStore) -> None:
        for i, hex_id in enumerate(PAPER_IDS):
            assert store._lookup_s2_id(i) == hex_id


class TestReferences:
    def test_get_references(self, store: CitationGraphStore) -> None:
        refs = store.get_references(PAPER_IDS[0])
        assert sorted(refs) == sorted([PAPER_IDS[1], PAPER_IDS[2], PAPER_IDS[3]])

    def test_get_references_empty(self, store: CitationGraphStore) -> None:
        assert store.get_references(PAPER_IDS[4]) == []

    def test_get_references_unknown(self, store: CitationGraphStore) -> None:
        assert store.get_references("ff" * 20) == []

    def test_get_references_int(self, store: CitationGraphStore) -> None:
        assert sorted(store.get_references_int(0)) == [1, 2, 3]

    def test_get_references_int_single(self, store: CitationGraphStore) -> None:
        assert store.get_references_int(5) == [9]


class TestCitations:
    def test_get_citations(self, store: CitationGraphStore) -> None:
        cits = store.get_citations(PAPER_IDS[1])
        assert cits == [PAPER_IDS[0]]

    def test_get_citations_empty(self, store: CitationGraphStore) -> None:
        assert store.get_citations(PAPER_IDS[0]) == []

    def test_get_citations_int(self, store: CitationGraphStore) -> None:
        assert store.get_citations_int(9) == [5]

    def test_get_citations_int_empty(self, store: CitationGraphStore) -> None:
        assert store.get_citations_int(0) == []
