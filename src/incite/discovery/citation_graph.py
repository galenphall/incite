"""Mmap-backed binary adjacency list store for the S2 citation graph.

File layout on disk::

    {path}/
    ├── metadata.json   # {release_id, paper_count, edge_count, build_date}
    ├── id_map.bin      # Sorted 20-byte binary S2 ID keys
    ├── forward.bin     # Paper → papers it cites (offset table + packed edges)
    └── backward.bin    # Paper → papers citing it (offset table + packed edges)

ID mapping
----------
S2 paper IDs are 40-char hex strings (SHA1).  They are stored as raw 20-byte
binary values in a sorted array (``id_map.bin``).  The integer ID of a paper
is its position in this sorted array, enabling O(log n) binary search on the
mmap'd buffer.

Adjacency format
----------------
Each ``.bin`` adjacency file starts with ``(paper_count + 1)`` little-endian
``uint64`` offset entries, followed by packed little-endian ``uint32`` edge IDs.
``offsets[i]`` is the byte position (relative to start of file) where paper
*i*'s neighbor list begins.  Neighbor count =
``(offsets[i+1] - offsets[i]) // 4``.
"""

from __future__ import annotations

import json
import logging
import mmap
import struct
from pathlib import Path
from types import TracebackType

logger = logging.getLogger(__name__)

_KEY_SIZE = 20  # bytes per S2 ID (SHA1)
_OFFSET_SIZE = 8  # uint64
_EDGE_SIZE = 4  # uint32


class CitationGraphStore:
    """Read-only, mmap-backed citation graph.

    Thread-safe: all reads go through mmap which is safe for concurrent access.
    """

    __slots__ = (
        "_path",
        "_paper_count",
        "_edge_count",
        "_mm_idmap",
        "_mm_forward",
        "_mm_backward",
    )

    def __init__(
        self,
        path: Path,
        paper_count: int,
        edge_count: int,
        mm_idmap: mmap.mmap,
        mm_forward: mmap.mmap,
        mm_backward: mmap.mmap,
    ) -> None:
        self._path = path
        self._paper_count = paper_count
        self._edge_count = edge_count
        self._mm_idmap = mm_idmap
        self._mm_forward = mm_forward
        self._mm_backward = mm_backward

    # ------------------------------------------------------------------
    # Construction / lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, path: Path) -> CitationGraphStore | None:
        """Open graph for reading.  Returns *None* if files are missing."""
        path = Path(path)
        meta_path = path / "metadata.json"
        idmap_path = path / "id_map.bin"
        fwd_path = path / "forward.bin"
        bwd_path = path / "backward.bin"

        for p in (meta_path, idmap_path, fwd_path, bwd_path):
            if not p.exists():
                logger.warning("Citation graph file missing: %s", p)
                return None

        try:
            with meta_path.open() as f:
                meta = json.load(f)
            paper_count: int = meta["paper_count"]
            edge_count: int = meta["edge_count"]

            mm_idmap = _mmap_file(idmap_path)
            mm_forward = _mmap_file(fwd_path)
            mm_backward = _mmap_file(bwd_path)
        except Exception:
            logger.warning("Failed to open citation graph at %s", path, exc_info=True)
            return None

        return cls(
            path=path,
            paper_count=paper_count,
            edge_count=edge_count,
            mm_idmap=mm_idmap,
            mm_forward=mm_forward,
            mm_backward=mm_backward,
        )

    def close(self) -> None:
        """Release mmap resources."""
        for mm in (self._mm_idmap, self._mm_forward, self._mm_backward):
            try:
                mm.close()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self) -> CitationGraphStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def paper_count(self) -> int:
        return self._paper_count

    @property
    def edge_count(self) -> int:
        return self._edge_count

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # ID mapping
    # ------------------------------------------------------------------

    def _lookup_id(self, s2_id: str) -> int | None:
        """Binary search id_map.bin for an S2 paper ID → integer index."""
        key = bytes.fromhex(s2_id)
        if len(key) != _KEY_SIZE:
            return None

        buf = self._mm_idmap
        lo, hi = 0, self._paper_count - 1
        while lo <= hi:
            mid = (lo + hi) >> 1
            offset = mid * _KEY_SIZE
            mid_key = buf[offset : offset + _KEY_SIZE]
            if mid_key < key:
                lo = mid + 1
            elif mid_key > key:
                hi = mid - 1
            else:
                return mid
        return None

    def _lookup_s2_id(self, int_id: int) -> str:
        """Reverse lookup: integer index → S2 hex string."""
        offset = int_id * _KEY_SIZE
        raw = self._mm_idmap[offset : offset + _KEY_SIZE]
        return raw.hex()

    # ------------------------------------------------------------------
    # Adjacency queries (string IDs)
    # ------------------------------------------------------------------

    def get_references(self, s2_id: str) -> list[str]:
        """S2 IDs of papers cited *by* this paper (forward edges)."""
        int_id = self._lookup_id(s2_id)
        if int_id is None:
            return []
        return [self._lookup_s2_id(i) for i in self._read_neighbors(self._mm_forward, int_id)]

    def get_citations(self, s2_id: str) -> list[str]:
        """S2 IDs of papers *citing* this paper (backward edges)."""
        int_id = self._lookup_id(s2_id)
        if int_id is None:
            return []
        return [self._lookup_s2_id(i) for i in self._read_neighbors(self._mm_backward, int_id)]

    # ------------------------------------------------------------------
    # Adjacency queries (integer IDs — avoids string conversion)
    # ------------------------------------------------------------------

    def get_references_int(self, int_id: int) -> list[int]:
        """Integer neighbor IDs for forward edges."""
        return self._read_neighbors(self._mm_forward, int_id)

    def get_citations_int(self, int_id: int) -> list[int]:
        """Integer neighbor IDs for backward edges."""
        return self._read_neighbors(self._mm_backward, int_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_neighbors(self, mm: mmap.mmap, int_id: int) -> list[int]:
        """Read packed uint32 neighbor list from an adjacency mmap."""
        # Offset table: (paper_count + 1) uint64 values at start of file
        start_off = struct.unpack_from("<Q", mm, int_id * _OFFSET_SIZE)[0]
        end_off = struct.unpack_from("<Q", mm, (int_id + 1) * _OFFSET_SIZE)[0]

        count = (end_off - start_off) // _EDGE_SIZE
        if count == 0:
            return []

        return list(struct.unpack_from(f"<{count}I", mm, start_off))


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _mmap_file(path: Path) -> mmap.mmap:
    """Memory-map a file read-only."""
    f = open(path, "rb")  # noqa: SIM115
    try:
        return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    finally:
        f.close()
