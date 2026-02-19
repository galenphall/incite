"""Cache for graph analysis metrics (PageRank, co-citation).

Stores computed metrics in a JSON file keyed by a hash of the library's
S2 IDs, so results are reused until the library composition changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".incite"


def _library_hash(s2_ids: set[str]) -> str:
    """Deterministic hash of sorted S2 IDs."""
    content = "\n".join(sorted(s2_ids))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class GraphMetricsCache:
    """JSON-backed cache for graph analysis results."""

    def __init__(self, s2_ids: set[str]):
        self._hash = _library_hash(s2_ids)
        self._path = CACHE_DIR / f"graph_metrics_{self._hash}.json"

    @property
    def path(self) -> Path:
        return self._path

    def is_stale(self, max_age_hours: float = 24.0) -> bool:
        """Check if cache is missing or too old."""
        if not self._path.exists():
            return True
        try:
            data = json.loads(self._path.read_text())
            age_hours = (time.time() - data.get("timestamp", 0)) / 3600
            return age_hours > max_age_hours
        except (json.JSONDecodeError, KeyError):
            return True

    def load(self) -> dict | None:
        """Load cached metrics. Returns None if cache is stale or missing."""
        if self.is_stale():
            return None
        try:
            data = json.loads(self._path.read_text())
            logger.info(
                "Loaded graph metrics cache (%d PageRank, %d co-citation scores)",
                len(data.get("pagerank", {})),
                len(data.get("cocitation", {})),
            )
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load graph metrics cache: %s", e)
            return None

    def save(
        self,
        pagerank: dict[str, float],
        cocitation: dict[str, float],
        seed_count: int,
        subgraph_size: int,
    ) -> None:
        """Persist metrics to disk."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "timestamp": time.time(),
            "library_hash": self._hash,
            "seed_count": seed_count,
            "subgraph_size": subgraph_size,
            "pagerank": pagerank,
            "cocitation": cocitation,
        }
        self._path.write_text(json.dumps(data, indent=2))
        logger.info("Saved graph metrics cache to %s", self._path)
