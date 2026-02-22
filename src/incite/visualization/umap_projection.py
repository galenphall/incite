"""UMAP projection of paper embeddings for visual library exploration."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UMAPPoint:
    """A single paper's position in 2D UMAP space."""

    paper_id: str
    x: float
    y: float
    title: str
    authors: list[str]
    year: Optional[int]
    journal: Optional[str]
    doi: Optional[str] = None


@dataclass(frozen=True)
class UMAPProjection:
    """Complete UMAP projection with metadata."""

    points: list[UMAPPoint]
    num_papers: int
    embedder_type: str
    computed_at: str  # ISO format

    def to_dict(self) -> dict:
        return {
            "points": [
                {
                    "paper_id": p.paper_id,
                    "x": p.x,
                    "y": p.y,
                    "title": p.title,
                    "authors": p.authors,
                    "year": p.year,
                    "journal": p.journal,
                    "doi": p.doi,
                }
                for p in self.points
            ],
            "num_papers": self.num_papers,
            "embedder_type": self.embedder_type,
            "computed_at": self.computed_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> UMAPProjection:
        points = [
            UMAPPoint(
                paper_id=p["paper_id"],
                x=p["x"],
                y=p["y"],
                title=p["title"],
                authors=p["authors"],
                year=p["year"],
                journal=p.get("journal"),
                doi=p.get("doi"),
            )
            for p in data["points"]
        ]
        return cls(
            points=points,
            num_papers=data["num_papers"],
            embedder_type=data["embedder_type"],
            computed_at=data["computed_at"],
        )


def extract_embeddings_from_faiss(store) -> tuple[list[str], np.ndarray]:
    """Extract all embeddings from a FAISSStore.

    Returns (paper_ids, embeddings) where embeddings is shape (n, dim).
    """
    if store._index is None or store.size == 0:
        dim = store.dimension or 0
        return [], np.zeros((0, dim), dtype=np.float32)

    n = store._index.ntotal
    dim = store._index.d
    embeddings = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        store._index.reconstruct(i, embeddings[i])
    paper_ids = [store._idx_to_id[i] for i in range(n)]
    return paper_ids, embeddings


def compute_umap_projection(
    paper_ids: list[str],
    embeddings: np.ndarray,
    paper_metadata: dict[str, dict],
    embedder_type: str,
    n_neighbors: int = 25,
    min_dist: float = 0.03,
    metric: str = "cosine",
    random_state: int = 42,
) -> UMAPProjection:
    """Compute 2D UMAP projection from paper embeddings.

    Args:
        paper_ids: List of paper IDs matching embedding rows.
        embeddings: (n, dim) array of paper embeddings.
        paper_metadata: Dict mapping paper_id to {"title", "authors", "year", "journal"}.
        embedder_type: Name of the embedder used.
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        metric: UMAP distance metric.
        random_state: Random seed for reproducibility.
    """
    import umap

    n_papers = len(paper_ids)
    # Adjust n_neighbors if we have very few papers
    effective_neighbors = min(n_neighbors, max(2, n_papers - 1))

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    coords = reducer.fit_transform(embeddings)

    points = []
    for i, pid in enumerate(paper_ids):
        meta = paper_metadata.get(pid, {})
        points.append(
            UMAPPoint(
                paper_id=pid,
                x=float(coords[i, 0]),
                y=float(coords[i, 1]),
                title=meta.get("title", "Unknown"),
                authors=meta.get("authors", []),
                year=meta.get("year"),
                journal=meta.get("journal"),
                doi=meta.get("doi"),
            )
        )

    return UMAPProjection(
        points=points,
        num_papers=n_papers,
        embedder_type=embedder_type,
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def compute_umap_projection_with_model(
    paper_ids: list[str],
    embeddings: np.ndarray,
    paper_metadata: dict[str, dict],
    embedder_type: str,
    n_neighbors: int = 25,
    min_dist: float = 0.03,
    metric: str = "cosine",
    random_state: int = 42,
) -> tuple["UMAPProjection", object]:
    """Compute UMAP projection and return both the projection and fitted reducer.

    Returns:
        Tuple of (UMAPProjection, fitted umap.UMAP reducer).
    """
    import umap

    n_papers = len(paper_ids)
    effective_neighbors = min(n_neighbors, max(2, n_papers - 1))

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    coords = reducer.fit_transform(embeddings)

    points = []
    for i, pid in enumerate(paper_ids):
        meta = paper_metadata.get(pid, {})
        points.append(
            UMAPPoint(
                paper_id=pid,
                x=float(coords[i, 0]),
                y=float(coords[i, 1]),
                title=meta.get("title", "Unknown"),
                authors=meta.get("authors", []),
                year=meta.get("year"),
                journal=meta.get("journal"),
                doi=meta.get("doi"),
            )
        )

    projection = UMAPProjection(
        points=points,
        num_papers=n_papers,
        embedder_type=embedder_type,
        computed_at=datetime.now(timezone.utc).isoformat(),
    )
    return projection, reducer


def transform_new_papers(
    reducer: object,
    new_paper_ids: list[str],
    new_embeddings: np.ndarray,
    paper_metadata: dict[str, dict],
) -> list[UMAPPoint]:
    """Project new papers into existing UMAP space via transform().

    Args:
        reducer: A fitted umap.UMAP reducer.
        new_paper_ids: Paper IDs for the new embeddings.
        new_embeddings: (n, dim) array of new paper embeddings.
        paper_metadata: Dict mapping paper_id to metadata.

    Returns:
        List of UMAPPoint for the new papers.
    """
    coords = reducer.transform(new_embeddings)

    points = []
    for i, pid in enumerate(new_paper_ids):
        meta = paper_metadata.get(pid, {})
        points.append(
            UMAPPoint(
                paper_id=pid,
                x=float(coords[i, 0]),
                y=float(coords[i, 1]),
                title=meta.get("title", "Unknown"),
                authors=meta.get("authors", []),
                year=meta.get("year"),
                journal=meta.get("journal"),
                doi=meta.get("doi"),
            )
        )
    return points


def save_projection_cache(projection: UMAPProjection, cache_path: Path) -> None:
    """Save UMAP projection to a JSON cache file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(projection.to_dict(), f)
    logger.info("Saved UMAP projection cache to %s", cache_path)


def load_projection_cache(cache_path: Path) -> Optional[UMAPProjection]:
    """Load UMAP projection from cache, or None if not found."""
    if not cache_path.exists():
        return None
    try:
        with open(cache_path) as f:
            data = json.load(f)
        return UMAPProjection.from_dict(data)
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to load UMAP cache from %s: %s", cache_path, e)
        return None
