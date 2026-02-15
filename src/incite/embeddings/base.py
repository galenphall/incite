"""Base embedder implementation."""

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for embedders."""

    def __init__(self):
        self._query_cache: dict[str, np.ndarray] = {}

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns (n, dim) array."""
        pass

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Uses cache if available."""
        if query in self._query_cache:
            return self._query_cache[query]
        result = self.embed([query])
        return result[0]

    def embed_queries(self, queries: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed a batch of queries. Override for prefix-aware models."""
        return self.embed(queries, show_progress=show_progress)

    def precompute_queries(self, queries: list[str], show_progress: bool = False) -> None:
        """Pre-embed queries in batch for fast subsequent embed_query calls."""
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)

        embeddings = self.embed_queries(unique, show_progress=show_progress)
        self._query_cache = {q: embeddings[i] for i, q in enumerate(unique)}

    def embed_query_weighted(
        self,
        sentences: list[str],
        focus_index: int,
        decay: float = 0.5,
    ) -> np.ndarray:
        """Embed sentences with exponential decay weighting from a focus position.

        Sentences closer to focus_index get higher weight, decaying exponentially
        with distance. This lets the embedding emphasize the current sentence
        while still incorporating surrounding context.

        Args:
            sentences: List of sentence strings
            focus_index: Index of the focal sentence (highest weight)
            decay: Decay rate per sentence of distance (0-1). Lower = more focused.
                   0.5 means ±1 sentence gets half the weight of the focus.

        Returns:
            Weighted average embedding vector (dim,)
        """
        if len(sentences) <= 1:
            return self.embed_query(sentences[0] if sentences else "")

        # Clamp focus_index to valid range
        focus_index = max(0, min(focus_index, len(sentences) - 1))

        embeddings = self.embed_queries(sentences)
        distances = np.abs(np.arange(len(sentences)) - focus_index)
        weights = np.power(decay, distances.astype(float))
        weights /= weights.sum()
        return (embeddings * weights[:, np.newaxis]).sum(axis=0)

    def clear_query_cache(self) -> None:
        """Clear pre-computed query cache."""
        self._query_cache = {}
