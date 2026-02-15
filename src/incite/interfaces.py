"""Abstract interfaces for inCite components."""

from abc import ABC, abstractmethod
from typing import Protocol, Union

import numpy as np

from incite.models import CitationContext, Paper, RetrievalResult


class VectorStore(Protocol):
    """Protocol for vector storage and search."""

    def add(self, ids: list[str], embeddings: np.ndarray) -> None:
        """Add embeddings to the store."""
        ...

    def search(self, query_embedding: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Search for k nearest neighbors. Returns list of (id, score) tuples."""
        ...

    def save(self, path: str) -> None:
        """Save index to disk."""
        ...

    def load(self, path: str) -> None:
        """Load index from disk."""
        ...


class CorpusSource(Protocol):
    """Protocol for corpus sources (Zotero, JSONL file, Mendeley, etc.).

    Any class with these attributes and methods satisfies the protocol
    without needing to inherit from it (structural typing).

    Modeled after the DataSource protocol in finetuning/data_sources.py.
    """

    name: str

    def load_papers(self) -> list[Paper]:
        """Load papers from this source.

        Returns:
            List of Paper objects
        """
        ...

    def needs_refresh(self) -> bool:
        """Check if the corpus should be reloaded.

        Returns True if the underlying data has changed since last load
        (e.g., Zotero DB modified, file updated on disk).
        """
        ...

    def cache_key(self) -> str:
        """Return a string key for caching indexes built from this source.

        Different sources (or different paths for the same source type)
        should return distinct keys so their FAISS indexes don't collide.
        """
        ...


class Retriever(ABC):
    """Abstract base class for retrievers."""

    @abstractmethod
    def retrieve(
        self, query: str, k: int = 10, **kwargs
    ) -> Union[list[RetrievalResult], tuple[list[RetrievalResult], dict]]:
        """Retrieve top-k papers for a query."""
        pass

    def retrieve_for_context(
        self,
        context: CitationContext,
        k: int = 10,
        scale: str = "local",
        clean: bool = True,
        prefix_section: bool = False,
    ) -> list[RetrievalResult]:
        """Retrieve papers for a citation context at given scale.

        Args:
            context: Citation context to query for
            k: Number of results to return
            scale: Context scale ("local", "narrow", "broad", "section", "global")
            clean: If True, remove citation markers from query text
            prefix_section: If True, prepend section heading to query

        Returns:
            List of retrieval results
        """
        query = context.get_query(scale, clean=clean, prefix_section=prefix_section)
        return self.retrieve(query, k)


class Reranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        papers: dict[str, Paper],
        k: int = 10,
    ) -> list[RetrievalResult]:
        """Rerank candidates and return top-k."""
        pass
