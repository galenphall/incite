"""Neural embedding-based retrieval."""

import time
from typing import Union

import numpy as np

from incite.embeddings.base import BaseEmbedder
from incite.interfaces import Retriever, VectorStore
from incite.models import Paper, RetrievalResult


def _build_paper_store(
    papers: list[Paper],
    embedder: BaseEmbedder,
    show_progress: bool = True,
    include_metadata: bool = True,
    embedder_type: str | None = None,
    embeddings: np.ndarray | None = None,
) -> tuple[list[str], "VectorStore"]:
    """Build a FAISS store from papers.

    Shared by NeuralRetriever.from_papers and factory.build_index.

    Args:
        papers: List of Paper objects
        embedder: Embedding model
        show_progress: Whether to show progress bar
        include_metadata: Include author/year/journal in embedding text
        embedder_type: Optional embedder type label for store metadata
        embeddings: Optional pre-computed embeddings (skips embedder.embed)

    Returns:
        (paper_ids, store) tuple
    """
    from incite.embeddings.stores import FAISSStore

    texts = [p.to_embedding_text(include_metadata=include_metadata) for p in papers]
    ids = [p.id for p in papers]

    if embeddings is None:
        embeddings = embedder.embed(texts, show_progress=show_progress)

    dim = embeddings.shape[1]
    kwargs = {"dimension": dim}
    if embedder_type:
        kwargs["embedder_type"] = embedder_type
    store = FAISSStore(**kwargs)
    store.add(ids, embeddings)

    return ids, store


class NeuralRetriever(Retriever):
    """Retriever using neural embeddings and vector search."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: VectorStore,
        papers: dict[str, Paper],
    ):
        """Initialize neural retriever.

        Args:
            embedder: Embedding model
            store: Any VectorStore implementation (FAISSStore, etc.)
            papers: Dict mapping paper_id to Paper objects
        """
        self.embedder = embedder
        self.store = store
        self.papers = papers

    def retrieve(
        self,
        query: str,
        k: int = 10,
        return_timing: bool = False,
        **kwargs,
    ) -> Union[list[RetrievalResult], tuple[list[RetrievalResult], dict]]:
        """Retrieve top-k papers for a query.

        Args:
            query: Query text
            k: Number of results to return
            return_timing: If True, return (results, timing_dict)
            **kwargs: Ignored (for API compatibility with HybridRetriever)

        Returns:
            List of RetrievalResult objects sorted by score.
            If return_timing=True, returns (results, timing_dict).
        """
        timing = {}

        # Use pre-computed embedding if provided, otherwise embed the query
        embed_start = time.perf_counter()
        query_embedding = kwargs.get("query_embedding")
        if query_embedding is None:
            query_embedding = self.embedder.embed_query(query)
        timing["embed_query_ms"] = (time.perf_counter() - embed_start) * 1000

        # Vector search
        search_start = time.perf_counter()
        results = self.store.search(query_embedding, k)
        timing["vector_search_ms"] = (time.perf_counter() - search_start) * 1000

        retrieval_results = []
        for rank, (paper_id, score) in enumerate(results):
            retrieval_results.append(
                RetrievalResult(
                    paper_id=paper_id,
                    score=score,
                    rank=rank + 1,
                    score_breakdown={"neural": score},
                )
            )

        if return_timing:
            return retrieval_results, timing
        return retrieval_results

    @classmethod
    def from_papers(
        cls,
        papers: list[Paper],
        embedder: BaseEmbedder,
        show_progress: bool = True,
        include_metadata: bool = True,
    ) -> "NeuralRetriever":
        """Build retriever from a list of papers.

        Args:
            papers: List of Paper objects
            embedder: Embedding model
            show_progress: Whether to show progress bar
            include_metadata: Include author/year/journal in embedding text

        Returns:
            Initialized NeuralRetriever
        """
        paper_dict = {p.id: p for p in papers}
        _, store = _build_paper_store(
            papers, embedder, show_progress=show_progress, include_metadata=include_metadata
        )
        return cls(embedder=embedder, store=store, papers=paper_dict)
