"""ChunkStore for paragraph-level embeddings.

Extends FAISSStore with chunk-to-paper mapping for aggregation
from chunk-level to paper-level results.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from incite.embeddings.stores import FAISSStore
from incite.models import Chunk


class ChunkStore(FAISSStore):
    """FAISS-based vector store for chunks with paper mapping.

    Extends FAISSStore to track which paper each chunk belongs to,
    enabling aggregation from chunk-level to paper-level results.

    Maintains a reverse index (_paper_to_chunks) mapping paper_id to
    FAISS internal indices for efficient scoped search via
    search_within_papers().
    """

    def __init__(self, dimension: Optional[int] = None, embedder_type: Optional[str] = None):
        super().__init__(dimension, embedder_type=embedder_type)
        self._chunk_to_paper: dict[str, str] = {}
        self._paper_to_chunks: dict[str, list[int]] = {}

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add chunk embeddings to the store.

        Args:
            chunks: List of Chunk objects
            embeddings: numpy array of shape (n, dim), should be L2 normalized
        """
        ids = [c.id for c in chunks]
        self.add(ids, embeddings)

        # Track chunk -> paper mapping and build reverse index
        for chunk in chunks:
            self._chunk_to_paper[chunk.id] = chunk.paper_id
            idx = self._id_to_idx.get(chunk.id)
            if idx is not None:
                self._paper_to_chunks.setdefault(chunk.paper_id, []).append(idx)

    def get_paper_id(self, chunk_id: str) -> Optional[str]:
        """Get the paper ID for a chunk.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Paper ID or None if not found
        """
        return self._chunk_to_paper.get(chunk_id)

    def search_with_papers(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> list[tuple[str, str, float]]:
        """Search for k nearest neighbors with paper IDs.

        Args:
            query_embedding: Query vector of shape (dim,)
            k: Number of results to return

        Returns:
            List of (chunk_id, paper_id, score) tuples, sorted by score descending
        """
        results = self.search(query_embedding, k)

        results_with_papers = []
        for chunk_id, score in results:
            paper_id = self._chunk_to_paper.get(chunk_id, "")
            results_with_papers.append((chunk_id, paper_id, score))

        return results_with_papers

    def save(self, path: str | Path) -> None:
        """Save index, ID mappings, and chunk-to-paper mapping to disk."""
        super().save(path)

        path = Path(path)
        with open(path / "chunk_to_paper.json", "w") as f:
            json.dump(self._chunk_to_paper, f)

    def load(self, path: str | Path) -> None:
        """Load index, ID mappings, and chunk-to-paper mapping from disk."""
        super().load(path)

        path = Path(path)
        chunk_to_paper_path = path / "chunk_to_paper.json"
        if chunk_to_paper_path.exists():
            with open(chunk_to_paper_path) as f:
                self._chunk_to_paper = json.load(f)

        # Rebuild reverse index from chunk_to_paper + id mappings
        self._paper_to_chunks = {}
        for chunk_id, paper_id in self._chunk_to_paper.items():
            idx = self._id_to_idx.get(chunk_id)
            if idx is not None:
                self._paper_to_chunks.setdefault(paper_id, []).append(idx)

    @property
    def paper_ids(self) -> set[str]:
        """Return set of all paper IDs in the store."""
        return set(self._chunk_to_paper.values())

    def chunks_for_paper(self, paper_id: str) -> list[str]:
        """Get all chunk IDs for a paper.

        Uses the reverse index for O(1) lookup instead of scanning all chunks.

        Args:
            paper_id: Paper identifier

        Returns:
            List of chunk IDs belonging to this paper
        """
        indices = self._paper_to_chunks.get(paper_id, [])
        return [self._idx_to_id[idx] for idx in indices if idx in self._idx_to_id]

    def search_within_papers(
        self,
        query_embedding: np.ndarray,
        paper_ids: list[str],
        top_per_paper: int = 3,
    ) -> dict[str, list[tuple[str, float]]]:
        """Search chunks belonging to specific papers only.

        Uses FAISS reconstruct() to fetch only the vectors for chunks in the
        specified papers, then computes dot products with the query. Much faster
        than a global search when only a small subset of papers is needed.

        Args:
            query_embedding: Query vector of shape (dim,) or (1, dim)
            paper_ids: List of paper IDs to scope the search to
            top_per_paper: Maximum chunks to return per paper

        Returns:
            Dict mapping paper_id to list of (chunk_id, score) tuples,
            sorted by score descending per paper. Papers with no chunks
            are omitted from the result.
        """
        if self._index is None or self._index.ntotal == 0:
            return {}

        # Ensure query is 1-D float32
        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim == 2:
            q = q[0]

        results: dict[str, list[tuple[str, float]]] = {}

        for paper_id in paper_ids:
            indices = self._paper_to_chunks.get(paper_id)
            if not indices:
                continue

            # Reconstruct vectors for this paper's chunks
            n = len(indices)
            vecs = np.zeros((n, self.dimension), dtype=np.float32)
            for i, idx in enumerate(indices):
                self._index.reconstruct(int(idx), vecs[i])

            # Dot product (works for both cosine-on-normalized and raw IP)
            scores = vecs @ q

            # Build (chunk_id, score) pairs, sort, truncate
            chunk_scores = []
            for i, idx in enumerate(indices):
                chunk_id = self._idx_to_id.get(idx)
                if chunk_id is not None:
                    chunk_scores.append((chunk_id, float(scores[i])))

            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            results[paper_id] = chunk_scores[:top_per_paper]

        return results


def build_chunk_index(
    chunks: list[Chunk],
    embedder,
    output_path: Optional[Path] = None,
    show_progress: bool = True,
    progress_callback=None,
    embeddings: Optional[np.ndarray] = None,
) -> ChunkStore:
    """Build a ChunkStore from chunks.

    Args:
        chunks: List of Chunk objects
        embedder: Embedder instance (should support long context, e.g., Nomic)
        output_path: Optional path to save the index
        show_progress: Whether to show progress bar (tqdm to stdout)
        progress_callback: Optional callback(current, total, message) for UI progress
        embeddings: Optional pre-computed embeddings (skips local embedding)

    Returns:
        Populated ChunkStore
    """
    if embeddings is None:
        # Generate embedding texts (with contextual enrichment if available)
        texts = [c.to_embedding_text() for c in chunks]

        # Embed all texts
        if show_progress:
            print(f"Embedding {len(chunks)} chunks...")

        if progress_callback:
            # Batch manually so we can report progress to the UI
            batch_size = getattr(embedder, "batch_size", 32)
            all_embeddings = []
            total = len(texts)
            progress_callback(0, total, f"Loading embedding model ({total} chunks to embed)...")
            for i in range(0, total, batch_size):
                batch = texts[i : i + batch_size]
                batch_emb = embedder.embed(batch, show_progress=False)
                all_embeddings.append(batch_emb)
                done = min(i + batch_size, total)
                progress_callback(done, total, f"Embedding chunks: {done}/{total}")
            embeddings = np.vstack(all_embeddings)
        else:
            embeddings = embedder.embed(texts, show_progress=show_progress)

    # Create store and add chunks
    dim = embeddings.shape[1]
    embedder_type_str = getattr(embedder, "embedder_type", None)
    store = ChunkStore(dimension=dim, embedder_type=embedder_type_str)
    store.add_chunks(chunks, embeddings)

    # Optionally save
    if output_path:
        output_path = Path(output_path)
        store.save(str(output_path))
        if show_progress:
            print(f"Saved chunk index to {output_path}")

    return store
