"""Vector stores for embedding search."""

import json
from pathlib import Path
from typing import Literal, Optional

import numpy as np


class FAISSStore:
    """FAISS-based vector store.

    Supports two index types:
    - "flat": Exact search (IndexFlatIP). Best for small corpora (<50K).
    - "hnsw": Approximate search (IndexHNSWFlat). ~0.1ms at 50K, >99% recall@10.
    - "sq8": Scalar-quantized flat index (IndexScalarQuantizer). 4x memory reduction
      with <1% recall loss. Best for large chunk indexes (>100K vectors).
    """

    def __init__(
        self,
        dimension: Optional[int] = None,
        index_type: Literal["flat", "hnsw", "sq8"] = "flat",
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 128,
        embedder_type: Optional[str] = None,
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.embedder_type = embedder_type
        self._index = None
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: dict[int, str] = {}

    def _ensure_index(self, dimension: int):
        """Create index if not exists."""
        if self._index is None:
            import faiss

            self.dimension = dimension

            if self.index_type == "hnsw":
                self._index = faiss.IndexHNSWFlat(
                    dimension, self.hnsw_m, faiss.METRIC_INNER_PRODUCT
                )
                self._index.hnsw.efConstruction = self.hnsw_ef_construction
                self._index.hnsw.efSearch = self.hnsw_ef_search
            elif self.index_type == "sq8":
                self._index = faiss.IndexScalarQuantizer(
                    dimension, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
                )
            else:
                self._index = faiss.IndexFlatIP(dimension)  # Inner product (cosine for normalized)

    def add(self, ids: list[str], embeddings: np.ndarray) -> None:
        """Add embeddings to the store.

        Args:
            ids: List of document IDs
            embeddings: numpy array of shape (n, dim), should be L2 normalized
        """
        if len(ids) == 0:
            return

        if len(ids) != len(embeddings):
            raise ValueError(f"Length mismatch: {len(ids)} ids vs {len(embeddings)} embeddings")

        self._ensure_index(embeddings.shape[1])

        # Ensure embeddings are float32 and contiguous
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

        # SQ8 index needs training before adding vectors
        if self.index_type == "sq8" and not self._index.is_trained:
            self._index.train(embeddings)

        # Map IDs to indices
        start_idx = len(self._id_to_idx)
        for i, doc_id in enumerate(ids):
            idx = start_idx + i
            self._id_to_idx[doc_id] = idx
            self._idx_to_id[idx] = doc_id

        self._index.add(embeddings)

    def search(self, query_embedding: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        """Search for k nearest neighbors.

        Args:
            query_embedding: Query vector of shape (dim,)
            k: Number of results to return

        Returns:
            List of (id, score) tuples, sorted by score descending
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        # Reshape query if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = np.ascontiguousarray(query_embedding.astype(np.float32))

        # Limit k to number of vectors
        k = min(k, self._index.ntotal)

        scores, indices = self._index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx in self._idx_to_id:
                doc_id = self._idx_to_id[idx]
                results.append((doc_id, float(score)))

        return results

    def search_two_stage(
        self,
        query_full: np.ndarray,
        k: int = 10,
        candidates: int = 100,
        truncate_dim: int = 128,
    ) -> list[tuple[str, float]]:
        """Two-stage Matryoshka search: fast truncated scan → precise full rerank.

        Stage 1: Search a truncated-dimension index for top candidates (~0.05ms).
        Stage 2: Rerank candidates using full-dimension cosine similarity (~0.02ms).

        Requires a separate truncated index (self._truncated_index) built via
        build_truncated_index(). Falls back to regular search if not available.

        Args:
            query_full: Full-dimension query vector (e.g., 384-dim)
            k: Number of final results
            candidates: Number of Stage 1 candidates to rerank
            truncate_dim: Dimension to truncate to for Stage 1

        Returns:
            List of (id, score) tuples, sorted by full-dimension score descending
        """
        if not hasattr(self, "_truncated_index") or self._truncated_index is None:
            return self.search(query_full, k)

        if self._truncated_index.ntotal == 0:
            return self.search(query_full, k)

        # Stage 1: Truncated search for candidates
        if query_full.ndim == 1:
            query_trunc = query_full[:truncate_dim]
        else:
            query_trunc = query_full[:, :truncate_dim]
        if query_trunc.ndim == 1:
            query_trunc = query_trunc.reshape(1, -1)

        # Normalize truncated query
        norm = np.linalg.norm(query_trunc, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1, norm)
        query_trunc = query_trunc / norm
        query_trunc = np.ascontiguousarray(query_trunc.astype(np.float32))

        candidates = min(candidates, self._truncated_index.ntotal)
        _, candidate_indices = self._truncated_index.search(query_trunc, candidates)

        # Stage 2: Rerank with full-dimension embeddings
        if query_full.ndim == 1:
            query_full = query_full.reshape(1, -1)
        query_full = np.ascontiguousarray(query_full.astype(np.float32))

        results = []
        for idx in candidate_indices[0]:
            if idx < 0 or idx not in self._idx_to_id:
                continue
            doc_id = self._idx_to_id[idx]
            # Reconstruct full vector from the main index
            full_vec = np.zeros((1, self.dimension), dtype=np.float32)
            self._index.reconstruct(int(idx), full_vec[0])
            score = float(np.dot(query_full[0], full_vec[0]))
            results.append((doc_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def build_truncated_index(
        self,
        truncate_dim: int = 128,
        index_type: Literal["flat", "hnsw"] = "hnsw",
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 128,
    ) -> None:
        """Build a truncated-dimension index for two-stage Matryoshka search.

        Reconstructs all vectors from the main index, truncates to truncate_dim,
        re-normalizes, and builds a secondary index for fast candidate retrieval.

        Args:
            truncate_dim: Target dimension (e.g., 128 for 3x speedup)
            index_type: Index type for truncated index ("flat" or "hnsw")
            hnsw_m: HNSW M parameter
            hnsw_ef_construction: HNSW construction parameter
            hnsw_ef_search: HNSW search parameter
        """
        import faiss

        if self._index is None or self._index.ntotal == 0:
            return

        n = self._index.ntotal
        full_vecs = np.zeros((n, self.dimension), dtype=np.float32)
        for i in range(n):
            self._index.reconstruct(i, full_vecs[i])

        truncated = full_vecs[:, :truncate_dim].copy()

        # Re-normalize truncated vectors
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        truncated = truncated / norms
        truncated = np.ascontiguousarray(truncated)

        if index_type == "hnsw":
            self._truncated_index = faiss.IndexHNSWFlat(
                truncate_dim, hnsw_m, faiss.METRIC_INNER_PRODUCT
            )
            self._truncated_index.hnsw.efConstruction = hnsw_ef_construction
            self._truncated_index.hnsw.efSearch = hnsw_ef_search
        else:
            self._truncated_index = faiss.IndexFlatIP(truncate_dim)

        self._truncated_index.add(truncated)

    def save(self, path: str | Path) -> None:
        """Save index and mappings to disk."""
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._index is not None:
            faiss.write_index(self._index, str(path / "index.faiss"))

        id_map_data = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "id_to_idx": self._id_to_idx,
        }
        if self.embedder_type:
            id_map_data["embedder_type"] = self.embedder_type

        with open(path / "id_map.json", "w") as f:
            json.dump(id_map_data, f)

        # Save truncated index if it exists
        if hasattr(self, "_truncated_index") and self._truncated_index is not None:
            faiss.write_index(self._truncated_index, str(path / "index_truncated.faiss"))

    def load(self, path: str | Path) -> None:
        """Load index and mappings from disk."""
        import faiss

        path = Path(path)

        if (path / "index.faiss").exists():
            self._index = faiss.read_index(str(path / "index.faiss"))

        if (path / "id_map.json").exists():
            with open(path / "id_map.json") as f:
                data = json.load(f)
                self.dimension = data.get("dimension")
                self.index_type = data.get("index_type", "flat")
                self.embedder_type = data.get("embedder_type")
                self._id_to_idx = data.get("id_to_idx", {})
                self._idx_to_id = {int(v): k for k, v in self._id_to_idx.items()}

        # Load truncated index if it exists
        truncated_path = path / "index_truncated.faiss"
        if truncated_path.exists():
            self._truncated_index = faiss.read_index(str(truncated_path))

    @property
    def size(self) -> int:
        """Return number of vectors in index."""
        if self._index is None:
            return 0
        return self._index.ntotal
