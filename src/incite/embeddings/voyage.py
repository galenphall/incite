"""Voyage AI embedding model for scientific papers."""

import os
import time

import numpy as np
from tqdm import tqdm

from incite.embeddings.base import BaseEmbedder


class VoyageEmbedder(BaseEmbedder):
    """Voyage AI embedder using the voyage-4 model.

    API-based embedder with separate document/query input types.
    Embeddings come pre-normalized from the API.

    Rate limiting: Automatically retries on rate limit errors with
    exponential backoff. Default batch size of 30 keeps token usage
    under free-tier limits (~10K TPM).
    """

    DIMENSION = 1024  # voyage-4 default

    def __init__(
        self,
        model_name: str = "voyage-4",
        batch_size: int = 30,
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self._client = None

    def _load_client(self):
        if self._client is None:
            import voyageai

            api_key = os.getenv("VOYAGE_API_KEY")
            if not api_key:
                raise ValueError(
                    "VOYAGE_API_KEY not found. Set it in .env or as an environment variable."
                )
            self._client = voyageai.Client(api_key=api_key)

    def _embed_with_retry(self, texts: list[str], input_type: str, max_retries: int = 5):
        """Call the Voyage embed API with retry on rate limit errors."""
        import voyageai.error

        for attempt in range(max_retries):
            try:
                return self._client.embed(texts, model=self.model_name, input_type=input_type)
            except voyageai.error.RateLimitError:
                wait = 2**attempt * 5  # 5, 10, 20, 40, 80 seconds
                print(f"  Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
        # Final attempt without catching
        return self._client.embed(texts, model=self.model_name, input_type=input_type)

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed documents via Voyage API."""
        self._load_client()

        if len(texts) == 0:
            return np.array([]).reshape(0, self.dimension)

        all_embeddings = []
        total_tokens = 0
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding (Voyage API)")

        for i in iterator:
            batch = texts[i : i + self.batch_size]
            result = self._embed_with_retry(batch, input_type="document")
            all_embeddings.extend(result.embeddings)
            total_tokens += result.total_tokens

        if total_tokens > 0:
            print(f"  Voyage API: {total_tokens:,} tokens used for {len(texts)} texts")

        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query with input_type='query'."""
        if query in self._query_cache:
            return self._query_cache[query]
        self._load_client()

        result = self._embed_with_retry([query], input_type="query")
        return np.array(result.embeddings[0], dtype=np.float32)

    def embed_queries(self, queries: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed a batch of queries with input_type='query'."""
        self._load_client()

        if len(queries) == 0:
            return np.array([]).reshape(0, self.dimension)

        all_embeddings = []
        total_tokens = 0
        iterator = range(0, len(queries), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding queries (Voyage API)")

        for i in iterator:
            batch = queries[i : i + self.batch_size]
            result = self._embed_with_retry(batch, input_type="query")
            all_embeddings.extend(result.embeddings)
            total_tokens += result.total_tokens

        if total_tokens > 0:
            print(f"  Voyage API: {total_tokens:,} tokens used for {len(queries)} queries")

        return np.array(all_embeddings, dtype=np.float32)
