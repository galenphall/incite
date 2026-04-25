"""Additional scientific embedding models.

Contains E5, Granite, Nomic, SciNCL, and ModernBERT embedders.
These are alternative embedding backends available via the EMBEDDERS
registry in retrieval/factory_registry.py.

Related modules:
    - incite.embeddings.base: BaseEmbedder ABC.
    - incite.embeddings.specter: SPECTEREmbedder (primary scientific embedder).
    - incite.embeddings.minilm: MiniLMEmbedder (fast baseline).
    - incite.retrieval.factory_registry: Embedder registry.
"""

from typing import Optional

import numpy as np

from incite.embeddings.base import BaseEmbedder


def _get_best_device() -> str:
    """Return best available device: MPS (Apple Silicon) > CUDA > CPU."""
    from incite.utils import get_best_device

    return get_best_device()


class SciNCLEmbedder(BaseEmbedder):
    """SciNCL embedder for scientific papers.

    Trained on citation graph neighborhoods via contrastive learning.
    Uses nearest-neighbor sampling instead of hard citation links (unlike SPECTER).
    768-dim, sentence-transformers compatible, no prefixes needed.
    """

    def __init__(
        self,
        model_name: str = "malteos/scincl",
        device: Optional[str] = None,
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device if device is not None else _get_best_device()
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._model = None
        self._dimension: Optional[int] = None

    def _load_model(self):
        """Lazy-load the sentence-transformers model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_dir,
            )
            self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        """Return the embedding dimension, loading the model if needed."""
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: List of texts to embed.
            show_progress: Whether to show a progress bar.

        Returns:
            numpy array of shape (n, dim) with L2-normalized embeddings.
        """
        self._load_model()

        if len(texts) == 0:
            return np.array([]).reshape(0, self.dimension)

        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


class ModernBERTEmbedder(BaseEmbedder):
    """ModernBERT-embed-base from Nomic AI.

    Built on the ModernBERT architecture with rotary embeddings, 8K context,
    and Matryoshka support. Uses same prefixes as Nomic Embed v1.5:
    'search_document: ' for documents, 'search_query: ' for queries.
    """

    def __init__(
        self,
        model_name: str = "nomic-ai/modernbert-embed-base",
        device: Optional[str] = None,
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device if device is not None else _get_best_device()
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._model = None
        self._dimension: Optional[int] = None

    def _load_model(self):
        """Lazy-load the sentence-transformers model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_dir,
            )
            self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        """Return the embedding dimension, loading the model if needed."""
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed documents (prefixed with 'search_document: ').

        Args:
            texts: List of document texts to embed.
            show_progress: Whether to show a progress bar.

        Returns:
            numpy array of shape (n, dim) with L2-normalized embeddings.
        """
        self._load_model()

        if len(texts) == 0:
            return np.array([]).reshape(0, self.dimension)

        prefixed = [f"search_document: {t}" for t in texts]

        return self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query (prefixed with 'search_query: ').

        Args:
            query: Query text to embed.

        Returns:
            numpy array of shape (dim,) with L2-normalized embedding.
        """
        if query in self._query_cache:
            return self._query_cache[query]
        self._load_model()

        prefixed = f"search_query: {query}"

        return self._model.encode(
            prefixed,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def embed_queries(self, queries: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed a batch of queries (prefixed with 'search_query: ').

        Args:
            queries: List of query texts to embed.
            show_progress: Whether to show a progress bar.

        Returns:
            numpy array of shape (n, dim) with L2-normalized embeddings.
        """
        self._load_model()

        if len(queries) == 0:
            return np.array([]).reshape(0, self.dimension)

        prefixed = [f"search_query: {q}" for q in queries]

        return self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


class E5Embedder(BaseEmbedder):
    """E5 embedder from Microsoft.

    E5 models require specific prefixes: 'query: ' for queries and
    'passage: ' for documents. This class handles that automatically.
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-small-v2",
        device: Optional[str] = None,
        batch_size: int = 64,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device if device is not None else _get_best_device()
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._model = None
        self._dimension: Optional[int] = None

    def _load_model(self):
        """Lazy-load the sentence-transformers model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_dir,
            )
            self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        """Return the embedding dimension, loading the model if needed."""
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed documents (prefixed with 'passage: ').

        Args:
            texts: List of document texts to embed.
            show_progress: Whether to show a progress bar.

        Returns:
            numpy array of shape (n, dim) with L2-normalized embeddings.
        """
        self._load_model()

        if len(texts) == 0:
            return np.array([]).reshape(0, self.dimension)

        # E5 requires "passage: " prefix for documents
        prefixed = [f"passage: {t}" for t in texts]

        return self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query (prefixed with 'query: ').

        Args:
            query: Query text to embed.

        Returns:
            numpy array of shape (dim,) with L2-normalized embedding.
        """
        if query in self._query_cache:
            return self._query_cache[query]
        self._load_model()

        prefixed = f"query: {query}"

        return self._model.encode(
            prefixed,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def embed_queries(self, queries: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed a batch of queries (prefixed with 'query: ').

        Args:
            queries: List of query texts to embed.
            show_progress: Whether to show a progress bar.

        Returns:
            numpy array of shape (n, dim) with L2-normalized embeddings.
        """
        self._load_model()

        if len(queries) == 0:
            return np.array([]).reshape(0, self.dimension)

        prefixed = [f"query: {q}" for q in queries]

        return self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


class GraniteEmbedder(BaseEmbedder):
    """IBM Granite-small-R2 base embedder (unfinetuned).

    384-dim, ModernBERT architecture with 8K context.
    Uses 'query: ' / 'passage: ' asymmetric prefixes.
    Sequence length capped at 512 to match MiniLM and avoid OOM on Apple Silicon.
    """

    def __init__(
        self,
        model_name: str = "ibm-granite/granite-embedding-small-english-r2",
        device: Optional[str] = None,
        batch_size: int = 64,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device if device is not None else _get_best_device()
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._model = None
        self._dimension: Optional[int] = None

    def _load_model(self):
        """Lazy-load the sentence-transformers model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_dir,
            )
            # Cap sequence length: 8192 default causes OOM on Apple Silicon.
            # 512 covers 91%+ of training/eval examples without truncation.
            self._model.max_seq_length = 512
            self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        """Return the embedding dimension, loading the model if needed."""
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed documents (prefixed with 'passage: ').

        Args:
            texts: List of document texts to embed.
            show_progress: Whether to show a progress bar.

        Returns:
            numpy array of shape (n, dim) with L2-normalized embeddings.
        """
        self._load_model()

        if len(texts) == 0:
            return np.array([]).reshape(0, self.dimension)

        prefixed = [f"passage: {t}" for t in texts]

        return self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query (prefixed with 'query: ').

        Args:
            query: Query text to embed.

        Returns:
            numpy array of shape (dim,) with L2-normalized embedding.
        """
        if query in self._query_cache:
            return self._query_cache[query]
        self._load_model()

        prefixed = f"query: {query}"

        return self._model.encode(
            prefixed,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def embed_queries(self, queries: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed a batch of queries (prefixed with 'query: ').

        Args:
            queries: List of query texts to embed.
            show_progress: Whether to show a progress bar.

        Returns:
            numpy array of shape (n, dim) with L2-normalized embeddings.
        """
        self._load_model()

        if len(queries) == 0:
            return np.array([]).reshape(0, self.dimension)

        prefixed = [f"query: {q}" for q in queries]

        return self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


class NomicEmbedder(BaseEmbedder):
    """Nomic Embed v1.5 embedder.

    768-dim, 8192-token context window, instruction-aware prefixes.
    Uses 'search_document: ' for documents and 'search_query: ' for queries.
    Requires trust_remote_code=True for the custom pooling layer.
    """

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        device: Optional[str] = None,
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device if device is not None else _get_best_device()
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._model = None
        self._dimension: Optional[int] = None

    def _load_model(self):
        """Lazy-load the sentence-transformers model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_dir,
                trust_remote_code=True,
            )
            self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        """Return the embedding dimension, loading the model if needed."""
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed documents (prefixed with 'search_document: ').

        Args:
            texts: List of document texts to embed.
            show_progress: Whether to show a progress bar.

        Returns:
            numpy array of shape (n, dim) with L2-normalized embeddings.
        """
        self._load_model()

        if len(texts) == 0:
            return np.array([]).reshape(0, self.dimension)

        prefixed = [f"search_document: {t}" for t in texts]

        return self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query (prefixed with 'search_query: ').

        Args:
            query: Query text to embed.

        Returns:
            numpy array of shape (dim,) with L2-normalized embedding.
        """
        if query in self._query_cache:
            return self._query_cache[query]
        self._load_model()

        prefixed = f"search_query: {query}"

        return self._model.encode(
            prefixed,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def embed_queries(self, queries: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed a batch of queries (prefixed with 'search_query: ').

        Args:
            queries: List of query texts to embed.
            show_progress: Whether to show a progress bar.

        Returns:
            numpy array of shape (n, dim) with L2-normalized embeddings.
        """
        self._load_model()

        if len(queries) == 0:
            return np.array([]).reshape(0, self.dimension)

        prefixed = [f"search_query: {q}" for q in queries]

        return self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
