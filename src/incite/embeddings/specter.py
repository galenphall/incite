"""SPECTER2 embedding model for scientific papers."""

from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from incite.embeddings.base import BaseEmbedder
from incite.utils import get_best_device


class SPECTEREmbedder(BaseEmbedder):
    """SPECTER2 embedder for scientific papers.

    SPECTER2 is trained on scientific paper pairs and produces embeddings
    where citation-linked papers are close together.

    Uses the adapters library as recommended by AllenAI.
    """

    DIMENSION = 768  # SPECTER2 output dimension

    def __init__(
        self,
        model_name: str = "allenai/specter2_base",
        adapter_name: str = "allenai/specter2",
        device: Optional[str] = None,
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.adapter_name = adapter_name
        self.device = device if device is not None else get_best_device()
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load the model with adapter."""
        if self._model is None:
            from adapters import AutoAdapterModel
            from transformers import AutoTokenizer

            print(f"Using device: {self.device}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoAdapterModel.from_pretrained(self.model_name)

            # Load the proximity adapter for retrieval tasks
            self._model.load_adapter(
                self.adapter_name, source="hf", load_as="proximity", set_active=True
            )
            # Explicitly activate the adapter
            self._model.set_active_adapters("proximity")
            self._model.to(self.device)
            self._model.eval()

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self.DIMENSION

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed a batch of texts.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape (n, dim)
        """
        self._load_model()

        if len(texts) == 0:
            return np.array([]).reshape(0, self.dimension)

        all_embeddings = []
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding")

        with torch.no_grad():
            for i in iterator:
                batch = texts[i : i + self.batch_size]
                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self._model(**inputs)
                # Take CLS token embedding (first token)
                embeddings = outputs.last_hidden_state[:, 0, :]

                # Normalize embeddings
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


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
        self.device = device if device is not None else get_best_device()
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._model = None
        self._dimension: Optional[int] = None

    def _load_model(self):
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
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
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
    and Matryoshka support. Uses same prefixes as Nomic Embed v1.5.
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
        self.device = device if device is not None else get_best_device()
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._model = None
        self._dimension: Optional[int] = None

    def _load_model(self):
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
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
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


class MiniLMEmbedder(BaseEmbedder):
    """Smaller, faster embedder using MiniLM.

    Good for quick prototyping or when compute is limited.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 64,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device if device is not None else get_best_device()
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._model = None
        self._dimension: Optional[int] = None

    def _load_model(self):
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
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
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


class E5Embedder(BaseEmbedder):
    """E5 embedder from Microsoft.

    E5 models require specific prefixes: "query: " for queries and
    "passage: " for documents. This class handles that automatically.
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
        self.device = device if device is not None else get_best_device()
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._model = None
        self._dimension: Optional[int] = None

    def _load_model(self):
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
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed documents (prefixed with 'passage: ')."""
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
        """Embed a query (prefixed with 'query: ')."""
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
        """Embed a batch of queries (prefixed with 'query: ')."""
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
        self.device = device if device is not None else get_best_device()
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._model = None
        self._dimension: Optional[int] = None

    def _load_model(self):
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
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed documents (prefixed with 'passage: ')."""
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
        """Embed a query (prefixed with 'query: ')."""
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
        """Embed a batch of queries (prefixed with 'query: ')."""
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
        self.device = device if device is not None else get_best_device()
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._model = None
        self._dimension: Optional[int] = None

    def _load_model(self):
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
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed documents (prefixed with 'search_document: ')."""
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
        """Embed a query (prefixed with 'search_query: ')."""
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
        """Embed a batch of queries (prefixed with 'search_query: ')."""
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
