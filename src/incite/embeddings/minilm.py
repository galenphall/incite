"""MiniLM embedding model for fast baseline retrieval.

Uses all-MiniLM-L6-v2 (384-dim) from sentence-transformers.
Fast inference, reasonable quality. Used as the default embedder
for new installations before fine-tuned models are available.

Related modules:
    - incite.embeddings.base: BaseEmbedder ABC.
    - incite.embeddings.finetuned: Fine-tuned MiniLM variants.
    - incite.retrieval.factory_registry: Registers this as "minilm".
"""

from typing import Optional

import numpy as np

from incite.embeddings.base import BaseEmbedder


def _get_best_device() -> str:
    """Return best available device: MPS (Apple Silicon) > CUDA > CPU."""
    from incite.utils import get_best_device

    return get_best_device()


class MiniLMEmbedder(BaseEmbedder):
    """Smaller, faster embedder using MiniLM.

    Uses all-MiniLM-L6-v2 (384-dim) from sentence-transformers.
    Good for quick prototyping or when compute is limited.
    Suitable as a default embedder before fine-tuned variants are available.
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
