"""SPECTER2 embedding model for scientific papers.

SPECTER2 is trained on scientific paper pairs and produces embeddings
where citation-linked papers are close together. Uses the adapters library
as recommended by AllenAI.

Related modules:
    - incite.embeddings.base: BaseEmbedder ABC.
    - incite.embeddings.minilm: MiniLMEmbedder (fast baseline).
    - incite.embeddings.other_embedders: E5, Granite, Nomic, SciNCL, ModernBERT.
    - incite.retrieval.factory_registry: Registers this as "specter".
"""

from typing import Optional

import numpy as np
from tqdm import tqdm

from incite.embeddings.base import BaseEmbedder


def _get_best_device() -> str:
    """Lazy wrapper for get_best_device() to avoid top-level torch import."""
    from incite.utils import get_best_device

    return get_best_device()


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
        self.device = device if device is not None else _get_best_device()
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
            texts: List of texts to embed.
            show_progress: Whether to show progress bar.

        Returns:
            numpy array of shape (n, dim) with L2-normalized embeddings.
        """
        self._load_model()

        if len(texts) == 0:
            return np.array([]).reshape(0, self.dimension)

        all_embeddings = []
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding")

        import torch

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


# --- Re-exports for backward compatibility ---
from incite.embeddings.minilm import MiniLMEmbedder  # noqa: F401, E402
from incite.embeddings.other_embedders import (  # noqa: F401, E402
    E5Embedder,
    GraniteEmbedder,
    ModernBERTEmbedder,
    NomicEmbedder,
    SciNCLEmbedder,
)
