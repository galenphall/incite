"""Fine-tuned embedders for citation retrieval."""

from pathlib import Path
from typing import Optional

import numpy as np

from incite.embeddings.base import BaseEmbedder
from incite.utils import get_best_device

# Default model paths (relative to project root).
# Override via EMBEDDERS["minilm-ft"]["model"] in factory.py or MINILM_FT_MODEL_PATH env var.
DEFAULT_MODEL_PATH = "models/minilm-citation-v4/final"
DEFAULT_GRANITE_MODEL_PATH = "models/granite-citation-v5/final"


class FineTunedMiniLMEmbedder(BaseEmbedder):
    """Fine-tuned MiniLM embedder for citation-specific retrieval.

    Loads a locally fine-tuned sentence-transformers model from disk.
    Same interface as MiniLMEmbedder but trained on citation context → paper pairs.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 64,
    ):
        super().__init__()
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.device = device if device is not None else get_best_device()
        self.batch_size = batch_size
        self._model = None
        self._dimension: Optional[int] = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            path = Path(self.model_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"Fine-tuned model not found at {path}. "
                    f"Run 'incite finetune train' first to create it."
                )

            self._model = SentenceTransformer(
                str(path),
                device=self.device,
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


class FineTunedGraniteEmbedder(BaseEmbedder):
    """Fine-tuned Granite-small-R2 embedder for citation retrieval.

    384-dim ModernBERT architecture with 8K context and asymmetric prefixes.
    Trained with 'query: ' / 'passage: ' prefixes for asymmetric retrieval.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 64,
    ):
        super().__init__()
        self.model_path = model_path or DEFAULT_GRANITE_MODEL_PATH
        self.device = device if device is not None else get_best_device()
        self.batch_size = batch_size
        self._model = None
        self._dimension: Optional[int] = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            path = Path(self.model_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"Fine-tuned Granite model not found at {path}. "
                    f"Train with: python scripts/train_granite.py"
                )

            self._model = SentenceTransformer(
                str(path),
                device=self.device,
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


class OnnxMiniLMEmbedder(BaseEmbedder):
    """ONNX-accelerated MiniLM embedder for faster CPU inference.

    Uses optimum.onnxruntime for 1.5-2x speedup over PyTorch on CPU.
    Requires: pip install optimum[onnxruntime]

    Export a trained model first:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("models/minilm-citation-v4/final")
        model.export_onnx("models/minilm-citation-v4/onnx/")
    """

    DEFAULT_ONNX_PATH = "models/minilm-citation-v4/onnx"

    def __init__(
        self,
        model_path: Optional[str] = None,
        batch_size: int = 64,
    ):
        super().__init__()
        self.model_path = model_path or self.DEFAULT_ONNX_PATH
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        self._dimension: Optional[int] = None

    def _load_model(self):
        if self._model is not None:
            return

        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {path}. Export with: model.export_onnx('{path}')"
            )

        self._tokenizer = AutoTokenizer.from_pretrained(str(path))
        self._model = ORTModelForFeatureExtraction.from_pretrained(str(path))
        self._dimension = 384  # MiniLM-L6-v2 output dimension

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def _mean_pooling(self, model_output, attention_mask) -> np.ndarray:
        """Mean pooling with attention mask."""
        token_embeddings = model_output[0]
        if hasattr(token_embeddings, "numpy"):
            token_embeddings = token_embeddings.numpy()

        if hasattr(attention_mask, "numpy"):
            attention_mask = attention_mask.numpy()

        input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape).astype(
            np.float32
        )

        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        self._load_model()

        if len(texts) == 0:
            return np.array([]).reshape(0, self.dimension)

        all_embeddings = []
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="ONNX encoding", total=len(texts) // self.batch_size + 1)

        for i in iterator:
            batch = texts[i : i + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            outputs = self._model(**encoded)
            embeddings = self._mean_pooling(outputs, encoded["attention_mask"])

            # L2 normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings).astype(np.float32)
