"""Embedder and chunking strategy registries.

Central registry of all available embedding models and chunking strategies.
New embedders/chunkers must be registered here. The factory functions in
factory.py consume these registries to build retrievers.

Registry dicts use lazy loading — each entry specifies a module path and
class name, loaded on first use via get_embedder()/get_chunker().

Key constants:
    EMBEDDERS: Dict mapping embedder name → {module, class, dimension, ...}
    CHUNKING_STRATEGIES: Dict mapping strategy name → {module, function}
    DEFAULT_EMBEDDER: Default embedder key (currently "minilm")
    DEFAULT_CHUNKING: Default chunking strategy (currently "paragraph")

Related modules:
    - incite.retrieval.factory: Retriever/index builder functions.
    - incite.embeddings.*: Actual embedder implementations.
    - incite.corpus.chunking: Chunking implementations.
"""

import logging
from pathlib import Path
from typing import Callable

from incite.embeddings.base import BaseEmbedder
from incite.models import Chunk, Paper

logger = logging.getLogger(__name__)

# Available embedder configurations.
# Each entry uses lazy loading: the module and class are imported on first use
# by get_embedder(). This avoids importing all embedder backends at startup.
#
# Required fields:
#   name   — Human-readable display name
#   class  — Class name within the module
#   module — Dotted module path (imported lazily via __import__)
#
# Optional fields:
#   model        — Path or HuggingFace model ID passed as model_path= to the class
#   storage_key  — Override the FAISS index namespace (used by ONNX variants that
#                  produce identical embeddings to their PyTorch counterpart, so
#                  they can share the same stored index)
#   cloud_only   — If True + "model" is a local path, skip if the path doesn't exist
#                  (guards open-source users from trying to load proprietary weights)
EMBEDDERS = {
    "minilm": {
        "name": "MiniLM (fast, recommended)",
        "class": "MiniLMEmbedder",
        "module": "incite.embeddings.minilm",
    },
    "e5": {
        "name": "E5-small (good quality, fast)",
        "class": "E5Embedder",
        "module": "incite.embeddings.other_embedders",
    },
    "specter": {
        "name": "SPECTER2 (scientific, slower)",
        "class": "SPECTEREmbedder",
        "module": "incite.embeddings.specter",
    },
    "nomic": {
        "name": "Nomic Embed v1.5 (768d, 8K context)",
        "class": "NomicEmbedder",
        "module": "incite.embeddings.other_embedders",
    },
    "voyage": {
        "name": "Voyage AI voyage-4 (API, 1024d)",
        "class": "VoyageEmbedder",
        "module": "incite.embeddings.voyage",
    },
    "minilm-ft": {
        "name": "MiniLM fine-tuned v4 (citation-specific, Matryoshka)",
        "class": "FineTunedMiniLMEmbedder",
        "module": "incite.embeddings.finetuned",
        "model": "galenphall/minilm-citation-v4",
    },
    "minilm-ft-onnx": {
        "name": "MiniLM fine-tuned v4 ONNX (fast CPU inference)",
        "class": "OnnxMiniLMEmbedder",
        "module": "incite.embeddings.finetuned",
    },
    "modernbert": {
        "name": "ModernBERT-embed-base (768d, 8K context)",
        "class": "ModernBERTEmbedder",
        "module": "incite.embeddings.other_embedders",
    },
    "scincl": {
        "name": "SciNCL (citation-graph trained, 768d)",
        "class": "SciNCLEmbedder",
        "module": "incite.embeddings.other_embedders",
    },
    "granite": {
        "name": "Granite-small-R2 base (384d, 8K context)",
        "class": "GraniteEmbedder",
        "module": "incite.embeddings.other_embedders",
    },
    "granite-ft": {
        "name": "Granite-small-R2 fine-tuned v6b (384d, Matryoshka)",
        "class": "FineTunedGraniteEmbedder",
        "module": "incite.embeddings.finetuned",
        "model": "models/granite-citation-v6/final",
        "cloud_only": True,
    },
    "granite-ft-onnx": {
        "name": "Granite-small-R2 fine-tuned v6b ONNX (fast CPU inference)",
        "class": "OnnxGraniteEmbedder",
        "module": "incite.embeddings.finetuned",
        "model": "models/granite-citation-v6/onnx",
        # ONNX produces identical embeddings to granite-ft; share the same index
        "storage_key": "granite-ft",
        "cloud_only": True,
    },
}

DEFAULT_EMBEDDER = "minilm"  # Current best performer with hybrid


def get_storage_key(embedder_type: str) -> str:
    """Return the storage key for an embedder type.

    ONNX variants produce identical embeddings to their PyTorch counterparts,
    so they share the same storage namespace (FAISS index dir, pgvector rows).
    If no explicit storage_key is configured, the embedder_type itself is used.

    Args:
        embedder_type: Key from EMBEDDERS dict

    Returns:
        Storage key string (may differ from embedder_type for ONNX variants)
    """
    config = EMBEDDERS.get(embedder_type, {})
    return config.get("storage_key", embedder_type)


def get_available_embedders() -> dict[str, dict]:
    """Return embedders available for local use.

    Excludes cloud-only models whose local model files are not present.
    This lets open-source users see only the models they can actually run.

    Returns:
        Dict of embedder_key → config for all locally runnable embedders
    """
    available = {}
    for key, config in EMBEDDERS.items():
        if config.get("cloud_only") and "model" in config:
            if not Path(config["model"]).exists():
                continue
        available[key] = config
    return available


# Available chunking strategies.
# Each entry lazily imports the chunking function on first use via get_chunker().
#
# Required fields:
#   name     — Human-readable display name
#   function — Function name within the module
#   module   — Dotted module path (imported lazily)
#   description — Brief description for CLI help text
CHUNKING_STRATEGIES = {
    "paragraph": {
        "name": "Paragraph-based (default)",
        "function": "chunk_papers",
        "module": "incite.corpus.chunking",
        "description": "Split on paragraph boundaries, detect headings",
    },
    "grobid": {
        "name": "GROBID ML-based (requires Docker)",
        "function": "chunk_papers_grobid",
        "module": "incite.corpus.grobid_chunking",
        "description": (
            "ML-based structure detection via GROBID. "
            "~90% accuracy, references extracted separately. "
            "Requires: docker run -p 8070:8070 grobid/grobid:0.8.0"
        ),
    },
    "sentence": {
        "name": "Sentence-level (spaCy)",
        "function": "chunk_papers_sentences",
        "module": "incite.corpus.sentence_chunking",
        "description": (
            "Split on sentence boundaries with context injection. "
            "Each chunk includes: title | section | previous sentence. "
            "Finer granularity than paragraph (~7x more chunks)."
        ),
    },
    # Future strategies can be added here:
    # "semantic": {
    #     "name": "Semantic chunking",
    #     "function": "semantic_chunk_papers",
    #     "module": "incite.corpus.semantic_chunking",
    # },
    # "late": {
    #     "name": "Late chunking (embed full doc, pool to chunks)",
    #     "function": "late_chunk_papers",
    #     "module": "incite.corpus.late_chunking",
    # },
}

DEFAULT_CHUNKING = "paragraph"


def get_chunker(
    strategy: str = DEFAULT_CHUNKING,
) -> Callable[[list[Paper]], list[Chunk]]:
    """Get a chunking function by strategy name.

    Lazily imports the chunking module on first call so that optional
    dependencies (spaCy, GROBID client) are only required when used.

    Args:
        strategy: Key from CHUNKING_STRATEGIES dict

    Returns:
        Callable that takes list[Paper] and returns list[Chunk]

    Raises:
        ValueError: If strategy is not a key in CHUNKING_STRATEGIES
    """
    if strategy not in CHUNKING_STRATEGIES:
        raise ValueError(
            f"Unknown chunking strategy: {strategy}. "
            f"Available: {list(CHUNKING_STRATEGIES.keys())}"
        )

    config = CHUNKING_STRATEGIES[strategy]
    module = __import__(config["module"], fromlist=[config["function"]])
    return getattr(module, config["function"])


# Module-level cache: maps embedder_type → instantiated embedder.
# Avoids reloading model weights on repeated calls within the same process.
_embedder_cache: dict[str, BaseEmbedder] = {}


def get_embedder(embedder_type: str = DEFAULT_EMBEDDER) -> BaseEmbedder:
    """Get a cached embedder instance by type.

    Returns the same instance for repeated calls with the same embedder_type,
    avoiding expensive model reloads.

    Auto-fallback: if "minilm-ft" is requested but sentence_transformers is not
    installed, falls back to "minilm-ft-onnx" (which only needs onnxruntime).

    Args:
        embedder_type: Key from EMBEDDERS dict ("minilm", "specter", etc.)

    Returns:
        Configured embedder instance (cached for the lifetime of the process)

    Raises:
        ValueError: If embedder_type is unknown or cloud-only without local model files.
    """
    # Auto-fallback: minilm-ft → minilm-ft-onnx if torch/sentence_transformers unavailable
    if embedder_type == "minilm-ft":
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            logger.info("sentence_transformers not available, falling back to minilm-ft-onnx")
            embedder_type = "minilm-ft-onnx"

    if embedder_type in _embedder_cache:
        return _embedder_cache[embedder_type]

    if embedder_type not in EMBEDDERS:
        raise ValueError(
            f"Unknown embedder: {embedder_type}. Available: {list(EMBEDDERS.keys())}"
        )

    config = EMBEDDERS[embedder_type]

    # Guard: cloud-only models require local model files to be present
    if config.get("cloud_only") and "model" in config:
        model_path = Path(config["model"])
        if not model_path.exists():
            raise ValueError(
                f"'{embedder_type}' requires model files that are not included in the "
                f"open-source release. Use 'minilm-ft' instead (MRR 0.428), or subscribe "
                f"to inCite Cloud at https://inciteref.com for access to Granite-FT "
                f"(MRR 0.550, +28% better)."
            )

    # Lazy-import the embedder class and instantiate
    module = __import__(config["module"], fromlist=[config["class"]])
    embedder_class = getattr(module, config["class"])
    kwargs: dict = {}
    if "model" in config:
        kwargs["model_path"] = config["model"]
    embedder = embedder_class(**kwargs)
    _embedder_cache[embedder_type] = embedder
    return embedder
