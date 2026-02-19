"""Factory functions for creating retrievers.

This module provides the plug-and-chug entry point for creating retrievers.
As the pipeline evolves (new embedders, better fusion methods), update this
module and all consumers automatically benefit.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Optional

from incite.embeddings.base import BaseEmbedder
from incite.interfaces import Retriever
from incite.models import Chunk, Paper

if TYPE_CHECKING:
    from incite.embeddings.chunk_store import ChunkStore
    from incite.retrieval.two_stage import TwoStageRetriever

# Available embedder configurations
# Add new embedders here as they become available
EMBEDDERS = {
    "minilm": {
        "name": "MiniLM (fast, recommended)",
        "class": "MiniLMEmbedder",
        "module": "incite.embeddings.specter",
    },
    "e5": {
        "name": "E5-small (good quality, fast)",
        "class": "E5Embedder",
        "module": "incite.embeddings.specter",
    },
    "specter": {
        "name": "SPECTER2 (scientific, slower)",
        "class": "SPECTEREmbedder",
        "module": "incite.embeddings.specter",
    },
    "nomic": {
        "name": "Nomic Embed v1.5 (768d, 8K context)",
        "class": "NomicEmbedder",
        "module": "incite.embeddings.specter",
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
        "module": "incite.embeddings.specter",
    },
    "scincl": {
        "name": "SciNCL (citation-graph trained, 768d)",
        "class": "SciNCLEmbedder",
        "module": "incite.embeddings.specter",
    },
    "granite": {
        "name": "Granite-small-R2 base (384d, 8K context)",
        "class": "GraniteEmbedder",
        "module": "incite.embeddings.specter",
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
        "storage_key": "granite-ft",  # ONNX produces identical embeddings; share index
        "cloud_only": True,
    },
}

DEFAULT_EMBEDDER = "minilm"  # Current best performer with hybrid


def get_storage_key(embedder_type: str) -> str:
    """Return the storage key for an embedder type.

    ONNX variants produce identical embeddings to their PyTorch counterparts,
    so they share the same storage namespace (FAISS index dir, pgvector rows).
    """
    config = EMBEDDERS.get(embedder_type, {})
    return config.get("storage_key", embedder_type)


def get_available_embedders() -> dict[str, dict]:
    """Return embedders available for local use (excludes cloud-only models without local files)."""
    available = {}
    for key, config in EMBEDDERS.items():
        if config.get("cloud_only") and "model" in config:
            if not Path(config["model"]).exists():
                continue
        available[key] = config
    return available


# Available chunking strategies
# Add new strategies here (semantic chunking, late chunking, etc.)
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

    Args:
        strategy: Key from CHUNKING_STRATEGIES dict

    Returns:
        Callable that takes list[Paper] and returns list[Chunk]
    """
    if strategy not in CHUNKING_STRATEGIES:
        raise ValueError(
            f"Unknown chunking strategy: {strategy}. Available: {list(CHUNKING_STRATEGIES.keys())}"
        )

    config = CHUNKING_STRATEGIES[strategy]
    module = __import__(config["module"], fromlist=[config["function"]])
    return getattr(module, config["function"])


_embedder_cache: dict[str, BaseEmbedder] = {}


def get_embedder(embedder_type: str = DEFAULT_EMBEDDER) -> BaseEmbedder:
    """Get a cached embedder instance by type.

    Returns the same instance for repeated calls with the same embedder_type,
    avoiding expensive model reloads.

    Args:
        embedder_type: Key from EMBEDDERS dict ("minilm", "specter", etc.)

    Returns:
        Configured embedder instance (cached)

    Raises:
        ValueError: If embedder_type is unknown or cloud-only without local model files.
    """
    if embedder_type in _embedder_cache:
        return _embedder_cache[embedder_type]

    if embedder_type not in EMBEDDERS:
        raise ValueError(f"Unknown embedder: {embedder_type}. Available: {list(EMBEDDERS.keys())}")

    config = EMBEDDERS[embedder_type]

    # Guard: cloud-only models require local model files
    if config.get("cloud_only") and "model" in config:
        model_path = Path(config["model"])
        if not model_path.exists():
            raise ValueError(
                f"'{embedder_type}' requires model files that are not included in the "
                f"open-source release. Use 'minilm-ft' instead (MRR 0.428), or subscribe "
                f"to inCite Cloud at https://inciteref.com for access to Granite-FT "
                f"(MRR 0.550, +28% better)."
            )

    module = __import__(config["module"], fromlist=[config["class"]])
    embedder_class = getattr(module, config["class"])
    kwargs = {}
    if "model" in config:
        kwargs["model_path"] = config["model"]
    embedder = embedder_class(**kwargs)
    _embedder_cache[embedder_type] = embedder
    return embedder


def create_retriever(
    papers: list[Paper],
    method: Literal["neural", "bm25", "hybrid"] = "hybrid",
    embedder_type: str = DEFAULT_EMBEDDER,
    index_path: Optional[Path] = None,
    fusion: str = "rrf",
    show_progress: bool = True,
    include_metadata: bool = True,
    storage_backend: Optional[object] = None,
) -> Retriever:
    """Create a retriever using current best practices.

    This is the plug-and-chug entry point. As the pipeline evolves,
    update this function and all consumers benefit automatically.

    Args:
        papers: List of Paper objects to search over
        method: Retrieval method ("neural", "bm25", or "hybrid")
        embedder_type: Which embedder to use ("minilm", "specter", "e5")
        index_path: Path to pre-built FAISS index (optional, will build if needed)
        fusion: Fusion method for hybrid ("rrf" or "weighted")
        show_progress: Whether to show progress bars when building
        include_metadata: Include author/year/journal in document embeddings
        storage_backend: Pre-built VectorStore (e.g., PgVectorStore). Skips FAISS.

    Returns:
        Configured Retriever instance
    """
    from incite.embeddings.stores import FAISSStore
    from incite.retrieval.bm25 import BM25Retriever
    from incite.retrieval.hybrid import HybridRetriever
    from incite.retrieval.neural import NeuralRetriever

    paper_dict = {p.id: p for p in papers}

    if method == "bm25":
        return BM25Retriever.from_papers(papers, include_metadata=include_metadata)

    # Neural or hybrid both need embedder and store
    embedder = get_embedder(embedder_type)

    if storage_backend is not None:
        # Use pre-built vector store (e.g., pgvector)
        neural = NeuralRetriever(embedder=embedder, store=storage_backend, papers=paper_dict)
    elif index_path and Path(index_path).exists():
        store = FAISSStore()
        store.load(str(index_path))
        neural = NeuralRetriever(embedder=embedder, store=store, papers=paper_dict)
    else:
        neural = NeuralRetriever.from_papers(
            papers, embedder, show_progress=show_progress, include_metadata=include_metadata
        )

    if method == "neural":
        return neural

    # Hybrid: combine neural + stemmed BM25
    # Per-embedder fusion params tuned via scripts/hybrid_sweep.py (3420 queries):
    #   MiniLM-FT v4: stemmed BM25, RRF k=5, 1:1 weights → MRR=0.428
    #   Granite-FT v5: stemmed BM25, RRF k=10, 2:1 neural:bm25 → MRR=0.437
    hybrid_params = {
        "granite-ft": {"rrf_k": 10, "neural_weight": 2.0, "bm25_weight": 1.0},
    }
    defaults = {"rrf_k": 5, "neural_weight": 1.0, "bm25_weight": 1.0}
    params = hybrid_params.get(embedder_type, defaults)

    from incite.retrieval.bm25 import tokenize_with_stemming

    bm25 = BM25Retriever.from_papers(
        papers, tokenizer=tokenize_with_stemming, include_metadata=include_metadata
    )
    return HybridRetriever(
        retrievers=[(neural, params["neural_weight"]), (bm25, params["bm25_weight"])],
        fusion=fusion,
        rrf_k=params["rrf_k"],
    )


def build_index(
    papers: list[Paper],
    output_path: Path,
    embedder_type: str = DEFAULT_EMBEDDER,
    show_progress: bool = True,
    include_metadata: bool = True,
    embeddings=None,
) -> None:
    """Build and save a FAISS index from papers.

    Args:
        papers: List of Paper objects to index
        output_path: Directory to save index files
        embedder_type: Which embedder to use ("minilm", "specter", "e5")
        show_progress: Whether to show progress bar
        include_metadata: Include author/year/journal in document embeddings
        embeddings: Optional pre-computed numpy embeddings (skips local embedding)
    """
    from incite.retrieval.neural import _build_paper_store

    embedder = get_embedder(embedder_type)
    _, store = _build_paper_store(
        papers,
        embedder,
        show_progress=show_progress,
        include_metadata=include_metadata,
        embedder_type=embedder_type,
        embeddings=embeddings,
    )
    store.save(str(Path(output_path)))


def create_two_stage_retriever(
    papers: list[Paper],
    chunk_store: "ChunkStore",
    chunks: dict[str, Chunk],
    embedder_type: str = DEFAULT_EMBEDDER,
    index_path: Optional[Path] = None,
    alpha: float = 0.6,
    stage1_k: int = 50,
    evidence_threshold: float = 0.35,
    max_evidence_per_paper: int = 3,
    show_progress: bool = True,
) -> "TwoStageRetriever":
    """Create a two-stage retriever: paper-level hybrid + paragraph reranking.

    Stage 1 uses HybridRetriever (neural + stemmed BM25) for candidate papers.
    Stage 2 searches only the chunks belonging to those candidates and blends
    paper-level and chunk-level scores.

    Args:
        papers: List of Paper objects for the corpus
        chunk_store: Pre-loaded ChunkStore with paragraph embeddings
        chunks: Dict mapping chunk_id -> Chunk for evidence text
        embedder_type: Embedder to use for stage 1 and query embedding
        index_path: Path to pre-built paper FAISS index (optional)
        alpha: Weight for paper score in blend (0-1). 0.6 means 60% paper, 40% chunk.
        stage1_k: Number of papers to pass from stage 1 to stage 2
        evidence_threshold: Minimum chunk score to attach evidence snippets
        max_evidence_per_paper: Maximum evidence snippets per paper
        show_progress: Whether to show progress bars when building indexes

    Returns:
        Configured TwoStageRetriever instance
    """
    from incite.retrieval.two_stage import TwoStageRetriever

    # Build the stage 1 hybrid retriever
    paper_retriever = create_retriever(
        papers=papers,
        method="hybrid",
        embedder_type=embedder_type,
        index_path=index_path,
        show_progress=show_progress,
    )

    embedder = get_embedder(embedder_type)

    return TwoStageRetriever(
        paper_retriever=paper_retriever,
        chunk_store=chunk_store,
        chunks=chunks,
        embedder=embedder,
        alpha=alpha,
        stage1_k=stage1_k,
        evidence_threshold=evidence_threshold,
        max_evidence_per_paper=max_evidence_per_paper,
    )


def create_paragraph_retriever(
    chunks: list[Chunk],
    papers: list[Paper],
    embedder_type: str = DEFAULT_EMBEDDER,
    index_path: Optional[Path] = None,
    method: Literal["neural", "hybrid"] = "hybrid",
    aggregation: str = "max",
    show_progress: bool = True,
    storage_backend: Optional[object] = None,
) -> Retriever:
    """Create a paragraph-level retriever.

    Args:
        chunks: List of Chunk objects (with optional context_text from enrichment)
        papers: List of Paper objects (for BM25 and paper lookup)
        embedder_type: Which embedder to use (default: DEFAULT_EMBEDDER)
        index_path: Path to pre-built chunk index (optional)
        method: "neural" for chunk-only, "hybrid" for chunk neural + paper BM25
        aggregation: How to aggregate chunk scores to paper level ("max", "mean", "sum")
        show_progress: Whether to show progress bars
        storage_backend: Pre-built ChunkStore (e.g., PgVectorChunkStore). Skips FAISS.

    Returns:
        Configured Retriever (ParagraphRetriever or HybridParagraphRetriever)
    """
    from incite.embeddings.chunk_store import ChunkStore
    from incite.retrieval.bm25 import BM25Retriever
    from incite.retrieval.paragraph import (
        HybridParagraphRetriever,
        ParagraphRetriever,
    )

    embedder = get_embedder(embedder_type)
    chunk_dict = {c.id: c for c in chunks}

    if storage_backend is not None:
        # Use pre-built chunk store (e.g., pgvector)
        para_retriever = ParagraphRetriever(
            embedder=embedder,
            chunk_store=storage_backend,
            chunks=chunk_dict,
            aggregation=aggregation,
        )
    elif index_path and Path(index_path).exists():
        chunk_store = ChunkStore()
        chunk_store.load(str(index_path))
        para_retriever = ParagraphRetriever(
            embedder=embedder,
            chunk_store=chunk_store,
            chunks=chunk_dict,
            aggregation=aggregation,
        )
    else:
        para_retriever = ParagraphRetriever.from_chunks(
            chunks, embedder, aggregation=aggregation, show_progress=show_progress
        )

    if method == "neural":
        return para_retriever

    # Hybrid: combine paragraph neural + stemmed BM25
    from incite.retrieval.bm25 import tokenize_with_stemming

    bm25 = BM25Retriever.from_papers(
        papers, tokenizer=tokenize_with_stemming, include_metadata=True
    )
    return HybridParagraphRetriever(
        paragraph_retriever=para_retriever,
        bm25_retriever=bm25,
        neural_weight=1.0,
        bm25_weight=1.0,
        rrf_k=10,
    )


def build_chunk_index(
    chunks: list[Chunk],
    output_path: Path,
    embedder_type: str = DEFAULT_EMBEDDER,
    show_progress: bool = True,
    progress_callback=None,
) -> None:
    """Build and save a chunk index.

    Args:
        chunks: List of Chunk objects to index
        output_path: Directory to save index files
        embedder_type: Which embedder to use (default: DEFAULT_EMBEDDER)
        show_progress: Whether to show progress bar
        progress_callback: Optional callback(current, total, message) for UI progress
    """
    from incite.embeddings.chunk_store import build_chunk_index as _build

    embedder = get_embedder(embedder_type)

    _build(
        chunks=chunks,
        embedder=embedder,
        output_path=output_path,
        show_progress=show_progress,
        progress_callback=progress_callback,
    )


def create_multi_scale_retriever(
    papers: list[Paper],
    base_dir: Path,
    embedder_type: str = DEFAULT_EMBEDDER,
    weights: dict[str, float] = None,
    show_progress: bool = True,
) -> Retriever:
    """Create a MultiScaleRetriever from a pre-built multi-scale index.

    Requires `base_dir` to contain:
    - paper/index.faiss
    - paragraph/index.faiss
    - sentence/index.faiss
    - chunks_paragraph.jsonl
    - chunks_sentence.jsonl

    Args:
        papers: List of Paper objects
        base_dir: Directory containing multi-scale indexes
        embedder_type: Embedder used for indexing
        weights: Optional weighting for fusion (paper, paragraph, sentence)
        show_progress: Whether to show progress bars

    Returns:
        Configured MultiScaleRetriever
    """
    from incite.corpus.loader import load_chunks
    from incite.embeddings.chunk_store import ChunkStore
    from incite.embeddings.stores import FAISSStore
    from incite.retrieval.multi_scale import MultiScaleRetriever
    from incite.retrieval.neural import NeuralRetriever
    from incite.retrieval.paragraph import ParagraphRetriever

    base_dir = Path(base_dir)
    embedder = get_embedder(embedder_type)
    paper_dict = {p.id: p for p in papers}

    # 1. Paper Retriever
    if not (base_dir / "paper/index.faiss").exists():
        raise FileNotFoundError(f"Paper index not found at {base_dir / 'paper'}")

    paper_store = FAISSStore()
    paper_store.load(str(base_dir / "paper"))
    paper_retriever = NeuralRetriever(embedder=embedder, store=paper_store, papers=paper_dict)

    # 2. Paragraph Retriever
    if not (base_dir / "chunks_paragraph.jsonl").exists():
        raise FileNotFoundError(f"Paragraph chunks not found at {base_dir}")
    if not (base_dir / "paragraph/index.faiss").exists():
        raise FileNotFoundError(f"Paragraph index not found at {base_dir / 'paragraph'}")

    para_chunks = load_chunks(str(base_dir / "chunks_paragraph.jsonl"))
    para_chunk_dict = {c.id: c for c in para_chunks}

    para_store = ChunkStore()
    para_store.load(str(base_dir / "paragraph"))

    paragraph_retriever = ParagraphRetriever(
        embedder=embedder,
        chunk_store=para_store,
        chunks=para_chunk_dict,
        aggregation="max",
    )

    # 3. Sentence Retriever
    if not (base_dir / "chunks_sentence.jsonl").exists():
        raise FileNotFoundError(f"Sentence chunks not found at {base_dir}")
    if not (base_dir / "sentence/index.faiss").exists():
        raise FileNotFoundError(f"Sentence index not found at {base_dir / 'sentence'}")

    sent_chunks = load_chunks(str(base_dir / "chunks_sentence.jsonl"))
    sent_chunk_dict = {c.id: c for c in sent_chunks}

    sent_store = ChunkStore()
    sent_store.load(str(base_dir / "sentence"))

    sentence_retriever = ParagraphRetriever(
        embedder=embedder,
        chunk_store=sent_store,
        chunks=sent_chunk_dict,
        aggregation="max",
    )

    return MultiScaleRetriever(
        paper_retriever=paper_retriever,
        paragraph_retriever=paragraph_retriever,
        sentence_retriever=sentence_retriever,
        weights=weights,
    )


def build_multi_scale_index(
    papers: list[Paper],
    output_dir: Path,
    embedder_type: str = DEFAULT_EMBEDDER,
    show_progress: bool = True,
) -> None:
    """Build paper, paragraph, and sentence indexes from corpus.

    Creates three indexes in subdirectories:
      - output_dir/paper/
      - output_dir/paragraph/
      - output_dir/sentence/

    Also saves the generated chunks to:
      - output_dir/chunks_paragraph.jsonl
      - output_dir/chunks_sentence.jsonl

    Args:
        papers: List of Paper objects
        output_dir: Base directory for indexes
        embedder_type: Embedder to use for all indexes
        show_progress: Whether to show progress bars
    """
    from incite.corpus.chunking import chunk_papers
    from incite.corpus.loader import save_chunks
    from incite.corpus.sentence_chunking import chunk_papers_sentences

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Paper Index
    print("Building paper-level index...")
    build_index(
        papers=papers,
        output_path=output_dir / "paper",
        embedder_type=embedder_type,
        show_progress=show_progress,
    )

    # 2. Paragraph Index
    print("Generating paragraph chunks...")
    paragraph_chunks = chunk_papers(papers, show_progress=show_progress)
    save_chunks(paragraph_chunks, output_dir / "chunks_paragraph.jsonl")

    print(f"Building paragraph-level index ({len(paragraph_chunks)} chunks)...")
    build_chunk_index(
        chunks=paragraph_chunks,
        output_path=output_dir / "paragraph",
        embedder_type=embedder_type,
        show_progress=show_progress,
    )

    # 3. Sentence Index
    print("Generating sentence chunks...")
    sentence_chunks = chunk_papers_sentences(papers, show_progress=show_progress)
    save_chunks(sentence_chunks, output_dir / "chunks_sentence.jsonl")

    print(f"Building sentence-level index ({len(sentence_chunks)} chunks)...")
    build_chunk_index(
        chunks=sentence_chunks,
        output_path=output_dir / "sentence",
        embedder_type=embedder_type,
        show_progress=show_progress,
    )
