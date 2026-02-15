"""Agent-friendly testing interface for inCite.

Provides a Python SDK and JSON-friendly interface for AI coding agents
to test the recommendation system programmatically.

Example usage:

    # Initialize from Zotero library
    agent = InCiteAgent.from_zotero()

    # Get recommendations with timing info
    response = agent.recommend("text about climate change", k=10)
    print(f"Total time: {response.timing.total_ms}ms")
    for rec in response.recommendations:
        print(f"  {rec.rank}. {rec.title} ({rec.score:.3f})")

    # Batch queries
    responses = agent.batch_recommend(["query 1", "query 2"], parallel=True)
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TimingInfo:
    """Timing breakdown for a retrieval operation."""

    total_ms: float
    embed_query_ms: float = 0.0
    vector_search_ms: float = 0.0
    bm25_search_ms: Optional[float] = None
    fusion_ms: Optional[float] = None
    evidence_ms: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class AgentRecommendation:
    """A single recommendation with full metadata."""

    paper_id: str
    rank: int
    score: float
    title: str
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None  # First 300 chars
    bibtex_key: Optional[str] = None
    score_breakdown: dict[str, float] = field(default_factory=dict)
    matched_paragraph: Optional[str] = None  # For paragraph mode
    matched_paragraphs: list[dict] = field(default_factory=list)
    zotero_uri: Optional[str] = None
    confidence: float = 0.0  # Neural similarity confidence in [0, 1]

    def to_dict(self) -> dict:
        """Convert to dict, excluding None values."""
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class AgentResponse:
    """Complete response from a recommendation query."""

    query: str
    recommendations: list[AgentRecommendation]
    timing: TimingInfo
    corpus_size: int
    method: str
    embedder: str
    timestamp: str
    mode: str = "paper"  # "paper" or "paragraph"

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "query": self.query,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "timing": self.timing.to_dict(),
            "corpus_size": self.corpus_size,
            "method": self.method,
            "embedder": self.embedder,
            "mode": self.mode,
            "timestamp": self.timestamp,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class InCiteAgent:
    """Agent-friendly interface for testing inCite recommendations.

    Supports multiple initialization modes:
    - from_source(): Load from any CorpusSource (preferred, extensible)
    - from_zotero(): Load from local Zotero library (uses ~/.incite/ cache)
    - from_corpus(): Load from a corpus JSONL file

    And two retrieval modes:
    - paper (default): Search by paper title+abstract embeddings
    - paragraph: Search by paragraph embeddings (requires PDF extraction)
    """

    # Sentence splitting regex — matches the Obsidian plugin's logic
    _ABBREV_RE = re.compile(
        r"(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|etc|Fig|Figs|Eq|Eqs|al|vs|i\.e|e\.g|cf"
        r"|no|vol|pp|ed|Rev|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.\s*",
        re.IGNORECASE,
    )

    # Default threshold for showing paragraph evidence in paper mode.
    # Chunk cosine similarity must be >= this to attach evidence.
    # Matches the "medium confidence" badge threshold (0.35).
    EVIDENCE_THRESHOLD = 0.35

    def __init__(
        self,
        retriever,
        papers: dict,
        method: str,
        embedder_type: str,
        mode: str = "paper",
        chunks: Optional[list] = None,
        chunking_strategy: str = "paragraph",
        evidence_store=None,
        evidence_chunks: Optional[dict] = None,
        two_stage: bool = False,
        paper_store=None,
    ):
        """Internal constructor. Use from_source(), from_zotero(), or from_corpus() instead."""
        self._retriever = retriever
        self._papers = papers
        self._method = method
        self._embedder_type = embedder_type
        self._mode = mode
        self._chunks = chunks or []
        self._chunking_strategy = chunking_strategy
        self._evidence_store = evidence_store  # ChunkStore for evidence lookup
        self._evidence_chunks = evidence_chunks  # {chunk_id: Chunk} for text
        self._two_stage = two_stage  # True when using TwoStageRetriever
        self._paper_store = paper_store  # Paper-level VectorStore (for find_similar)
        self._embedder = self._find_embedder()

    # --- Public properties (used by api.py instead of accessing _private attrs) ---

    @property
    def papers(self) -> dict:
        """Dict mapping paper_id to Paper objects."""
        return self._papers

    @property
    def method(self) -> str:
        """Retrieval method name (neural, bm25, hybrid)."""
        return self._method

    @property
    def mode(self) -> str:
        """Retrieval mode (paper or paragraph)."""
        return self._mode

    @property
    def chunking_strategy(self) -> str:
        """Chunking strategy name (paragraph, sentence, grobid)."""
        return self._chunking_strategy

    @property
    def corpus_size(self) -> int:
        """Number of papers in the corpus."""
        return len(self._papers)

    @classmethod
    def from_source(
        cls,
        source,
        method: str = "hybrid",
        embedder_type: str = "minilm",
        mode: str = "paper",
        chunking_strategy: str = "paragraph",
    ) -> "InCiteAgent":
        """Initialize from any CorpusSource.

        This is the preferred, extensible constructor. from_zotero() and
        from_corpus() delegate to this method.

        Args:
            source: Any object satisfying the CorpusSource protocol
                (must have load_papers(), needs_refresh(), cache_key())
            method: Retrieval method ("neural", "bm25", "hybrid")
            embedder_type: Embedder to use ("minilm", "e5", "specter", etc.)
            mode: "paper" (title+abstract) or "paragraph" (PDF chunks)
            chunking_strategy: Chunking strategy for paragraph mode

        Returns:
            Configured InCiteAgent instance
        """
        from incite.retrieval.factory import create_paragraph_retriever, create_retriever

        papers = source.load_papers()
        paper_dict = {p.id: p for p in papers}

        if mode == "paragraph":
            retriever = create_paragraph_retriever(
                chunks=[],  # Will be populated below
                papers=papers,
                embedder_type=embedder_type,
                method=method if method != "bm25" else "hybrid",
                show_progress=True,
            )
            # Paragraph mode needs chunks — try loading from cache or building
            from incite.retrieval.factory import get_chunker

            chunker = get_chunker(chunking_strategy)
            chunks = chunker(papers, show_progress=True)

            retriever = create_paragraph_retriever(
                chunks=chunks,
                papers=papers,
                embedder_type=embedder_type,
                method=method if method != "bm25" else "hybrid",
                show_progress=True,
            )
            return cls(
                retriever=retriever,
                papers=paper_dict,
                method=method,
                embedder_type=embedder_type,
                mode="paragraph",
                chunks=chunks,
                chunking_strategy=chunking_strategy,
            )
        else:
            retriever = create_retriever(
                papers=papers,
                method=method,
                embedder_type=embedder_type,
                show_progress=True,
            )
            return cls(
                retriever=retriever,
                papers=paper_dict,
                method=method,
                embedder_type=embedder_type,
                mode="paper",
            )

    @classmethod
    def from_zotero(
        cls,
        zotero_dir: Optional[str] = None,
        method: str = "hybrid",
        embedder_type: str = "minilm",
        mode: str = "paper",
        force_refresh: bool = False,
        chunking_strategy: str = "paragraph",
    ) -> "InCiteAgent":
        """Initialize from local Zotero library.

        Uses the same caching infrastructure as the webapp (~/.incite/).
        First call may take 30-60s for a new library; subsequent calls are instant.

        Args:
            zotero_dir: Path to Zotero data directory. If None, auto-detects.
            method: Retrieval method ("neural", "bm25", "hybrid")
            embedder_type: Embedder to use ("minilm", "e5", "specter", "nomic", "voyage")
            mode: "paper" (title+abstract) or "paragraph" (PDF chunks)
            force_refresh: If True, re-read from Zotero database ignoring cache
            chunking_strategy: Chunking strategy for paragraph mode
                ("paragraph", "sentence", "grobid")

        Returns:
            Configured InCiteAgent instance
        """
        from incite.corpus.zotero_reader import find_zotero_data_dir
        from incite.webapp.state import (
            get_cache_dir,
            get_paragraph_retriever,
            get_retriever,
            has_chunks,
            load_zotero_chunks,
            load_zotero_direct,
        )

        # Auto-detect Zotero directory
        if zotero_dir is None:
            detected = find_zotero_data_dir()
            if detected is None:
                raise ValueError(
                    "Could not auto-detect Zotero directory. Please provide zotero_dir parameter."
                )
            zotero_dir = str(detected)

        zotero_path = Path(zotero_dir)

        # Load papers from Zotero (with caching)
        papers = load_zotero_direct(zotero_path, force_refresh=force_refresh)
        paper_dict = {p.id: p for p in papers}

        if mode == "paragraph":
            # Check if we have full text on papers OR cached chunks file
            cache_dir = get_cache_dir()
            cached_chunks = cache_dir / f"zotero_chunks_{chunking_strategy}.jsonl"
            if not has_chunks(papers) and not cached_chunks.exists():
                raise ValueError(
                    "No PDF text extracted. Run agent.extract_pdfs() first, "
                    "or use mode='paper' for title+abstract retrieval."
                )

            chunks = load_zotero_chunks(
                papers,
                chunking_strategy=chunking_strategy,
                force_rebuild=force_refresh,
            )
            retriever = get_paragraph_retriever(
                chunks=chunks,
                papers=papers,
                method=method if method != "bm25" else "hybrid",
                embedder_type=embedder_type,
                force_rebuild=force_refresh,
            )
            return cls(
                retriever=retriever,
                papers=paper_dict,
                method=method,
                embedder_type=embedder_type,
                mode="paragraph",
                chunks=chunks,
                chunking_strategy=chunking_strategy,
            )
        else:
            # Paper-level retrieval
            retriever = get_retriever(
                papers=papers,
                method=method,
                embedder_type=embedder_type,
                force_rebuild=force_refresh,
            )

            # Try to load evidence store for paragraph snippets (optional).
            # When a chunk store is available, create a TwoStageRetriever
            # so evidence affects ranking (not just display).
            evidence_store = None
            evidence_chunks = None
            two_stage = False
            try:
                cache_dir = get_cache_dir()
                cached_chunks_path = cache_dir / f"zotero_chunks_{chunking_strategy}.jsonl"
                if has_chunks(papers) or cached_chunks_path.exists():
                    from incite.webapp.state import get_evidence_store

                    evidence_chunks, evidence_store = get_evidence_store(
                        papers=papers,
                        embedder_type=embedder_type,
                        chunking_strategy=chunking_strategy,
                        force_rebuild=force_refresh,
                    )
                    logger.info(
                        "Evidence store loaded: %d chunks (%s)",
                        len(evidence_chunks),
                        chunking_strategy,
                    )

                    # Upgrade to TwoStageRetriever if we have a chunk store
                    if evidence_store is not None and evidence_chunks is not None:
                        from incite.retrieval.two_stage import TwoStageRetriever

                        embedder = cls._find_embedder_from(retriever)
                        if embedder is None:
                            from incite.retrieval.factory import get_embedder

                            embedder = get_embedder(embedder_type)

                        retriever = TwoStageRetriever(
                            paper_retriever=retriever,
                            chunk_store=evidence_store,
                            chunks=evidence_chunks,
                            embedder=embedder,
                        )
                        two_stage = True
                        logger.info("Using TwoStageRetriever (alpha=0.6)")
            except Exception as e:
                logger.warning("Could not load evidence store: %s", e)
                evidence_store = None
                evidence_chunks = None

            return cls(
                retriever=retriever,
                papers=paper_dict,
                method=method,
                embedder_type=embedder_type,
                mode="paper",
                evidence_store=evidence_store,
                evidence_chunks=evidence_chunks,
                two_stage=two_stage,
            )

    @classmethod
    def from_paperpile(
        cls,
        bibtex_url: Optional[str] = None,
        bibtex_path: Optional[str] = None,
        pdf_folder: Optional[str] = None,
        method: str = "hybrid",
        embedder_type: str = "minilm",
        mode: str = "paper",
        force_refresh: bool = False,
        chunking_strategy: str = "paragraph",
    ) -> "InCiteAgent":
        """Initialize from Paperpile library.

        Uses BibTeX sync URL or local .bib file for metadata, and optionally
        matches PDFs from a Google Drive folder. If args are None, loads
        configuration from ~/.incite/config.json.

        Args:
            bibtex_url: Paperpile BibTeX auto-sync URL
            bibtex_path: Path to local .bib file (alternative to URL)
            pdf_folder: Path to Paperpile's Google Drive PDF folder
            method: Retrieval method ("neural", "bm25", "hybrid")
            embedder_type: Embedder to use
            mode: "paper" or "paragraph"
            force_refresh: If True, re-fetch BibTeX ignoring cache
            chunking_strategy: Chunking strategy for paragraph mode

        Returns:
            Configured InCiteAgent instance
        """
        from incite.corpus.paperpile_source import PaperpileSource
        from incite.webapp.state import (
            get_cache_dir,
            get_config,
            get_paragraph_retriever,
            get_retriever,
            has_chunks,
            load_zotero_chunks,
        )

        # Load from config if no args provided
        if not bibtex_url and not bibtex_path:
            config = get_config()
            pp_config = config.get("paperpile", {})
            bibtex_url = pp_config.get("bibtex_url") or None
            bibtex_path = pp_config.get("bibtex_path") or None
            pdf_folder = pdf_folder or pp_config.get("pdf_folder") or None

        source = PaperpileSource(
            bibtex_url=bibtex_url,
            bibtex_path=Path(bibtex_path) if bibtex_path else None,
            pdf_folder=Path(pdf_folder) if pdf_folder else None,
        )

        # Load papers (PaperpileSource handles its own caching)
        papers = source.load_papers()
        paper_dict = {p.id: p for p in papers}

        cache_dir = get_cache_dir()

        if mode == "paragraph":
            cached_chunks = cache_dir / f"paperpile_chunks_{chunking_strategy}.jsonl"
            if not has_chunks(papers) and not cached_chunks.exists():
                raise ValueError(
                    "No PDF text extracted. Provide --pdf-folder to match PDFs, "
                    "or use mode='paper' for title+abstract retrieval."
                )

            chunks = load_zotero_chunks(
                papers,
                chunking_strategy=chunking_strategy,
                force_rebuild=force_refresh,
            )
            retriever = get_paragraph_retriever(
                chunks=chunks,
                papers=papers,
                method=method if method != "bm25" else "hybrid",
                embedder_type=embedder_type,
                force_rebuild=force_refresh,
            )
            return cls(
                retriever=retriever,
                papers=paper_dict,
                method=method,
                embedder_type=embedder_type,
                mode="paragraph",
                chunks=chunks,
                chunking_strategy=chunking_strategy,
            )
        else:
            retriever = get_retriever(
                papers=papers,
                method=method,
                embedder_type=embedder_type,
                force_rebuild=force_refresh,
            )

            # Try to load evidence store and upgrade to two-stage retrieval
            evidence_store = None
            evidence_chunks = None
            two_stage = False
            try:
                cached_chunks_path = cache_dir / f"paperpile_chunks_{chunking_strategy}.jsonl"
                if has_chunks(papers) or cached_chunks_path.exists():
                    from incite.webapp.state import get_evidence_store

                    evidence_chunks, evidence_store = get_evidence_store(
                        papers=papers,
                        embedder_type=embedder_type,
                        chunking_strategy=chunking_strategy,
                        force_rebuild=force_refresh,
                    )
                    logger.info(
                        "Evidence store loaded: %d chunks (%s)",
                        len(evidence_chunks),
                        chunking_strategy,
                    )

                    if evidence_store is not None and evidence_chunks is not None:
                        from incite.retrieval.two_stage import TwoStageRetriever

                        embedder = cls._find_embedder_from(retriever)
                        if embedder is None:
                            from incite.retrieval.factory import get_embedder

                            embedder = get_embedder(embedder_type)

                        retriever = TwoStageRetriever(
                            paper_retriever=retriever,
                            chunk_store=evidence_store,
                            chunks=evidence_chunks,
                            embedder=embedder,
                        )
                        two_stage = True
                        logger.info("Using TwoStageRetriever (alpha=0.6)")
            except Exception as e:
                logger.warning("Could not load evidence store: %s", e)

            return cls(
                retriever=retriever,
                papers=paper_dict,
                method=method,
                embedder_type=embedder_type,
                mode="paper",
                evidence_store=evidence_store,
                evidence_chunks=evidence_chunks,
                two_stage=two_stage,
            )

    @classmethod
    def from_corpus(
        cls,
        corpus_path: str,
        index_path: Optional[str] = None,
        method: str = "hybrid",
        embedder_type: str = "minilm",
        mode: str = "paper",
        chunks_path: Optional[str] = None,
        chunk_index_path: Optional[str] = None,
    ) -> "InCiteAgent":
        """Initialize from a corpus JSONL file.

        Args:
            corpus_path: Path to corpus.jsonl file
            index_path: Path to pre-built FAISS index (optional)
            method: Retrieval method ("neural", "bm25", "hybrid")
            embedder_type: Embedder to use ("minilm", "e5", "specter", "nomic", "voyage")
            mode: "paper" (title+abstract) or "paragraph" (chunks)
            chunks_path: Path to chunks.jsonl file (for paragraph mode)
            chunk_index_path: Path to chunk FAISS index (for paragraph mode)

        Returns:
            Configured InCiteAgent instance
        """
        from incite.corpus.loader import load_chunks, load_corpus
        from incite.retrieval.factory import create_paragraph_retriever, create_retriever

        papers = load_corpus(corpus_path)
        paper_dict = {p.id: p for p in papers}

        if mode == "paragraph":
            if chunks_path is None:
                raise ValueError("chunks_path required for paragraph mode")

            chunks = load_chunks(chunks_path)

            retriever = create_paragraph_retriever(
                chunks=chunks,
                papers=papers,
                embedder_type=embedder_type,
                index_path=Path(chunk_index_path) if chunk_index_path else None,
                method=method if method != "bm25" else "hybrid",
                show_progress=True,
            )
            return cls(
                retriever=retriever,
                papers=paper_dict,
                method=method,
                embedder_type=embedder_type,
                mode="paragraph",
                chunks=chunks,
            )
        else:
            retriever = create_retriever(
                papers=papers,
                method=method,
                embedder_type=embedder_type,
                index_path=Path(index_path) if index_path else None,
                show_progress=True,
            )
            return cls(
                retriever=retriever,
                papers=paper_dict,
                method=method,
                embedder_type=embedder_type,
                mode="paper",
            )

    @classmethod
    def from_folder(
        cls,
        folder_path: str | Path,
        method: str = "hybrid",
        embedder_type: str = "minilm",
        mode: str = "paper",
        chunking_strategy: str = "paragraph",
    ) -> "InCiteAgent":
        """Initialize from a folder of PDFs.

        Scans the folder recursively for PDF files, extracts metadata and
        full text, and builds a retrieval index. Results are cached in
        ~/.incite/ for fast subsequent loads.

        Args:
            folder_path: Path to a folder containing PDF files
            method: Retrieval method ("neural", "bm25", "hybrid")
            embedder_type: Embedder to use ("minilm", "e5", "specter", etc.)
            mode: "paper" (title+abstract) or "paragraph" (PDF chunks)
            chunking_strategy: Chunking strategy for paragraph mode

        Returns:
            Configured InCiteAgent instance
        """
        from incite.corpus.folder_source import FolderCorpusSource

        source = FolderCorpusSource(folder_path)
        return cls.from_source(
            source,
            method=method,
            embedder_type=embedder_type,
            mode=mode,
            chunking_strategy=chunking_strategy,
        )

    def _find_embedder(self):
        """Extract the neural embedder from the retriever chain."""
        r = self._retriever
        if hasattr(r, "embedder"):
            return r.embedder
        # TwoStageRetriever wrapping a paper retriever
        if hasattr(r, "paper_retriever"):
            return self._find_embedder_from(r.paper_retriever) or getattr(r, "embedder", None)
        # HybridRetriever wrapping (neural, bm25)
        if hasattr(r, "retrievers"):
            for sub, _ in r.retrievers:
                if hasattr(sub, "embedder"):
                    return sub.embedder
        # HybridParagraphRetriever
        if hasattr(r, "paragraph_retriever"):
            pr = r.paragraph_retriever
            if hasattr(pr, "embedder"):
                return pr.embedder
            if hasattr(pr, "paragraph_retriever") and hasattr(pr.paragraph_retriever, "embedder"):
                return pr.paragraph_retriever.embedder
        return None

    @staticmethod
    def _find_embedder_from(retriever):
        """Recursively find embedder from a retriever chain."""
        if hasattr(retriever, "embedder"):
            return retriever.embedder
        if hasattr(retriever, "retrievers"):
            for sub, _ in retriever.retrievers:
                if hasattr(sub, "embedder"):
                    return sub.embedder
        return None

    def _attach_evidence(
        self,
        results: list,
        query_embedding,
        threshold: Optional[float] = None,
        max_per_paper: int = 3,
    ) -> None:
        """Attach paragraph evidence snippets to paper-level results.

        Searches the chunk index for the query and attaches the best matching
        paragraphs to each result whose chunk score exceeds the threshold.
        This enables showing evidence even in paper mode (dual-path retrieval).

        Following OpenScholar (Asai et al., 2026), returns up to max_per_paper
        evidence snippets per paper instead of just the single best.

        Args:
            results: List of RetrievalResult to decorate (modified in place)
            query_embedding: Pre-computed query embedding vector
            threshold: Minimum chunk score to show evidence (default: EVIDENCE_THRESHOLD)
            max_per_paper: Maximum evidence snippets per paper (default: 3)
        """
        if self._evidence_store is None or self._evidence_chunks is None:
            return

        if threshold is None:
            threshold = self.EVIDENCE_THRESHOLD

        from incite.retrieval.paragraph import _highlight_sentence_in_parent

        # Search chunk store — get enough chunks to cover result papers
        n_chunks = max(100, len(results) * 10)
        chunk_results = self._evidence_store.search_with_papers(query_embedding, k=n_chunks)

        # Collect all chunks per paper above threshold
        paper_chunks: dict[str, list[tuple[str, float]]] = {}
        for chunk_id, paper_id, score in chunk_results:
            if score >= threshold and chunk_id in self._evidence_chunks:
                if paper_id not in paper_chunks:
                    paper_chunks[paper_id] = []
                paper_chunks[paper_id].append((chunk_id, score))

        # Attach evidence to results
        for result in results:
            if result.paper_id in paper_chunks:
                # Sort by score descending, cap at max_per_paper
                chunks_for_paper = sorted(
                    paper_chunks[result.paper_id], key=lambda x: x[1], reverse=True
                )[:max_per_paper]

                # Build matched_paragraphs list
                paragraphs = []
                for chunk_id, score in chunks_for_paper:
                    chunk = self._evidence_chunks[chunk_id]
                    text = _highlight_sentence_in_parent(chunk)
                    paragraphs.append(
                        {
                            "text": text,
                            "score": score,
                            "section": chunk.section,
                            "page": chunk.page_number,
                        }
                    )
                result.matched_paragraphs = paragraphs

                # Backward compat: matched_paragraph = best snippet
                if paragraphs:
                    result.matched_paragraph = paragraphs[0]["text"]
                    result.score_breakdown["best_chunk_score"] = paragraphs[0]["score"]
                    result.score_breakdown["num_chunks_matched"] = len(paragraphs)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences, matching the Obsidian plugin's logic."""
        placeholder = "\x00"
        processed = InCiteAgent._ABBREV_RE.sub(
            lambda m: re.sub(r"\.\s*", placeholder, m.group()), text
        )
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"])', processed)
        return [s.replace(placeholder, ". ").strip() for s in parts if s.strip()]


    def _find_neural_store(self) -> Optional["FAISSStore"]:
        """Extract the neural FAISSStore from the retriever chain."""
        r = self._retriever
        if hasattr(r, "store"):
            return r.store
        if hasattr(r, "paper_retriever"):
            pr = r.paper_retriever
            if hasattr(pr, "store"):
                return pr.store
            if hasattr(pr, "retrievers"):
                for sub, _ in pr.retrievers:
                    if hasattr(sub, "store"):
                        return sub.store
        if hasattr(r, "retrievers"):
            for sub, _ in r.retrievers:
                if hasattr(sub, "store"):
                    return sub.store
        return None

    def find_similar(self, paper_id: str, k: int = 5) -> AgentResponse:
        """Find papers similar to an existing paper using its stored embedding.

        Looks up the paper's vector in the store — supports both FAISS (via
        index reconstruct) and pgvector (via get_embedding method).
        """
        import numpy as np

        # Prefer explicit paper store (set for paragraph mode), fall back to retriever chain
        store = self._paper_store or self._find_neural_store()
        if store is None:
            return AgentResponse(
                query=f"similar:{paper_id}",
                recommendations=[],
                timing=TimingInfo(total_ms=0),
                corpus_size=len(self._papers),
                method=self._method,
                embedder=self._embedder_type,
                mode=self._mode,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # Get the paper's embedding: pgvector path or FAISS path
        query_embedding = None
        if hasattr(store, "get_embedding"):
            query_embedding = store.get_embedding(paper_id)
        elif hasattr(store, "_index") and store._index is not None:
            idx = store._id_to_idx.get(paper_id)
            if idx is not None:
                query_embedding = store._index.reconstruct(int(idx))

        if query_embedding is None:
            return AgentResponse(
                query=f"similar:{paper_id}",
                recommendations=[],
                timing=TimingInfo(total_ms=0),
                corpus_size=len(self._papers),
                method=self._method,
                embedder=self._embedder_type,
                mode=self._mode,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        start_time = time.perf_counter()

        # Search for k+1 to account for self-match, then filter
        raw_results = store.search(query_embedding, k=k + 1)
        filtered = [(pid, score) for pid, score in raw_results if pid != paper_id][:k]

        recommendations = []
        for rank, (pid, score) in enumerate(filtered):
            paper = self._papers.get(pid)
            if paper:
                abstract_preview = None
                if paper.abstract:
                    abstract_preview = paper.abstract[:300]
                    if len(paper.abstract) > 300:
                        abstract_preview += "..."
                recommendations.append(AgentRecommendation(
                    paper_id=pid,
                    rank=rank + 1,
                    score=score,
                    title=paper.title,
                    authors=paper.authors,
                    year=paper.year,
                    abstract=abstract_preview,
                    bibtex_key=paper.bibtex_key,
                    confidence=min(1.0, max(0.0, score)),
                ))

        total_ms = (time.perf_counter() - start_time) * 1000

        return AgentResponse(
            query=f"similar:{paper_id}",
            recommendations=recommendations,
            timing=TimingInfo(
                total_ms=total_ms,
                vector_search_ms=total_ms,
            ),
            corpus_size=len(self._papers),
            method=self._method,
            embedder=self._embedder_type,
            mode=self._mode,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def recommend(
        self,
        query: str,
        k: int = 10,
        author_boost: float = 1.0,
        cursor_sentence_index: Optional[int] = None,
        focus_decay: float = 0.5,
    ) -> AgentResponse:
        """Get recommendations for a query with timing information.

        Args:
            query: The citation context or search text
            k: Number of recommendations to return
            author_boost: Boost papers whose authors appear in query (1.0 = no boost)
            cursor_sentence_index: If provided, weight the embedding toward this
                sentence position (exponential decay from cursor). Sentences further
                from the cursor contribute less to the neural embedding.
            focus_decay: Decay rate per sentence of distance (0.5 = ±1 sentence
                gets half the weight of the cursor sentence)

        Returns:
            AgentResponse with recommendations and timing breakdown
        """
        start_time = time.perf_counter()

        # Cursor-weighted embedding: default to last sentence when not specified
        extra_kwargs = {}
        effective_cursor = cursor_sentence_index
        if effective_cursor is None and self._embedder is not None and 0 < focus_decay < 1.0:
            sentences = self._split_sentences(query)
            if len(sentences) > 1:
                effective_cursor = len(sentences) - 1

        if effective_cursor is not None and self._embedder is not None and 0 < focus_decay < 1.0:
            sentences = self._split_sentences(query)
            if len(sentences) > 1 and 0 <= effective_cursor < len(sentences):
                weighted_emb = self._embedder.embed_query_weighted(
                    sentences, effective_cursor, decay=focus_decay
                )
                extra_kwargs["query_embedding"] = weighted_emb

        # Call retriever with timing and deduplication
        results, timing = self._retriever.retrieve(
            query,
            k=k,
            papers=self._papers,
            author_boost=author_boost,
            return_timing=True,
            deduplicate=True,
            **extra_kwargs,
        )

        # Attach paragraph evidence to paper-mode results.
        # Skip when using TwoStageRetriever — evidence is already attached inline.
        if not self._two_stage and self._evidence_store is not None and self._mode == "paper":
            query_embedding = extra_kwargs.get("query_embedding")
            if query_embedding is None and self._embedder is not None:
                query_embedding = self._embedder.embed_query(query)
            if query_embedding is not None:
                evidence_start = time.perf_counter()
                self._attach_evidence(results, query_embedding)
                timing["evidence_ms"] = (time.perf_counter() - evidence_start) * 1000

        total_ms = (time.perf_counter() - start_time) * 1000

        # Build recommendations with full metadata
        recommendations = []
        for result in results:
            paper = self._papers.get(result.paper_id)
            if paper:
                abstract_preview = None
                if paper.abstract:
                    abstract_preview = paper.abstract[:300]
                    if len(paper.abstract) > 300:
                        abstract_preview += "..."

                rec = AgentRecommendation(
                    paper_id=result.paper_id,
                    rank=result.rank,
                    score=result.score,
                    title=paper.title,
                    authors=paper.authors,
                    year=paper.year,
                    abstract=abstract_preview,
                    bibtex_key=paper.bibtex_key,
                    score_breakdown=result.score_breakdown,
                    matched_paragraph=result.matched_paragraph,
                    matched_paragraphs=result.matched_paragraphs,
                    zotero_uri=paper.zotero_uri,
                    confidence=result.confidence,
                )
                recommendations.append(rec)

        # Build timing info
        timing_info = TimingInfo(
            total_ms=total_ms,
            embed_query_ms=timing.get("embed_query_ms", 0.0),
            vector_search_ms=timing.get("vector_search_ms", 0.0),
            bm25_search_ms=timing.get("bm25_search_ms"),
            fusion_ms=timing.get("fusion_ms"),
            evidence_ms=timing.get("evidence_ms"),
        )

        response = AgentResponse(
            query=query,
            recommendations=recommendations,
            timing=timing_info,
            corpus_size=len(self._papers),
            method=self._method,
            embedder=self._embedder_type,
            mode=self._mode,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Save last recommendations for `incite acquire --from-last-recommendations`
        self._save_last_recommendations(response)

        return response

    def _save_last_recommendations(self, response: AgentResponse) -> None:
        """Save the most recent recommendations to ~/.incite/last_recommendations.json.

        This allows `incite acquire --from-last-recommendations` to acquire
        PDFs for papers that were just recommended.
        """
        try:
            cache_dir = Path.home() / ".incite"
            cache_dir.mkdir(parents=True, exist_ok=True)
            out_path = cache_dir / "last_recommendations.json"

            entries = []
            for rec in response.recommendations:
                paper = self._papers.get(rec.paper_id)
                entry = {
                    "paper_id": rec.paper_id,
                    "title": rec.title,
                }
                if paper and paper.doi:
                    entry["doi"] = paper.doi
                entries.append(entry)

            data = {
                "query": response.query[:200],
                "timestamp": response.timestamp,
                "recommendations": entries,
            }
            out_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass  # Best-effort; don't break recommend() on save failure

    def batch_recommend(
        self,
        queries: list[str],
        k: int = 10,
        parallel: bool = True,
        max_workers: int = 4,
        author_boost: float = 1.0,
    ) -> list[AgentResponse]:
        """Get recommendations for multiple queries.

        Args:
            queries: List of query texts
            k: Number of recommendations per query
            parallel: If True, run queries in parallel
            max_workers: Number of parallel workers (if parallel=True)
            author_boost: Boost papers whose authors appear in query

        Returns:
            List of AgentResponse objects in same order as queries
        """
        if not queries:
            return []

        if not parallel or len(queries) == 1:
            return [self.recommend(q, k=k, author_boost=author_boost) for q in queries]

        # Run first query sequentially to warm up the model
        # (avoids race condition when multiple threads try to lazy-load)
        results = [None] * len(queries)
        results[0] = self.recommend(queries[0], k=k, author_boost=author_boost)

        if len(queries) == 1:
            return results

        # Parallel execution for remaining queries (model now loaded)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.recommend, q, k, author_boost): i
                for i, q in enumerate(queries[1:], start=1)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return results

    def get_stats(self) -> dict:
        """Get corpus and configuration statistics.

        Returns:
            Dict with corpus_size, method, embedder, mode, etc.
        """
        stats = {
            "corpus_size": len(self._papers),
            "method": self._method,
            "embedder": self._embedder_type,
            "mode": self._mode,
        }

        if self._mode == "paragraph":
            stats["num_chunks"] = len(self._chunks)
            stats["chunking_strategy"] = self._chunking_strategy

        # Count papers with various metadata
        with_abstract = sum(1 for p in self._papers.values() if p.abstract)
        with_year = sum(1 for p in self._papers.values() if p.year)
        with_authors = sum(1 for p in self._papers.values() if p.authors)
        with_doi = sum(1 for p in self._papers.values() if p.doi)
        with_full_text = sum(1 for p in self._papers.values() if p.full_text)

        stats["papers_with_abstract"] = with_abstract
        stats["papers_with_year"] = with_year
        stats["papers_with_authors"] = with_authors
        stats["papers_with_doi"] = with_doi
        stats["papers_with_full_text"] = with_full_text

        return stats

    def extract_pdfs(
        self,
        max_workers: int = 8,
        progress_callback=None,
    ) -> dict:
        """Extract text from PDFs for paragraph-level retrieval.

        This must be called before using mode='paragraph'. The extraction
        results are cached in ~/.incite/.

        Args:
            max_workers: Number of parallel workers for PDF extraction
            progress_callback: Optional callback(current, total, message)

        Returns:
            Stats dict with extracted count, total, etc.
        """
        from incite.webapp.state import extract_and_save_pdfs

        # Get papers list from dict
        papers = list(self._papers.values())

        stats = extract_and_save_pdfs(
            papers=papers,
            progress_callback=progress_callback,
            max_workers=max_workers,
        )

        return stats
