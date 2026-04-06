"""Data models for InCiteAgent responses.

These dataclasses define the structured output format for the agent SDK.
They are serializable to JSON for use by editor plugins and the REST API.

Related modules:
    - incite.agent: InCiteAgent class that produces these responses.
    - incite.api: REST API that serializes these to HTTP responses.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class TimingInfo:
    """Timing breakdown for a retrieval operation.

    All times are in milliseconds. Optional fields are None when the
    corresponding step was not performed (e.g., bm25_search_ms is None
    for neural-only retrieval).
    """

    total_ms: float
    """Wall-clock time for the full recommend() call."""
    embed_query_ms: float = 0.0
    """Time to embed the query text."""
    vector_search_ms: float = 0.0
    """Time for FAISS/pgvector nearest-neighbor search."""
    bm25_search_ms: Optional[float] = None
    """Time for BM25 search (None for neural-only)."""
    fusion_ms: Optional[float] = None
    """Time for RRF score fusion (None for single-method retrieval)."""
    evidence_ms: Optional[float] = None
    """Time for paragraph evidence lookup (None when not performed)."""

    def to_dict(self) -> dict:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class AgentRecommendation:
    """A single recommendation with full metadata.

    Returned as elements of AgentResponse.recommendations. All optional
    fields are None when the data was not available in the corpus.
    """

    paper_id: str
    """Stable identifier for the paper (Zotero key or corpus ID)."""
    rank: int
    """1-based rank in the result list."""
    score: float
    """Retrieval score (higher is more relevant; scale depends on method)."""
    title: str
    """Paper title."""
    authors: list[str] = field(default_factory=list)
    """Author last names."""
    year: Optional[int] = None
    """Publication year."""
    abstract: Optional[str] = None
    """First 300 chars of abstract, truncated with '...' if longer."""
    bibtex_key: Optional[str] = None
    """BibTeX cite key (e.g. 'smith2023')."""
    doi: Optional[str] = None
    """DOI string without 'https://doi.org/' prefix."""
    journal: Optional[str] = None
    """Journal or venue name."""
    score_breakdown: dict[str, float] = field(default_factory=dict)
    """Per-signal scores (neural, bm25, best_chunk_score, etc.)."""
    matched_paragraph: Optional[str] = None
    """Best matching paragraph text (paragraph mode or evidence lookup)."""
    matched_paragraphs: list[dict] = field(default_factory=list)
    """Up to 3 evidence snippets: [{text, score, section, page}, ...]."""
    zotero_uri: Optional[str] = None
    """Zotero URI for opening in the desktop app (zotero://...)."""
    confidence: float = 0.0
    """Neural similarity confidence in [0, 1]."""

    def to_dict(self) -> dict:
        """Convert to dict, excluding None values."""
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class AgentResponse:
    """Complete response from a recommendation query.

    Top-level return value from InCiteAgent.recommend() and
    InCiteAgent.find_similar(). Serializable to JSON via to_dict() / to_json().
    """

    query: str
    """The original query text (or 'similar:<paper_id>' for find_similar)."""
    recommendations: list[AgentRecommendation]
    """Ranked list of recommendations."""
    timing: TimingInfo
    """Timing breakdown for the retrieval operation."""
    corpus_size: int
    """Number of papers in the indexed corpus."""
    method: str
    """Retrieval method used ('neural', 'bm25', or 'hybrid')."""
    embedder: str
    """Embedder type used ('minilm', 'e5', 'specter', etc.)."""
    timestamp: str
    """ISO 8601 UTC timestamp of the response."""
    mode: str = "paper"
    """Retrieval mode: 'paper' (title+abstract) or 'paragraph' (PDF chunks)."""

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
