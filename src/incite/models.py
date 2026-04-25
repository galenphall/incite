"""Core data models for the inCite citation recommendation system.

This is the central model hub. All modules depend on these definitions.

Key classes:
    Paper: Academic paper with metadata, abstract, and optional full text.
    Chunk: Paragraph-level text segment with section context.
    CitationContext: Multi-scale citation context for evaluation queries.
    RetrievalResult: Single paper/chunk result with score and rank.

Canonical embedding text formatters (MUST live here per project contract):
    format_author_string: Author list → display string.
    format_paper_embedding_text: Paper → embedding input text.
    format_paper_metadata_prefix: Paper → metadata-only prefix.
    format_passage_embedding_text: Chunk → embedding input text.
    strip_metadata_prefix: Inverse of format_paper_embedding_text.
    clean_citation_markers: Remove [CITE] and similar markers.

See also:
    incite.eval_models: EvaluationResult, QueryResult (re-exported here).
    docs/code/deep/embedding-text-format.md: Format specification.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

# Pattern matches metadata prefix: "Title. Author(s). YYYY. [optional Journal.] "
# Examples:
#   "Ground-state cooling of mechanical resonators. Martin et al.. 2004. We propose..."
#   "Title. Smith and Jones. 2023. Nature. Abstract text..."
#   "Title. Smith. 2023. Abstract text..."
_METADATA_PREFIX_PATTERN = re.compile(
    r"^.+?\.\s+"  # Title ending with ". "
    r"(?:[A-Z][a-z]+"  # First author last name
    r"(?:\s+(?:et al\.|and\s+[A-Z][a-z]+))?\.?\s+)?"  # Optional "et al." or "and Name"
    r"\d{4}\.\s*"  # Year (4 digits + ". ")
    r"(?:[A-Z][A-Za-z\s&:,]+?\.\s+)?"  # Optional journal name
)


def strip_metadata_prefix(text: str) -> str:
    """Strip 'title. authors. year. [journal.]' prefix, leaving abstract/passage text.

    Inverse of format_paper_embedding_text(): extracts the core content (abstract
    or passage text) from a string that may or may not have a metadata prefix.

    Used for format-aware deduplication — comparing papers that appear in different
    formats (with vs without metadata prefix) across training data sources.

    Only strips if substantial text remains after the prefix (>50 chars),
    to avoid stripping text that just happens to match the pattern.
    """
    match = _METADATA_PREFIX_PATTERN.match(text)
    if match:
        remainder = text[match.end() :]
        if len(remainder) > 50:
            return remainder
    return text


# Patterns for cleaning citation markers from query text
_CITE_PATTERNS = [
    re.compile(r"\[CITE\]"),  # [CITE] marker
    re.compile(r"\{\{cite:[a-f0-9]*\}?\}?"),  # {{cite:hash}} complete or truncated
    re.compile(r"\{\{formula:[a-f0-9\-]*\}?\}?"),  # {{formula:uuid}} complete or truncated
    re.compile(r"\((?:e\.g\.|i\.e\.|cf\.)?\s*,?\s*\)"),  # Parentheses with only abbrevs/commas
    re.compile(r"\(\s*\)"),  # Empty parentheses
    re.compile(r",\s*,"),  # Double commas
    re.compile(r",\s*\."),  # Comma followed by period
    re.compile(r"\s+"),  # Normalize whitespace
]


def clean_citation_markers(text: str) -> str:
    """Remove citation markers and clean up resulting text.

    Removes:
    - [CITE] markers
    - {{cite:hexhash}} patterns
    - Empty parentheses and double commas left behind
    - Normalizes whitespace
    """
    result = text
    for pattern in _CITE_PATTERNS[:-1]:  # All but whitespace
        result = pattern.sub(" ", result)
    # Normalize whitespace last
    result = _CITE_PATTERNS[-1].sub(" ", result)
    return result.strip()


def format_author_string(author_lastnames: list[str]) -> str:
    """Canonical author formatting for embedding text.

    Args:
        author_lastnames: List of author last names.

    Returns:
        Formatted author string: "" (0), "Smith" (1),
        "Smith and Jones" (2), "Smith et al." (3+).
    """
    if not author_lastnames:
        return ""
    if len(author_lastnames) == 1:
        return author_lastnames[0]
    if len(author_lastnames) == 2:
        return f"{author_lastnames[0]} and {author_lastnames[1]}"
    return f"{author_lastnames[0]} et al."


def format_paper_embedding_text(
    title: str,
    abstract: str = "",
    author_lastnames: list[str] | None = None,
    year: int | None = None,
    journal: str | None = None,
    llm_description: str | None = None,
    include_abstract: bool = True,
    include_metadata: bool = True,
) -> str:
    """Canonical paper embedding text format — single source of truth.

    All paths that format a paper for embedding (retrieval indexing, training
    data generation, evaluation) MUST delegate to this function.

    Format: ". ".join([title, authors, year, journal, abstract, llm_description])
    with empty/disabled parts skipped.

    Args:
        title: Paper title (required).
        abstract: Paper abstract text.
        author_lastnames: List of author last names.
        year: Publication year.
        journal: Journal/venue name.
        llm_description: LLM-generated enriched description.
        include_abstract: Whether to include the abstract.
        include_metadata: Whether to include authors/year/journal.

    Returns:
        Formatted text for embedding.
    """
    parts = [title]

    if include_metadata:
        author_str = format_author_string(author_lastnames or [])
        if author_str:
            parts.append(author_str)
        if year:
            parts.append(str(year))
        if journal:
            parts.append(journal)

    if include_abstract and abstract:
        parts.append(abstract)

    if llm_description:
        parts.append(llm_description)

    return ". ".join(parts)


def format_paper_metadata_prefix(
    title: str,
    author_lastnames: list[str] | None = None,
    year: int | None = None,
    journal: str | None = None,
) -> str:
    """Canonical paper metadata prefix for chunk embedding context.

    Used as `context_text` on Chunk objects to give the retriever
    paper-identity signal at the chunk level.

    Returns:
        e.g. "Title. Smith and Jones. 2023. Nature"
    """
    return format_paper_embedding_text(
        title=title,
        author_lastnames=author_lastnames,
        year=year,
        journal=journal,
        include_abstract=False,
        include_metadata=True,
    )


def format_passage_embedding_text(
    chunk_text: str,
    metadata_prefix: str | None = None,
) -> str:
    """Canonical passage embedding text format — single source of truth.

    All paths that format a chunk/passage for embedding (retrieval indexing,
    training data generation) MUST delegate to this function.

    Args:
        chunk_text: Raw chunk/passage text.
        metadata_prefix: Paper metadata prefix (from format_paper_metadata_prefix).

    Returns:
        Formatted text: "{prefix}\\n\\n{chunk_text}" or just chunk_text.
    """
    if metadata_prefix:
        return f"{metadata_prefix}\n\n{chunk_text}"
    return chunk_text


@dataclass
class Chunk:
    """A chunk of text from a paper for paragraph-level retrieval.

    Chunks are created by splitting paper full text into paragraphs.
    Each chunk can optionally have LLM-generated context prepended
    for better retrieval (Anthropic contextual retrieval approach).
    """

    id: str  # Format: "{paper_id}::chunk_{idx}"
    paper_id: str  # Parent paper reference
    text: str  # Raw chunk text
    section: Optional[str] = None  # Section heading this chunk belongs to
    char_offset: int = 0  # Position in full text (for reconstruction)
    page_number: Optional[int] = None  # 1-indexed page number in source PDF
    source: Optional[str] = None  # Extraction method: "html", "grobid", "abstract", "pymupdf"
    context_text: Optional[str] = None
    # Paper metadata prefix for embedding context (e.g., "Title. Authors. 2023. Nature").
    # Set during chunking via format_paper_metadata_prefix(). Prepended to chunk text
    # when embedding via to_embedding_text(). Originally designed for LLM-generated
    # contextual enrichment but currently used exclusively as a metadata prefix.
    parent_text: Optional[str] = None  # Parent paragraph (for display, NOT embedding)

    def __post_init__(self):
        if not self.id:
            raise ValueError("Chunk must have an id")
        if not self.paper_id:
            raise ValueError("Chunk must have a paper_id")
        if not self.text:
            raise ValueError("Chunk must have text")

    @classmethod
    def parse_id(cls, chunk_id: str) -> tuple[str, int]:
        """Parse a chunk ID into (paper_id, chunk_index).

        Args:
            chunk_id: Chunk ID in format "{paper_id}::chunk_{idx}"

        Returns:
            Tuple of (paper_id, chunk_index)

        Raises:
            ValueError: If chunk_id is not in expected format
        """
        if "::chunk_" not in chunk_id:
            raise ValueError(f"Invalid chunk ID format: {chunk_id}")
        parts = chunk_id.rsplit("::chunk_", 1)
        return parts[0], int(parts[1])

    def to_embedding_text(self) -> str:
        """Get text for embedding.

        Delegates to format_passage_embedding_text() for consistent formatting.
        If context_text is set (from LLM enrichment or metadata prefix),
        prepends it to the chunk.

        Returns:
            Text ready for embedding
        """
        return format_passage_embedding_text(self.text, self.context_text)

    def to_bm25_text(self) -> str:
        """Get text for BM25 indexing.

        Per Anthropic's recommendation, use contextual text for both
        embedding AND BM25 to maximize retrieval improvement.

        Returns:
            Text ready for BM25 tokenization
        """
        return self.to_embedding_text()


class ReferenceItem(Protocol):
    """Protocol for any reference item type (paper, case law, statute, etc.)."""

    id: str
    title: str
    source_type: str
    full_text: Optional[str]

    def to_embedding_text(self, **kwargs) -> str: ...

    def to_display_dict(self) -> dict[str, Any]: ...


@dataclass
class Paper:
    """Represents a paper in the corpus."""

    id: str
    title: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    bibtex_key: Optional[str] = None
    journal: Optional[str] = None
    full_text: Optional[str] = None
    paragraphs: list[str] = field(default_factory=list)
    source_file: Optional[str] = None
    llm_description: Optional[str] = None  # LLM-generated enriched description
    zotero_uri: Optional[str] = None  # e.g. zotero://select/items/0_KEY
    pdf_url: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            raise ValueError("Paper must have an id")
        if not self.title:
            raise ValueError("Paper must have a title")

    @property
    def has_full_text(self) -> bool:
        return self.full_text is not None and len(self.full_text) > 0

    @property
    def author_lastnames(self) -> list[str]:
        """Extract last names from author list for matching."""
        lastnames = []
        for author in self.authors:
            if "," in author:
                name = author.split(",")[0].strip()
            else:
                parts = author.split()
                name = parts[-1] if parts else ""
            # Skip single-char names (initials like "J" or "Li" abbreviations)
            if len(name) >= 2:
                lastnames.append(name)
        return lastnames

    def to_embedding_text(
        self,
        include_abstract: bool = True,
        include_metadata: bool = True,
    ) -> str:
        """Format paper for embedding as a single vector.

        This produces the paper-level embedding: title + authors + year + journal + abstract.
        This is effectively the "abstract embedding" — the paper's identity for discovery.
        Chunk-level embeddings (from Chunk.to_embedding_text) cover full-text passages.

        Delegates to format_paper_embedding_text() for consistent formatting.

        Args:
            include_abstract: Include the abstract in embedding text.
            include_metadata: Include authors, year, and journal metadata.

        Returns:
            Formatted text for embedding. With metadata enabled, format is:
            "{title}. {authors}. {year}. {journal}. {abstract}"
        """
        return format_paper_embedding_text(
            title=self.title,
            abstract=self.abstract,
            author_lastnames=self.author_lastnames,
            year=self.year,
            journal=self.journal,
            llm_description=self.llm_description,
            include_abstract=include_abstract,
            include_metadata=include_metadata,
        )

    @property
    def source_type(self) -> str:
        """Item type for ReferenceItem protocol."""
        return "paper"

    @property
    def date(self) -> Optional[str]:
        """Date string for ReferenceItem protocol."""
        return str(self.year) if self.year else None

    def to_display_dict(self) -> dict[str, Any]:
        """Convert to display-friendly dict for UI rendering."""
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "journal": self.journal,
            "abstract": self.abstract,
            "doi": self.doi,
            "source_type": self.source_type,
            "date": self.date,
            "bibtex_key": self.bibtex_key,
            "has_full_text": self.has_full_text,
        }


@dataclass
class CitationContext:
    """A citation context for retrieval or evaluation."""

    id: str
    local_context: str  # 1-3 sentences around citation
    narrow_context: str = ""  # 2 sentences before + citing sentence (3 total)
    broad_context: str = ""  # 5 sentences before + citing sentence (6 total)
    section_context: str = ""  # Section heading + surrounding text
    global_context: str = ""  # Title + abstract of source paper
    source_paper_id: Optional[str] = None  # Paper this context came from
    ground_truth_ids: list[str] = field(default_factory=list)  # For evaluation
    # All papers cited by source (search space)
    reference_set_ids: list[str] = field(default_factory=list)
    mentioned_authors: list[str] = field(default_factory=list)
    mentioned_years: list[int] = field(default_factory=list)
    reformulated_query: str = ""  # LLM-generated hypothetical paper description (HyDE)
    difficulty: str = ""  # "", "standard", or "moderate"

    def __post_init__(self):
        if not self.id:
            raise ValueError("CitationContext must have an id")
        if not self.local_context:
            raise ValueError("CitationContext must have local_context")

    def get_query(
        self,
        scale: str = "local",
        clean: bool = True,
        prefix_section: bool = False,
    ) -> str:
        """Get query text at specified scale.

        Args:
            scale: Context scale - "local", "narrow", "broad", "section",
                "global", or "reformulated"
            clean: If True, remove citation markers ([CITE], {{cite:...}})
            prefix_section: If True, prepend the section heading to the query
                regardless of scale. E.g., "Related Work: <context>". This gives
                the embedder structural signal about the citation's role.

        Returns:
            Query text, optionally cleaned of citation markers.
        """
        # Reformulated queries are already clean LLM output -- return directly
        if scale == "reformulated":
            if self.reformulated_query:
                return self.reformulated_query
            # Fall back to narrow context if no reformulated query
            text = self.narrow_context if self.narrow_context else self.local_context
            if clean:
                text = clean_citation_markers(text)
            return text

        if scale == "local":
            text = self.local_context
        elif scale == "narrow":
            # 3 sentences: 2 before + citing sentence
            text = self.narrow_context if self.narrow_context else self.local_context
        elif scale == "broad":
            # 6 sentences: 5 before + citing sentence
            text = self.broad_context if self.broad_context else self.local_context
        elif scale == "section":
            if self.section_context:
                text = f"{self.section_context} {self.local_context}"
            else:
                text = self.local_context
        elif scale == "global":
            parts = []
            if self.global_context:
                parts.append(self.global_context)
            if self.section_context:
                parts.append(self.section_context)
            parts.append(self.local_context)
            text = " ".join(parts)
        else:
            raise ValueError(f"Unknown scale: {scale}")

        # Optionally prepend section heading (unless scale already includes it)
        if prefix_section and self.section_context and scale not in ("section", "global"):
            text = f"{self.section_context} {text}"

        if clean:
            text = clean_citation_markers(text)

        return text


@dataclass
class RetrievalResult:
    """A single retrieval result with scores."""

    paper_id: str
    score: float
    rank: int = 0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    matched_paragraph: Optional[str] = None
    matched_paragraphs: list[dict] = field(default_factory=list)
    # Each entry: {"text": str, "score": float, "section": Optional[str], "page": Optional[int]}
    display_mode: str = "paper"  # "paper", "paragraph", or "paper_with_summary"
    confidence: float = 0.0  # Neural similarity confidence in [0, 1]

    def __lt__(self, other: "RetrievalResult") -> bool:
        return self.score > other.score  # Higher score = better rank

    def get_display_mode(self, para_threshold: float = 0.65, multi_chunk_threshold: int = 3) -> str:
        """Determine how to display this result based on score characteristics.

        Args:
            para_threshold: Minimum best_chunk_score to show paragraph (default 0.65)
            multi_chunk_threshold: Minimum chunks to trigger paper_with_summary (default 3)

        Returns:
            "paragraph": Show matched paragraph text (high chunk score)
            "paper_with_summary": Show paper + note about multiple matches
            "paper": Show paper title/abstract only (default)
        """
        if self.matched_paragraph is None:
            return "paper"

        best_chunk_score = self.score_breakdown.get("best_chunk_score", 0)
        num_chunks = self.score_breakdown.get("num_chunks_matched", 1)

        # High-confidence paragraph match
        if best_chunk_score >= para_threshold:
            return "paragraph"

        # Multiple chunks matched - paper is broadly relevant
        if num_chunks >= multi_chunk_threshold:
            return "paper_with_summary"

        return "paper"


# --- Re-exports for backward compatibility ---
from incite.eval_models import (  # noqa: F401, E402
    EvaluationResult,
    QueryResult,
    _bootstrap_ci,
)
