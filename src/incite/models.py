"""Core data models for inCite."""

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


@dataclass
class QueryResult:
    """Per-query evaluation result for detailed analysis."""

    context_id: str
    source_paper_id: Optional[str]
    ground_truth_ids: list[str]
    scores: dict[str, float]  # metric_name -> value
    first_relevant_rank: Optional[int] = None  # 1-indexed, None if not found

    def to_dict(self) -> dict:
        return {
            "context_id": self.context_id,
            "source_paper_id": self.source_paper_id,
            "ground_truth_ids": self.ground_truth_ids,
            "scores": self.scores,
            "first_relevant_rank": self.first_relevant_rank,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QueryResult":
        return cls(
            context_id=data["context_id"],
            source_paper_id=data.get("source_paper_id"),
            ground_truth_ids=data["ground_truth_ids"],
            scores=data["scores"],
            first_relevant_rank=data.get("first_relevant_rank"),
        )


@dataclass
class EvaluationResult:
    """Aggregated evaluation metrics."""

    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    recall_at_50: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg_at_10: float = 0.0
    concordance: float = 0.0  # C-index: P(correct ranked above random incorrect)
    skill_mrr: float = 0.0  # MRR normalized to 0=random, 1=perfect
    num_queries: int = 0
    per_query: list[QueryResult] = field(default_factory=list, repr=False)
    # Evidence quality metrics (OpenScholar citation accuracy)
    evidence_precision: float = 0.0
    evidence_recall: float = 0.0
    evidence_f1: float = 0.0
    # Two-stage retrieval metrics
    evidence_coverage: float = 0.0  # Fraction of correct papers with evidence attached
    mean_best_chunk_score: float = 0.0  # Average best_chunk_score for correct papers

    def to_dict(self) -> dict[str, float]:
        # per_query intentionally excluded — backward-compatible with experiment log
        d = {
            "recall@1": self.recall_at_1,
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "recall@20": self.recall_at_20,
            "recall@50": self.recall_at_50,
            "mrr": self.mrr,
            "ndcg@10": self.ndcg_at_10,
            "num_queries": self.num_queries,
        }
        # Include corpus-adjusted metrics only when computed (backward compat)
        if self.concordance > 0 or self.skill_mrr != 0:
            d["concordance"] = self.concordance
            d["skill_mrr"] = self.skill_mrr
        # Include evidence metrics only if populated (backward compat)
        if self.evidence_precision > 0 or self.evidence_recall > 0:
            d["evidence_precision"] = self.evidence_precision
            d["evidence_recall"] = self.evidence_recall
            d["evidence_f1"] = self.evidence_f1
        # Include two-stage metrics only if populated (backward compat)
        if self.evidence_coverage > 0 or self.mean_best_chunk_score > 0:
            d["evidence_coverage"] = self.evidence_coverage
            d["mean_best_chunk_score"] = self.mean_best_chunk_score
        return d

    def _format_metric(self, name: str, value: float, ci: tuple[float, float] | None) -> str:
        if ci is not None:
            return f"  {name} {value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
        return f"  {name} {value:.3f}"

    def __str__(self) -> str:
        # Compute CIs if per-query data is available
        cis: dict[str, tuple[float, float] | None] = {}
        if self.per_query:
            from incite.models import _bootstrap_ci as bootstrap_ci

            metric_map = {
                "Recall@1:": "recall@1",
                "Recall@5:": "recall@5",
                "Recall@10:": "recall@10",
                "Recall@20:": "recall@20",
                "Recall@50:": "recall@50",
                "MRR:": "mrr",
                "NDCG@10:": "ndcg@10",
            }
            for label, key in metric_map.items():
                scores = [qr.scores.get(key, 0.0) for qr in self.per_query]
                cis[label] = bootstrap_ci(scores)

            # CIs for corpus-adjusted metrics (only if computed)
            if self.concordance > 0 or self.skill_mrr != 0:
                for label, key in [
                    ("C-index:", "concordance"),
                    ("Skill MRR:", "skill_mrr"),
                ]:
                    scores = [qr.scores.get(key, 0.0) for qr in self.per_query]
                    cis[label] = bootstrap_ci(scores)
        else:
            for label in [
                "Recall@1:",
                "Recall@5:",
                "Recall@10:",
                "Recall@20:",
                "Recall@50:",
                "MRR:",
                "NDCG@10:",
            ]:
                cis[label] = None

        lines = [
            f"Evaluation Results (n={self.num_queries}):",
            self._format_metric("Recall@1: ", self.recall_at_1, cis["Recall@1:"]),
            self._format_metric("Recall@5: ", self.recall_at_5, cis["Recall@5:"]),
            self._format_metric("Recall@10:", self.recall_at_10, cis["Recall@10:"]),
            self._format_metric("Recall@20:", self.recall_at_20, cis["Recall@20:"]),
            self._format_metric("Recall@50:", self.recall_at_50, cis["Recall@50:"]),
            self._format_metric("MRR:      ", self.mrr, cis["MRR:"]),
            self._format_metric("NDCG@10:  ", self.ndcg_at_10, cis["NDCG@10:"]),
        ]
        if self.concordance > 0 or self.skill_mrr != 0:
            lines.append("")
            lines.append("Corpus-size-adjusted (0=random, 1=perfect):")
            lines.append(self._format_metric("C-index:  ", self.concordance, cis.get("C-index:")))
            lines.append(self._format_metric("Skill MRR:", self.skill_mrr, cis.get("Skill MRR:")))
        if self.evidence_precision > 0 or self.evidence_recall > 0:
            lines.append("")
            lines.append("Evidence Quality (OpenScholar citation accuracy):")
            lines.append(f"  Evidence Precision: {self.evidence_precision:.3f}")
            lines.append(f"  Evidence Recall:    {self.evidence_recall:.3f}")
            lines.append(f"  Evidence F1:        {self.evidence_f1:.3f}")
        if self.evidence_coverage > 0 or self.mean_best_chunk_score > 0:
            lines.append("")
            lines.append("Two-stage retrieval:")
            lines.append(f"  Evidence Coverage:    {self.evidence_coverage:.1%}")
            lines.append(f"  Mean Best Chunk Score: {self.mean_best_chunk_score:.3f}")
        return "\n".join(lines)


def _bootstrap_ci(
    scores: "Sequence[float]",
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a metric."""

    import numpy as np

    scores_arr = np.array(scores, dtype=np.float64)
    if len(scores_arr) == 0:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    n = len(scores_arr)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = scores_arr[indices].mean(axis=1)

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lower, upper)
