"""Data models for paper discovery."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DiscoveryCandidate:
    """A paper discovered through citation/semantic expansion, not yet in the library."""

    s2_id: str
    title: str
    authors: list[str]
    year: int | None
    doi: str | None
    abstract: str

    # Signal accumulators
    citation_overlap: int = 0
    citing_library_ids: list[str] = field(default_factory=list)
    bib_coupling_score: float = 0.0
    bib_coupling_refs: int = 0
    semantic_score: float = 0.0
    semantic_source_title: str = ""
    author_overlap: int = 0
    overlapping_authors: list[str] = field(default_factory=list)
    pagerank_score: float = 0.0
    cocitation_score: float = 0.0

    # Metadata for downstream use
    pdf_url: str | None = None
    venue: str | None = None

    @property
    def signal_count(self) -> int:
        """Number of distinct discovery signals that fired."""
        return (
            (1 if self.citation_overlap > 0 else 0)
            + (1 if self.bib_coupling_score > 0.1 else 0)
            + (1 if self.semantic_score > 0.4 else 0)
            + (1 if self.author_overlap > 0 else 0)
            + (1 if self.pagerank_score > 0.1 else 0)
            + (1 if self.cocitation_score > 0.1 else 0)
        )

    @property
    def discovery_score(self) -> float:
        """Composite score blending all signals."""
        capped_semantic = min(self.semantic_score, 1.0) * 0.7
        return (
            0.20 * min(self.citation_overlap / 3, 1.0)
            + 0.25 * self.bib_coupling_score
            + 0.20 * capped_semantic
            + 0.10 * min(self.author_overlap / 2, 1.0)
            + 0.15 * self.pagerank_score
            + 0.10 * self.cocitation_score
        )

    def to_dict(self) -> dict:
        """Serialize for DB storage."""
        return {
            "s2_id": self.s2_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "doi": self.doi,
            "abstract": self.abstract,
            "discovery_score": self.discovery_score,
            "citation_overlap": self.citation_overlap,
            "citing_library_ids": self.citing_library_ids,
            "bib_coupling_score": self.bib_coupling_score,
            "bib_coupling_refs": self.bib_coupling_refs,
            "semantic_score": self.semantic_score,
            "semantic_source_title": self.semantic_source_title,
            "author_overlap": self.author_overlap,
            "overlapping_authors": self.overlapping_authors,
            "pagerank_score": self.pagerank_score,
            "cocitation_score": self.cocitation_score,
            "signal_count": self.signal_count,
            "pdf_url": self.pdf_url,
            "venue": self.venue,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DiscoveryCandidate:
        """Deserialize from DB storage."""
        return cls(
            s2_id=d["s2_id"],
            title=d["title"],
            authors=d.get("authors", []),
            year=d.get("year"),
            doi=d.get("doi"),
            abstract=d.get("abstract", ""),
            citation_overlap=d.get("citation_overlap", 0),
            citing_library_ids=d.get("citing_library_ids", []),
            bib_coupling_score=d.get("bib_coupling_score", 0.0),
            bib_coupling_refs=d.get("bib_coupling_refs", 0),
            semantic_score=d.get("semantic_score", 0.0),
            semantic_source_title=d.get("semantic_source_title", ""),
            author_overlap=d.get("author_overlap", 0),
            overlapping_authors=d.get("overlapping_authors", []),
            pagerank_score=d.get("pagerank_score", 0.0),
            cocitation_score=d.get("cocitation_score", 0.0),
            pdf_url=d.get("pdf_url"),
            venue=d.get("venue"),
        )
