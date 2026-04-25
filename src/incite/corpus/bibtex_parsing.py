"""BibTeX parsing and PDF-to-paper matching utilities.

Handles parsing BibTeX files into structured data, resolving DOIs via
Crossref, and matching local PDF files to their corresponding papers
using a multi-pass heuristic (exact key, DOI, title overlap, fuzzy, author+year).

Related modules:
    - incite.corpus.enrichment: Metadata enrichment using external APIs.
    - incite.corpus.paperpile_source: Uses BibTeXParser for Paperpile libraries.
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

from incite.models import Paper

logger = logging.getLogger(__name__)


class BibTeXParser:
    """Parser for BibTeX files into structured entry dicts.

    All methods are static — no instantiation required. Use
    :meth:`parse_file` or :meth:`parse_string` as entry points;
    they delegate to :meth:`_extract_entry` for field extraction.
    """

    @staticmethod
    def parse_file(path: str | Path) -> list[dict]:
        """Parse a .bib file and return list of entry dicts.

        Args:
            path: Path to .bib file

        Returns:
            List of entry dictionaries with extracted fields
        """

        with open(path, encoding="utf-8") as f:
            content = f.read()
        return BibTeXParser.parse_string(content)

    @staticmethod
    def parse_string(bibtex: str) -> list[dict]:
        """Parse a BibTeX string and return list of entry dicts.

        Args:
            bibtex: BibTeX string content

        Returns:
            List of entry dictionaries with extracted fields
        """
        import bibtexparser

        library = bibtexparser.parse_string(bibtex)

        entries = []
        for entry in library.entries:
            parsed = BibTeXParser._extract_entry(entry)
            if parsed:
                entries.append(parsed)

        return entries

    @staticmethod
    def _extract_entry(entry) -> Optional[dict]:
        """Extract relevant fields from a bibtexparser entry.

        Args:
            entry: bibtexparser Entry object

        Returns:
            Dict with extracted fields or None if entry is invalid
        """
        # Skip entries without a title
        title = entry.fields_dict.get("title")
        if not title:
            return None

        # Clean title (remove braces used for capitalization)
        title_val = BibTeXParser._clean_latex(title.value)

        # Extract authors
        authors = []
        author_field = entry.fields_dict.get("author")
        if author_field:
            authors = BibTeXParser._parse_authors(author_field.value)

        # Extract year
        year = None
        year_field = entry.fields_dict.get("year")
        if year_field:
            try:
                year = int(re.sub(r"[^\d]", "", year_field.value))
            except (ValueError, TypeError):
                pass

        # Extract DOI
        doi = None
        doi_field = entry.fields_dict.get("doi")
        if doi_field:
            doi = BibTeXParser._clean_doi(doi_field.value)

        # Fallback: check bdsk-url-1, bdsk-url-2, url for DOI URLs
        # (Paperpile exports often store DOIs in these fields)
        if not doi:
            for fallback_key in ("bdsk-url-1", "bdsk-url-2", "url"):
                fb_field = entry.fields_dict.get(fallback_key)
                if fb_field and "doi.org" in fb_field.value:
                    doi = BibTeXParser._clean_doi(fb_field.value)
                    if doi:
                        break

        # Extract abstract (if present)
        abstract = None
        abstract_field = entry.fields_dict.get("abstract")
        if abstract_field:
            abstract = BibTeXParser._clean_latex(abstract_field.value)

        # Extract journal/venue (try journal first, then booktitle for conferences)
        journal = None
        journal_field = entry.fields_dict.get("journal")
        if journal_field:
            journal = BibTeXParser._clean_latex(journal_field.value)
        elif entry.fields_dict.get("booktitle"):
            journal = BibTeXParser._clean_latex(entry.fields_dict["booktitle"].value)

        return {
            "key": entry.key,
            "title": title_val,
            "authors": authors,
            "year": year,
            "doi": doi,
            "abstract": abstract,
            "journal": journal,
            "entry_type": entry.entry_type,
        }

    @staticmethod
    def _clean_latex(text: str) -> str:
        """Remove LaTeX formatting from text.

        Args:
            text: Raw LaTeX string

        Returns:
            Cleaned plain-text string
        """
        if not text:
            return ""
        # Remove braces (used for capitalization protection)
        text = re.sub(r"[{}]", "", text)
        # Convert common LaTeX commands
        text = text.replace(r"\'", "'")
        text = text.replace(r"\"", '"')
        text = text.replace(r"\&", "&")
        text = re.sub(r"\\textit\{([^}]+)\}", r"\1", text)
        text = re.sub(r"\\textbf\{([^}]+)\}", r"\1", text)
        text = re.sub(r"\\emph\{([^}]+)\}", r"\1", text)
        # Clean up whitespace
        text = " ".join(text.split())
        return text

    @staticmethod
    def _parse_authors(author_str: str) -> list[str]:
        """Parse BibTeX author string into list of names.

        Args:
            author_str: Raw BibTeX author field value (e.g. "Smith, John and Doe, Jane")

        Returns:
            List of author name strings in "First Last" order
        """
        if not author_str:
            return []

        # Split on " and " (BibTeX author separator)
        author_str = BibTeXParser._clean_latex(author_str)
        authors = re.split(r"\s+and\s+", author_str, flags=re.IGNORECASE)

        result = []
        for author in authors:
            author = author.strip()
            if not author:
                continue
            # Handle "Last, First" format -> "First Last"
            if "," in author:
                parts = [p.strip() for p in author.split(",", 1)]
                if len(parts) == 2:
                    author = f"{parts[1]} {parts[0]}"
            result.append(author)

        return result

    @staticmethod
    def _clean_doi(doi: str) -> Optional[str]:
        """Clean and normalize a DOI string.

        Strips URL prefixes (doi.org, dx.doi.org) and surrounding whitespace.

        Args:
            doi: Raw DOI string (may include URL prefix)

        Returns:
            Normalized DOI string, or None if empty after cleaning
        """
        if not doi:
            return None
        # Remove URL prefix if present
        doi = re.sub(r"https?://doi\.org/", "", doi)
        doi = re.sub(r"https?://dx\.doi\.org/", "", doi)
        doi = doi.strip()
        return doi if doi else None


def bibtex_entries_to_papers(entries: list[dict]) -> list[Paper]:
    """Convert parsed BibTeX entry dicts to Paper objects.

    Args:
        entries: List of dicts from BibTeXParser.parse_string()

    Returns:
        List of Paper objects (entries without titles are skipped)
    """
    papers = []
    for entry in entries:
        title = entry.get("title", "").strip()
        if not title:
            continue

        # Deterministic ID from bibtex key
        key = entry.get("key", "")
        id_hash = hashlib.md5(key.encode()).hexdigest()[:12]
        paper_id = f"bib_{id_hash}"

        papers.append(
            Paper(
                id=paper_id,
                title=title,
                abstract=entry.get("abstract", ""),
                authors=entry.get("authors", []),
                year=entry.get("year"),
                doi=entry.get("doi"),
                journal=entry.get("journal"),
                bibtex_key=key,
                source_file="paperpile",
            )
        )
    return papers


def resolve_dois_via_crossref(papers: list[Paper], max_papers: int = 200) -> list[Paper]:
    """Resolve missing DOIs via CrossRef title search.

    Wraps :func:`incite.corpus.crossref.resolve_dois_batch` with graceful
    fallback when CrossRef or rapidfuzz are unavailable.

    Args:
        papers: List of Paper objects (modified in-place).
        max_papers: Maximum number of lookups per call.

    Returns:
        The same list of papers (for chaining convenience).
    """
    try:
        from incite.corpus.crossref import resolve_dois_batch
    except ImportError:
        logger.info("CrossRef client not available, skipping DOI resolution")
        return papers

    try:
        resolved = resolve_dois_batch(papers, max_papers=max_papers)
        if resolved:
            logger.info("Resolved %d DOIs via CrossRef", resolved)
    except Exception:
        logger.warning("CrossRef DOI resolution failed", exc_info=True)

    return papers


def _word_set(s: str) -> set[str]:
    """Tokenize into lowercase alphanumeric words >= 3 chars."""
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) >= 3}


def match_pdfs_to_papers(pdf_filenames: list[str], papers: list[Paper]) -> dict[str, str]:
    """Match uploaded PDF filenames to papers using heuristics.

    Tries five strategies in order:
    1. Exact bibtex_key match (key.pdf)
    2. DOI-in-filename match (e.g. "10.1038_s41558-023-01234-5.pdf")
    3. Normalized title substring in filename
    4. Fuzzy Jaccard title similarity (word-level, threshold >= 0.5)
    5. Author last name + year in filename

    Args:
        pdf_filenames: List of uploaded PDF filenames (basename only)
        papers: List of Paper objects to match against

    Returns:
        Dict mapping paper_id to matched PDF filename
    """
    matches: dict[str, str] = {}
    unmatched_files = set(pdf_filenames)

    def _normalize(s: str) -> str:
        """Lowercase, strip non-alphanumeric."""
        return re.sub(r"[^a-z0-9]", "", s.lower())

    def _normalize_doi(doi: str) -> str:
        """Normalize a DOI for filename comparison."""
        return doi.lower().replace("/", "_").replace(":", "_").replace("-", "_")

    # Pass 1: exact bibtex key match (e.g. "Smith2020.pdf" matches key "Smith2020")
    for paper in papers:
        if paper.id in matches:
            continue
        if not paper.bibtex_key:
            continue
        norm_key = _normalize(paper.bibtex_key)
        for fname in list(unmatched_files):
            stem = _normalize(Path(fname).stem)
            if stem == norm_key:
                matches[paper.id] = fname
                unmatched_files.discard(fname)
                break

    # Pass 2: DOI-in-filename match — normalize both DOI and filename stem for comparison
    for paper in papers:
        if paper.id in matches:
            continue
        if not paper.doi:
            continue
        norm_doi = _normalize_doi(paper.doi)
        if len(norm_doi) < 5:
            continue
        for fname in list(unmatched_files):
            norm_fname = Path(fname).stem.lower().replace("/", "_").replace("-", "_")
            if norm_doi in norm_fname:
                matches[paper.id] = fname
                unmatched_files.discard(fname)
                break

    # Pass 3: title substring in filename — check if first 40 chars of normalized title
    # appear in the normalized filename stem, or the stem is contained in the title
    for paper in papers:
        if paper.id in matches:
            continue
        norm_title = _normalize(paper.title)
        if len(norm_title) < 10:
            continue
        for fname in list(unmatched_files):
            norm_fname = _normalize(Path(fname).stem)
            if norm_title[:40] in norm_fname or norm_fname in norm_title:
                matches[paper.id] = fname
                unmatched_files.discard(fname)
                break

    # Pass 4: fuzzy Jaccard title similarity — match if word-level overlap >= 0.5
    # with at least 3 shared words (guards against short-title false positives)
    for paper in papers:
        if paper.id in matches:
            continue
        title_words = _word_set(paper.title)
        if len(title_words) < 3:
            continue
        for fname in list(unmatched_files):
            fname_words = _word_set(Path(fname).stem)
            if not fname_words:
                continue
            overlap = title_words & fname_words
            union = title_words | fname_words
            jaccard = len(overlap) / len(union)
            if jaccard >= 0.5 and len(overlap) >= 3:
                matches[paper.id] = fname
                unmatched_files.discard(fname)
                break

    # Pass 5: author last name + year — match if first author's last name AND
    # publication year both appear in the filename (e.g. "Smith2020_draft.pdf")
    for paper in papers:
        if paper.id in matches:
            continue
        if not paper.authors or not paper.year:
            continue
        first_author_last = _normalize(paper.authors[0].split()[-1])
        year_str = str(paper.year)
        if len(first_author_last) < 3:
            continue
        for fname in list(unmatched_files):
            norm_fname = _normalize(Path(fname).stem)
            if first_author_last in norm_fname and year_str in fname:
                matches[paper.id] = fname
                unmatched_files.discard(fname)
                break

    return matches
