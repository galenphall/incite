"""Paperpile corpus source via BibTeX sync + Google Drive PDFs.

Paperpile has no local database or API, but supports automatic BibTeX sync
via a persistent download URL and stores PDFs in Google Drive (accessible
locally via Google Drive Desktop). This module fetches metadata from the
BibTeX URL, matches PDFs from the local Google Drive folder, and produces
Paper objects for the inCite pipeline.
"""

import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional

from incite.corpus.enrichment import BibTeXParser, MetadataEnricher
from incite.corpus.loader import load_corpus, save_corpus
from incite.models import Paper

logger = logging.getLogger(__name__)


def _normalize_for_match(text: str) -> str:
    """Normalize text for fuzzy matching: lowercase, strip accents, non-alnum."""
    text = text.lower()
    # Strip accents
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Keep only alphanumeric and spaces
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_author_year_title_from_filename(filename: str) -> tuple[str, str, str]:
    """Parse Paperpile's 'Author Year - Title.pdf' naming convention.

    Args:
        filename: PDF filename (without path)

    Returns:
        (normalized_author, year_str, normalized_title_prefix)
    """
    stem = Path(filename).stem

    # Pattern: "Author(s) Year - Title" or "Author(s) Year Title"
    # Examples:
    #   "Smith 2020 - Deep Learning for NLP"
    #   "Smith and Jones 2019 - A Survey"
    #   "Smith et al. 2021 - Some Title"
    match = re.match(
        r"^(.+?)\s+((?:19|20)\d{2})\s*[-–—]\s*(.+)$",
        stem,
    )
    if not match:
        # Try without dash separator
        match = re.match(
            r"^(.+?)\s+((?:19|20)\d{2})\s+(.+)$",
            stem,
        )
    if not match:
        return ("", "", _normalize_for_match(stem)[:20])

    author_part = match.group(1).strip()
    year = match.group(2)
    title_part = match.group(3).strip()

    # Normalize author: take first last name
    author_normalized = _normalize_for_match(author_part.split()[0]) if author_part else ""

    # Normalize title: first 20 chars
    title_normalized = _normalize_for_match(title_part)[:20]

    return (author_normalized, year, title_normalized)


def find_paperpile_pdfs(pdf_folder: Path) -> dict[str, Path]:
    """Scan a Paperpile Google Drive folder for PDFs.

    Paperpile stores PDFs in Google Drive as:
        Paperpile/{A-Z}/{Author Year - Title}.pdf
    (The subfolder structure is customizable.)

    Args:
        pdf_folder: Root of the Paperpile PDF folder (e.g., ~/Google Drive/My Drive/Paperpile)

    Returns:
        Dict mapping match keys to PDF paths. Keys are "author|year|title_prefix".
    """
    pdf_index: dict[str, Path] = {}

    if not pdf_folder.exists():
        logger.warning("PDF folder does not exist: %s", pdf_folder)
        return pdf_index

    for pdf_path in sorted(pdf_folder.rglob("*.pdf")):
        author, year, title_prefix = _extract_author_year_title_from_filename(pdf_path.name)
        if author and year:
            key = f"{author}|{year}|{title_prefix}"
            pdf_index[key] = pdf_path
        else:
            # Fallback: index by normalized filename
            key = f"file|{_normalize_for_match(pdf_path.stem)[:40]}"
            pdf_index[key] = pdf_path

    logger.info("Found %d PDFs in %s", len(pdf_index), pdf_folder)
    return pdf_index


def match_paper_to_pdf(paper: Paper, pdf_index: dict[str, Path]) -> Optional[Path]:
    """Fuzzy-match a Paper to a PDF from the Paperpile folder.

    Matching strategy:
    1. First author last name + year + title prefix (15 chars)
    2. First author last name + year (no title)
    3. Title prefix only (20 chars)

    Args:
        paper: Paper to match
        pdf_index: Dict from find_paperpile_pdfs()

    Returns:
        Path to matched PDF, or None
    """
    if not pdf_index:
        return None

    # Build match key from paper metadata
    first_author = ""
    if paper.authors:
        # Get first author's last name
        author = paper.authors[0]
        if "," in author:
            first_author = author.split(",")[0].strip()
        else:
            parts = author.split()
            first_author = parts[-1] if parts else ""
    first_author_norm = _normalize_for_match(first_author)

    year_str = str(paper.year) if paper.year else ""
    title_norm = _normalize_for_match(paper.title)

    # Strategy 1: author + year + title prefix
    if first_author_norm and year_str:
        key = f"{first_author_norm}|{year_str}|{title_norm[:20]}"
        if key in pdf_index:
            return pdf_index[key]

    # Strategy 2: author + year (check all keys with matching author+year)
    if first_author_norm and year_str:
        prefix = f"{first_author_norm}|{year_str}|"
        for key, path in pdf_index.items():
            if key.startswith(prefix):
                return path

    # Strategy 3: title substring match against all indexed filenames
    if len(title_norm) >= 10:
        title_prefix = title_norm[:20]
        for key, path in pdf_index.items():
            if title_prefix in key:
                return path

    return None


def _fetch_bibtex(url: str, cache_dir: Path) -> str:
    """Fetch BibTeX from URL with ETag-based caching.

    Args:
        url: Paperpile BibTeX download URL
        cache_dir: Directory for cached files

    Returns:
        BibTeX string content
    """
    import requests

    etag_path = cache_dir / "paperpile_bibtex.etag"
    bib_path = cache_dir / "paperpile_library.bib"

    headers = {}
    if etag_path.exists():
        headers["If-None-Match"] = etag_path.read_text().strip()

    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 304:
        # Not Modified — use cached version
        logger.info("BibTeX not modified (HTTP 304), using cache")
        return bib_path.read_text(encoding="utf-8")
    resp.raise_for_status()

    bib_path.write_text(resp.text, encoding="utf-8")
    if "ETag" in resp.headers:
        etag_path.write_text(resp.headers["ETag"])
    elif etag_path.exists():
        etag_path.unlink()

    logger.info("Downloaded BibTeX (%d bytes)", len(resp.text))
    return resp.text


class PaperpileSource:
    """CorpusSource implementation for Paperpile libraries.

    Fetches metadata from a BibTeX URL (or local .bib file), enriches
    via Semantic Scholar/OpenAlex, and optionally matches PDFs from a
    local Google Drive folder.

    Satisfies the CorpusSource protocol via structural typing.
    """

    name: str = "paperpile"

    def __init__(
        self,
        bibtex_url: Optional[str] = None,
        bibtex_path: Optional[Path] = None,
        pdf_folder: Optional[Path] = None,
    ):
        """Initialize PaperpileSource.

        Args:
            bibtex_url: Paperpile BibTeX auto-sync URL
            bibtex_path: Path to a local .bib file (alternative to URL)
            pdf_folder: Path to Paperpile's Google Drive PDF folder

        Raises:
            ValueError: If neither bibtex_url nor bibtex_path is provided
        """
        if not bibtex_url and not bibtex_path:
            raise ValueError("Must provide either bibtex_url or bibtex_path")

        self._bibtex_url = bibtex_url
        self._bibtex_path = Path(bibtex_path) if bibtex_path else None
        self._pdf_folder = Path(pdf_folder) if pdf_folder else None
        self._cache_dir = Path.home() / ".incite"

    def load_papers(self) -> list[Paper]:
        """Load papers from Paperpile BibTeX, with enrichment and PDF matching.

        Returns:
            List of Paper objects
        """
        import os

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = self._cache_dir / "paperpile_corpus.jsonl"

        # Fetch or read BibTeX
        if self._bibtex_url:
            bibtex_str = _fetch_bibtex(self._bibtex_url, self._cache_dir)
        else:
            bibtex_str = self._bibtex_path.read_text(encoding="utf-8")

        # Parse BibTeX entries
        entries = BibTeXParser.parse_string(bibtex_str)
        logger.info("Parsed %d BibTeX entries", len(entries))

        if not entries:
            logger.warning("No entries found in BibTeX")
            return []

        # Create API clients for enrichment (best-effort)
        from incite.corpus.openalex import OpenAlexClient
        from incite.corpus.semantic_scholar import SemanticScholarClient

        s2_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        openalex_email = os.environ.get("OPENALEX_EMAIL")

        s2_client = SemanticScholarClient(api_key=s2_key) if s2_key else None
        oa_client = OpenAlexClient(email=openalex_email) if openalex_email else None

        enricher = MetadataEnricher(s2_client=s2_client, openalex_client=oa_client)

        # Check if we have a cached corpus we can update incrementally
        existing_by_key: dict[str, Paper] = {}
        if corpus_path.exists():
            try:
                existing = load_corpus(corpus_path)
                existing_by_key = {p.bibtex_key: p for p in existing if p.bibtex_key}
            except Exception as e:
                logger.warning("Could not load existing corpus: %s", e)

        # Enrich entries, reusing cached papers when possible
        papers = []
        new_count = 0
        for entry in entries:
            bib_key = entry.get("key", "")
            if bib_key in existing_by_key:
                papers.append(existing_by_key[bib_key])
            else:
                paper = enricher.enrich_from_bibtex_entry(entry)
                papers.append(paper)
                new_count += 1

        logger.info(
            "Enriched %d new papers (%d reused from cache)",
            new_count,
            len(papers) - new_count,
        )

        # Match PDFs if folder is provided
        if self._pdf_folder and self._pdf_folder.exists():
            pdf_index = find_paperpile_pdfs(self._pdf_folder)
            matched = 0
            for paper in papers:
                pdf_path = match_paper_to_pdf(paper, pdf_index)
                if pdf_path:
                    paper.source_file = str(pdf_path)
                    matched += 1
            logger.info("Matched %d/%d papers to PDFs", matched, len(papers))

        # Cache the corpus
        save_corpus(papers, corpus_path)

        return papers

    def needs_refresh(self) -> bool:
        """Check if the corpus should be reloaded.

        For URL source: does HTTP HEAD to check ETag/Last-Modified.
        For file source: compares mtime against cached corpus.

        Returns:
            True if the underlying data has changed
        """
        corpus_path = self._cache_dir / "paperpile_corpus.jsonl"
        if not corpus_path.exists():
            return True

        if self._bibtex_url:
            return self._needs_refresh_url()
        elif self._bibtex_path:
            return self._needs_refresh_file(corpus_path)

        return True

    def _needs_refresh_url(self) -> bool:
        """Check if the remote BibTeX has changed via HTTP HEAD."""
        import requests

        etag_path = self._cache_dir / "paperpile_bibtex.etag"
        if not etag_path.exists():
            return True

        try:
            cached_etag = etag_path.read_text().strip()
            resp = requests.head(self._bibtex_url, timeout=10)
            resp.raise_for_status()
            remote_etag = resp.headers.get("ETag", "")
            return remote_etag != cached_etag
        except Exception as e:
            logger.warning("Could not check BibTeX URL: %s", e)
            return False  # Don't force refresh on network errors

    def _needs_refresh_file(self, corpus_path: Path) -> bool:
        """Check if the local .bib file is newer than the cached corpus."""
        if not self._bibtex_path.exists():
            return False
        return self._bibtex_path.stat().st_mtime > corpus_path.stat().st_mtime

    def cache_key(self) -> str:
        """Return cache key for this source.

        Returns:
            'paperpile' — all Paperpile sources share the same cache prefix.
        """
        return "paperpile"
