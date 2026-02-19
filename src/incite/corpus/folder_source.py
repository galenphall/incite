"""Folder-of-PDFs corpus source.

Enables users who don't use Zotero to point inCite at any folder of PDFs.
PDFs are scanned recursively, metadata is extracted via PyMuPDF, and full text
is extracted using the existing pdf_extractor module.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional

from incite.corpus.loader import load_corpus, save_corpus
from incite.corpus.pdf_extractor import extract_pdf_text
from incite.models import Paper

logger = logging.getLogger(__name__)


def _scan_folder_pdfs(folder: Path) -> list[Path]:
    """Recursively find all .pdf files in a folder.

    Args:
        folder: Root folder to scan

    Returns:
        Sorted list of PDF paths (sorted for deterministic ordering)
    """
    pdfs = sorted(folder.rglob("*.pdf"))
    return pdfs


def _clean_filename_title(filename: str) -> str:
    """Convert a PDF filename into a reasonable title.

    Strips the .pdf extension, replaces underscores and hyphens with spaces,
    and collapses whitespace.
    """
    name = Path(filename).stem
    # Replace underscores and hyphens with spaces
    name = re.sub(r"[_\-]+", " ", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _extract_year_from_creation_date(date_str: str) -> Optional[int]:
    """Extract year from PyMuPDF creation date format.

    PyMuPDF reports dates as "D:YYYYMMDDHHmmSS..." or sometimes just "YYYY...".
    """
    if not date_str:
        return None
    match = re.search(r"D:(\d{4})", date_str)
    if match:
        year = int(match.group(1))
        if 1900 <= year <= 2100:
            return year
    # Fallback: any 4-digit year
    match = re.search(r"\b(19|20)\d{2}\b", date_str)
    if match:
        return int(match.group())
    return None


def _extract_year_from_filename(filename: str) -> Optional[int]:
    """Try to extract a year from a filename."""
    match = re.search(r"(?<!\d)(19|20)\d{2}(?!\d)", filename)
    if match:
        return int(match.group())
    return None


def _extract_largest_font_text(pdf_path: Path) -> Optional[str]:
    """Extract the largest-font text from page 1 of a PDF.

    This is used as a title fallback when PDF metadata has no title.

    Returns:
        The largest-font text block on page 1, or None.
    """
    try:
        import fitz
    except ImportError:
        return None

    try:
        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            doc.close()
            return None

        page = doc[0]
        page_dict = page.get_text("dict", flags=0)
        doc.close()

        best_text = None
        best_size = 0.0

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                line_text_parts = []
                line_sizes = []
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        line_text_parts.append(text)
                        line_sizes.append(span.get("size", 0.0))
                if line_text_parts and line_sizes:
                    avg_size = sum(line_sizes) / len(line_sizes)
                    text = " ".join(line_text_parts)
                    if avg_size > best_size and 5 < len(text) < 300:
                        best_size = avg_size
                        best_text = text

        return best_text
    except (OSError, ValueError, RuntimeError) as e:
        logger.warning("Failed to extract font text from %s: %s", pdf_path, e)
        return None


def _extract_abstract_from_result(extraction_result) -> str:
    """Extract the abstract from a PDFExtractionResult.

    Looks for the first substantial paragraph after the title region,
    using the same front-matter filtering as pdf_extractor.
    """
    paragraphs = extraction_result.paragraphs
    if not paragraphs:
        return ""

    # Look for an explicit "Abstract" section
    for i, para in enumerate(paragraphs):
        if re.match(r"^abstract\b", para.strip(), re.IGNORECASE):
            # The abstract text is the next paragraph(s)
            if i + 1 < len(paragraphs):
                abstract_text = paragraphs[i + 1]
                # If it's very short, concatenate with the next paragraph
                if len(abstract_text) < 100 and i + 2 < len(paragraphs):
                    abstract_text += " " + paragraphs[i + 2]
                return abstract_text.strip()

    # Fallback: first paragraph longer than 100 chars (skip short title/author blocks)
    for para in paragraphs:
        if len(para) > 100:
            return para.strip()

    return ""


def extract_pdf_metadata(pdf_path: Path) -> dict:
    """Extract lightweight metadata from a PDF using PyMuPDF.

    Returns a dict with keys: title, authors, year.
    Falls back to filename-based extraction when metadata is missing.
    """
    try:
        import fitz
    except ImportError:
        return {
            "title": _clean_filename_title(pdf_path.name),
            "authors": [],
            "year": _extract_year_from_filename(pdf_path.name),
        }

    title = None
    authors = []
    year = None

    try:
        doc = fitz.open(pdf_path)
        meta = doc.metadata or {}
        doc.close()
    except (OSError, ValueError, RuntimeError) as e:
        logger.warning(
            "Failed to read PDF metadata from %s: %s; falling back to filename",
            pdf_path,
            e,
        )
        return {
            "title": _clean_filename_title(pdf_path.name),
            "authors": [],
            "year": _extract_year_from_filename(pdf_path.name),
        }

    # Title: metadata -> largest font on page 1 -> cleaned filename
    raw_title = (meta.get("title") or "").strip()
    if raw_title and len(raw_title) > 3 and raw_title.lower() != "untitled":
        title = raw_title
    else:
        largest_font = _extract_largest_font_text(pdf_path)
        if largest_font:
            title = largest_font
        else:
            title = _clean_filename_title(pdf_path.name)

    # Authors: metadata field, split on common separators.
    # Semicolons separate authors in "Last, First; Last, First" format.
    # If no semicolons, try " and " as separator, then fall back to comma.
    raw_author = (meta.get("author") or "").strip()
    if raw_author:
        if ";" in raw_author:
            parts = raw_author.split(";")
        elif " and " in raw_author.lower():
            parts = re.split(r"\band\b", raw_author, flags=re.IGNORECASE)
        else:
            parts = [raw_author]
        authors = [a.strip() for a in parts if a.strip()]

    # Year: creation date metadata -> filename regex
    year = _extract_year_from_creation_date(meta.get("creationDate", ""))
    if year is None:
        year = _extract_year_from_filename(pdf_path.name)

    return {
        "title": title,
        "authors": authors,
        "year": year,
    }


def _paper_id_from_path(pdf_path: Path) -> str:
    """Generate a deterministic paper ID from a PDF path.

    Uses MD5 hash of the absolute path for a compact, collision-resistant ID.
    """
    return hashlib.md5(str(pdf_path.resolve()).encode()).hexdigest()[:16]


class FolderCorpusSource:
    """CorpusSource implementation that reads from a folder of PDFs.

    Satisfies the CorpusSource protocol via structural typing.
    Scans a folder recursively for PDFs, extracts metadata and full text,
    and caches the result to ~/.incite/.

    Incremental: on re-scan, only processes new/modified PDFs and removes
    papers for deleted PDFs.
    """

    name: str = "folder"

    def __init__(self, folder_path: str | Path):
        """Initialize FolderCorpusSource.

        Args:
            folder_path: Path to a folder containing PDF files

        Raises:
            FileNotFoundError: If folder_path does not exist
            NotADirectoryError: If folder_path is not a directory
        """
        self._folder = Path(folder_path).expanduser().resolve()
        if not self._folder.exists():
            raise FileNotFoundError(f"Folder not found: {self._folder}")
        if not self._folder.is_dir():
            raise NotADirectoryError(f"Not a directory: {self._folder}")

    def load_papers(self) -> list[Paper]:
        """Load papers from the folder, using cache when possible.

        On first call, processes all PDFs. On subsequent calls, only
        processes new/modified PDFs and removes papers for deleted PDFs.

        Returns:
            List of Paper objects

        Raises:
            ValueError: If folder contains no PDF files
        """
        cache_path = self._corpus_cache_path()
        pdfs = _scan_folder_pdfs(self._folder)

        if not pdfs:
            raise ValueError(f"No PDF files found in {self._folder}")

        # Build a map of current PDFs and their mtimes
        pdf_mtimes = {}
        for pdf in pdfs:
            try:
                pdf_mtimes[str(pdf.resolve())] = pdf.stat().st_mtime
            except OSError:
                continue

        # Try loading cached corpus for incremental update
        cached_papers: dict[str, Paper] = {}
        cached_mtimes: dict[str, float] = {}
        meta_path = self._meta_cache_path()

        if cache_path.exists() and meta_path.exists():
            try:
                cached_papers = {p.id: p for p in load_corpus(cache_path)}
                with open(meta_path) as f:
                    cached_meta = json.load(f)
                cached_mtimes = cached_meta.get("mtimes", {})
            except Exception as e:
                logger.warning("Could not load folder cache, rebuilding: %s", e)
                cached_papers = {}
                cached_mtimes = {}

        # Determine which PDFs need (re)processing
        # A paper's source_file stores the resolved PDF path
        cached_by_path: dict[str, Paper] = {}
        for paper in cached_papers.values():
            if paper.source_file:
                cached_by_path[paper.source_file] = paper

        papers_out: list[Paper] = []
        new_mtimes: dict[str, float] = {}
        processed = 0
        skipped = 0

        for pdf_path_str, mtime in pdf_mtimes.items():
            pdf_path = Path(pdf_path_str)
            old_mtime = cached_mtimes.get(pdf_path_str)

            if pdf_path_str in cached_by_path and old_mtime is not None and mtime <= old_mtime:
                # PDF unchanged, reuse cached paper
                papers_out.append(cached_by_path[pdf_path_str])
                new_mtimes[pdf_path_str] = old_mtime
                skipped += 1
            else:
                # New or modified PDF — process it
                paper = self._process_pdf(pdf_path)
                if paper is not None:
                    papers_out.append(paper)
                    new_mtimes[pdf_path_str] = mtime
                    processed += 1

        logger.info(
            "Folder scan: %d papers (%d processed, %d cached, %d PDFs removed)",
            len(papers_out),
            processed,
            skipped,
            len(cached_papers) - skipped,
        )

        # Save updated cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_corpus(papers_out, cache_path)
        with open(meta_path, "w") as f:
            json.dump({"mtimes": new_mtimes}, f)

        return papers_out

    def needs_refresh(self) -> bool:
        """Check if any PDF in the folder is newer than the cached corpus.

        Returns:
            True if new/modified PDFs exist or the cache is missing
        """
        cache_path = self._corpus_cache_path()
        if not cache_path.exists():
            return True

        cache_mtime = cache_path.stat().st_mtime

        # Check if any PDF is newer than the cache
        for pdf in _scan_folder_pdfs(self._folder):
            try:
                if pdf.stat().st_mtime > cache_mtime:
                    return True
            except OSError:
                continue

        return False

    def cache_key(self) -> str:
        """Return a deterministic cache key for this folder.

        Based on MD5 hash of the folder path.
        """
        path_hash = hashlib.md5(str(self._folder).encode()).hexdigest()[:12]
        return f"folder_{path_hash}"

    def _corpus_cache_path(self) -> Path:
        """Path to the cached corpus JSONL file."""
        cache_dir = Path.home() / ".incite"
        return cache_dir / f"{self.cache_key()}_corpus.jsonl"

    def _meta_cache_path(self) -> Path:
        """Path to the cached metadata (mtimes) JSON file."""
        cache_dir = Path.home() / ".incite"
        return cache_dir / f"{self.cache_key()}_meta.json"

    def _process_pdf(self, pdf_path: Path) -> Optional[Paper]:
        """Extract metadata and full text from a single PDF.

        Returns None if the PDF cannot be read (corrupt, empty, etc.).
        """
        paper_id = _paper_id_from_path(pdf_path)

        # Extract metadata
        meta = extract_pdf_metadata(pdf_path)
        title = meta["title"]
        if not title:
            title = _clean_filename_title(pdf_path.name)

        # Extract full text
        extraction = extract_pdf_text(pdf_path)
        if extraction.error:
            logger.warning("Skipping %s: %s", pdf_path.name, extraction.error)
            # Still create the paper with metadata but no full text
            if not extraction.full_text:
                # If we can't even open the file, skip entirely
                if "Failed to open" in (extraction.error or ""):
                    return None

        abstract = meta.get("abstract") or _extract_abstract_from_result(extraction)

        return Paper(
            id=paper_id,
            title=title,
            abstract=abstract,
            authors=meta.get("authors", []),
            year=meta.get("year"),
            full_text=extraction.full_text or None,
            paragraphs=extraction.paragraphs,
            source_file=str(pdf_path.resolve()),
        )
