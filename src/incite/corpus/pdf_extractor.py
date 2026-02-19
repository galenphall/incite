"""PDF text extraction using PyMuPDF (fitz).

This module extracts text from academic PDFs while preserving paragraph structure
and detecting section headings via font size analysis.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PDFExtractionResult:
    """Result of PDF text extraction."""

    full_text: str
    paragraphs: list[str]
    section_headings: list[str]
    num_pages: int
    error: Optional[str] = None


def extract_pdf_text(pdf_path: Path | str) -> PDFExtractionResult:
    """Extract text from a PDF file.

    Uses PyMuPDF (fitz) to extract text while preserving paragraph structure.
    Detects section headings by analyzing font sizes.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        PDFExtractionResult with full_text, paragraphs, and section_headings
    """
    try:
        import fitz
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF extraction. Install with: pip install pymupdf"
        )

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.warning("PDF file not found: %s", pdf_path)
        return PDFExtractionResult(
            full_text="",
            paragraphs=[],
            section_headings=[],
            num_pages=0,
            error=f"PDF file not found: {pdf_path}",
        )

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return PDFExtractionResult(
            full_text="",
            paragraphs=[],
            section_headings=[],
            num_pages=0,
            error=f"Failed to open PDF: {e}",
        )

    try:
        blocks = []
        section_headings = []
        font_sizes: list[float] = []
        num_pages = doc.page_count  # Save before closing

        # Collect blocks with position info for header/footer detection
        _edge_texts: dict[str, int] = {}  # normalized text -> count of pages

        # First pass: collect all text blocks and their font sizes + positions
        for page in doc:
            page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            page_height = page_dict.get("height", 792.0)
            edge_zone = page_height * 0.10  # top/bottom 10%

            page_edge_texts_seen: set[str] = set()

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:  # Skip non-text blocks
                    continue

                block_text = []
                block_font_sizes = []

                for line in block.get("lines", []):
                    line_text = []
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            line_text.append(text)
                            block_font_sizes.append(span.get("size", 10.0))
                    if line_text:
                        block_text.append(" ".join(line_text))

                if block_text:
                    text = " ".join(block_text)
                    avg_font_size = (
                        sum(block_font_sizes) / len(block_font_sizes) if block_font_sizes else 10.0
                    )

                    # Check if block is in top/bottom edge zone
                    bbox = block.get("bbox", (0, 0, 0, 0))
                    block_top = bbox[1]
                    block_bottom = bbox[3]
                    in_edge = block_top < edge_zone or block_bottom > (page_height - edge_zone)

                    blocks.append((text, avg_font_size, in_edge))
                    font_sizes.extend(block_font_sizes)

                    # Track edge text for repeat detection
                    if in_edge and len(text) < 120:
                        # Normalize: strip digits (page numbers vary) for matching
                        norm = re.sub(r"\d+", "#", text.strip().lower())
                        if norm not in page_edge_texts_seen:
                            page_edge_texts_seen.add(norm)
                            _edge_texts[norm] = _edge_texts.get(norm, 0) + 1

        doc.close()

        # Identify repeating header/footer patterns (appear on 3+ pages)
        min_repeat = min(3, max(2, num_pages // 3))
        header_footer_patterns = {
            norm for norm, count in _edge_texts.items() if count >= min_repeat
        }

        if not blocks:
            return PDFExtractionResult(
                full_text="",
                paragraphs=[],
                section_headings=[],
                num_pages=num_pages,
                error="No text extracted from PDF",
            )

        # Calculate body text font size (most common)
        if font_sizes:
            sorted_sizes = sorted(font_sizes)
            body_size = sorted_sizes[len(sorted_sizes) // 2]  # median
        else:
            body_size = 10.0

        # Identify section headings (significantly larger font, short text)
        heading_threshold = body_size * 1.15  # 15% larger than body
        paragraphs = []
        current_paragraph = []

        for text, font_size, in_edge in blocks:
            # Clean the text
            text = _clean_text(text)
            if not text:
                continue

            # Skip repeating headers/footers
            if in_edge and len(text) < 120:
                norm = re.sub(r"\d+", "#", text.strip().lower())
                if norm in header_footer_patterns:
                    continue

            # Check if this is likely a section heading
            is_heading = (
                font_size > heading_threshold
                and len(text) < 100
                and not text.endswith(".")
                and _looks_like_heading(text)
            )

            if is_heading:
                # Save any accumulated paragraph
                if current_paragraph:
                    para_text = " ".join(current_paragraph)
                    if para_text.strip():
                        paragraphs.append(para_text.strip())
                    current_paragraph = []

                section_headings.append(text)
                paragraphs.append(text)  # Include headings in paragraph list
            else:
                # Check if this starts a new paragraph
                if _starts_new_paragraph(text, current_paragraph):
                    if current_paragraph:
                        para_text = " ".join(current_paragraph)
                        if para_text.strip():
                            paragraphs.append(para_text.strip())
                    current_paragraph = [text]
                else:
                    current_paragraph.append(text)

        # Don't forget the last paragraph
        if current_paragraph:
            para_text = " ".join(current_paragraph)
            if para_text.strip():
                paragraphs.append(para_text.strip())

        # Filter front matter (affiliations, emails before first body section)
        paragraphs = _filter_front_matter(paragraphs, section_headings)

        # Build full text
        full_text = "\n\n".join(paragraphs)

        return PDFExtractionResult(
            full_text=full_text,
            paragraphs=paragraphs,
            section_headings=section_headings,
            num_pages=num_pages,
        )

    except Exception as e:
        try:
            doc.close()
        except Exception:
            pass  # Already closed
        return PDFExtractionResult(
            full_text="",
            paragraphs=[],
            section_headings=[],
            num_pages=0,
            error=f"Error extracting text: {e}",
        )


def _clean_text(text: str) -> str:
    """Clean extracted text.

    Removes:
    - Excessive whitespace
    - Common PDF artifacts
    - Page numbers and headers/footers patterns
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove common artifacts
    text = re.sub(r"^\d+\s*$", "", text)  # Lone page numbers
    text = re.sub(r"^Page \d+.*$", "", text, flags=re.IGNORECASE)

    # Remove hyphenation at line breaks
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)

    return text.strip()


def _looks_like_heading(text: str) -> bool:
    """Check if text looks like a section heading."""
    # Too short to be a meaningful heading (avoid artifacts like "2 A")
    if len(text.strip()) < 5:
        return False

    # Reject patterns that look like page metadata: "414 B. Rabe", "123 Author"
    # These are page numbers followed by short author names, not section headings
    if re.match(r"^\d{2,}\s+[A-Z]\.\s+\w+$", text):
        return False
    # Reject "414 B. Rabe" style with middle initial
    if re.match(r"^\d{2,}\s+[A-Z]\.?\s+[A-Z][a-z]+$", text):
        return False

    # Common heading patterns
    heading_patterns = [
        r"^\d+\.?\s+[A-Z][a-z]",  # "1. Introduction" (require lowercase after cap)
        r"^\d+\.?\s+[A-Z]{2,}",  # "1. INTRODUCTION" (all caps word)
        r"^[IVX]+\.?\s+[A-Z]",  # Roman numerals
        (
            r"^(Abstract|Introduction|Background|Methods?|Results?"
            r"|Discussion|Conclusion|References|Acknowledgments?"
            r"|Related Work|Appendix)"
        ),
    ]

    for pattern in heading_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return True

    # Short text that's all/mostly capitalized (but must have >1 word)
    if len(text) < 60 and text.upper() == text and len(text.split()) > 1:
        return True

    return False


def _is_front_matter(text: str) -> bool:
    """Check if text looks like front matter (affiliations, emails, ORCID, etc.)."""
    patterns = [
        r"^(Department|School|Faculty|Institute|College|Center|Centre|Laboratory)\s+of\b",
        r"^\d+\s+(Department|School|Faculty|Institute|College|Center|Centre)\b",
        r"^[a-z]\s+(Department|School|Faculty|Institute|College|Center|Centre)\b",
        r"\b(email|e-mail|correspondence):\s*\S+@",
        r"\S+@\S+\.\S+",  # email addresses
        r"\bORCID\b",
        r"^\*\s*(Corresponding|To whom)",
    ]
    for p in patterns:
        if re.search(p, text[:200], re.IGNORECASE):
            return True
    return False


def _filter_front_matter(paragraphs: list[str], section_headings: list[str]) -> list[str]:
    """Remove front matter paragraphs (affiliations, emails) before first body section.

    Only filters paragraphs before the first real section heading (Abstract,
    Introduction, etc.), and only if they match front matter patterns.
    """
    if not paragraphs:
        return paragraphs

    # Find where body content starts (first section heading)
    body_start = 0
    for i, para in enumerate(paragraphs):
        if para in section_headings or _looks_like_heading(para):
            body_start = i
            break
    else:
        # No headings found — don't filter anything
        return paragraphs

    # Filter front matter paragraphs before body start
    filtered = []
    for i, para in enumerate(paragraphs):
        if i < body_start and _is_front_matter(para):
            continue
        filtered.append(para)

    return filtered


def _starts_new_paragraph(text: str, current_paragraph: list[str]) -> bool:
    """Heuristically determine if text starts a new paragraph."""
    if not current_paragraph:
        return True

    # Starts with common paragraph indicators
    if re.match(r"^[A-Z]", text):
        last_text = current_paragraph[-1] if current_paragraph else ""
        # Previous block ended with period/question mark
        if last_text.rstrip().endswith((".", "?", "!", ":")):
            return True

    return False


def find_zotero_pdfs(
    storage_dir: Path | str,
    paper_keys: Optional[list[str]] = None,
) -> dict[str, Path]:
    """Find PDF files in Zotero storage directory.

    Zotero stores attachments in: ~/Zotero/storage/<item_key>/<filename>.pdf

    Args:
        storage_dir: Path to Zotero storage directory (e.g., ~/Zotero/storage)
        paper_keys: Optional list of item keys to look for. If None, finds all PDFs.

    Returns:
        Dict mapping item key -> PDF path
    """
    storage_dir = Path(storage_dir).expanduser()

    if not storage_dir.exists():
        return {}

    pdf_map: dict[str, Path] = {}

    # If specific keys requested, only look in those directories
    if paper_keys:
        dirs_to_check = [storage_dir / key for key in paper_keys]
    else:
        dirs_to_check = list(storage_dir.iterdir())

    for item_dir in dirs_to_check:
        if not item_dir.is_dir():
            continue

        item_key = item_dir.name

        # Find PDFs in this directory
        pdfs = list(item_dir.glob("*.pdf"))
        if pdfs:
            # Prefer the largest PDF (usually the main paper, not supplement)
            pdfs.sort(key=lambda p: p.stat().st_size, reverse=True)
            pdf_map[item_key] = pdfs[0]

    return pdf_map


def _normalize_title(title: str) -> str:
    """Normalize a title for fuzzy matching."""
    # Lowercase, remove punctuation, collapse whitespace
    title = title.lower()
    title = re.sub(r"[^\w\s]", " ", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def _title_matches(title1: str, title2: str, threshold: float = 0.8) -> bool:
    """Check if two titles match (fuzzy)."""
    t1 = _normalize_title(title1)
    t2 = _normalize_title(title2)

    # Exact match after normalization
    if t1 == t2:
        return True

    # One contains the other (for truncated titles)
    if len(t1) > 20 and len(t2) > 20:
        if t1 in t2 or t2 in t1:
            return True

    # Word overlap ratio
    words1 = set(t1.split())
    words2 = set(t2.split())
    if not words1 or not words2:
        return False

    overlap = len(words1 & words2)
    ratio = overlap / max(len(words1), len(words2))
    return ratio >= threshold


def extract_pdfs_for_corpus(
    papers: list,
    zotero_storage: Optional[Path | str] = None,
    show_progress: bool = True,
) -> dict[str, PDFExtractionResult]:
    """Extract text from PDFs for a corpus of papers.

    Matching strategy (in order):
    1. Direct source_file path from Paper (set by Zotero database reader)
    2. Zotero item key from bibtex_key (legacy BibTeX-based loading)
    3. PDF filename matches paper title (fuzzy, legacy fallback)

    Args:
        papers: List of Paper objects
        zotero_storage: Optional path to Zotero storage directory (only needed for
            legacy BibTeX-based loading; not needed if papers have source_file set)
        show_progress: Whether to show progress bar

    Returns:
        Dict mapping paper.id -> PDFExtractionResult
    """
    from tqdm import tqdm

    # Build fallback maps only if needed (when some papers lack source_file)
    pdf_map: dict[str, Path] = {}
    pdf_by_filename: dict[str, Path] = {}

    needs_fallback = any(not p.source_file for p in papers)
    if needs_fallback and zotero_storage:
        zotero_storage = Path(zotero_storage).expanduser()
        pdf_map = find_zotero_pdfs(zotero_storage)
        for item_key, pdf_path in pdf_map.items():
            filename = pdf_path.stem
            pdf_by_filename[_normalize_title(filename)] = pdf_path

    results: dict[str, PDFExtractionResult] = {}

    iterator = tqdm(papers, desc="Extracting PDFs", disable=not show_progress)
    for paper in iterator:
        pdf_path = None

        # Strategy 1: Use direct source_file path (from Zotero database reader)
        if paper.source_file:
            candidate = Path(paper.source_file)
            if candidate.exists():
                pdf_path = candidate

        # Strategy 2: Try Zotero item key matching (legacy)
        if pdf_path is None and pdf_map:
            potential_keys = []
            if paper.bibtex_key:
                potential_keys.append(paper.bibtex_key)
                # Extract 8-char Zotero key pattern
                match = re.search(r"[A-Z0-9]{8}", paper.bibtex_key)
                if match:
                    potential_keys.append(match.group())

            for key in potential_keys:
                if key in pdf_map:
                    pdf_path = pdf_map[key]
                    break

        # Strategy 3: Try matching by PDF filename to paper title (legacy)
        if pdf_path is None and paper.title and pdf_by_filename:
            paper_title_norm = _normalize_title(paper.title)

            if paper_title_norm in pdf_by_filename:
                pdf_path = pdf_by_filename[paper_title_norm]
            else:
                for filename_norm, path in pdf_by_filename.items():
                    if _title_matches(paper.title, filename_norm, threshold=0.7):
                        pdf_path = path
                        break

        # Extract text if we found a matching PDF
        if pdf_path:
            result = extract_pdf_text(pdf_path)
            if result.full_text:
                results[paper.id] = result

    return results
