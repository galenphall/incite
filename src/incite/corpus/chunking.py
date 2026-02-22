"""Chunking module for splitting papers into paragraph-level chunks.

This module provides functions to chunk papers for paragraph-level retrieval.
The default strategy splits on paragraph boundaries (double newlines).
"""

import re
from typing import Optional

from incite.models import Chunk, Paper, format_paper_metadata_prefix


def _build_paper_metadata_prefix(paper: Paper) -> str:
    """Build a metadata prefix string for chunk embedding context.

    Delegates to format_paper_metadata_prefix() for consistent formatting
    across retrieval and training data.
    """
    return format_paper_metadata_prefix(
        title=paper.title,
        author_lastnames=paper.author_lastnames,
        year=paper.year,
        journal=paper.journal,
    )


def chunk_paper(
    paper: Paper,
    max_tokens: int = 512,
    min_chunk_length: int = 150,
    source: str | None = None,
    pre_structured: tuple[list[str], list[str | None]] | None = None,
) -> list[Chunk]:
    """Split a paper into chunks for paragraph-level retrieval.

    Strategy:
    1. If pre_structured provided, use those (paragraphs, sections) directly
    2. Else if paper has `paragraphs` populated (from PDF extraction), use those
    3. Else if paper has `full_text`, split on double-newlines
    4. Else fallback to treating abstract as a single chunk

    Args:
        paper: Paper object to chunk
        max_tokens: Maximum tokens per chunk (approximate, uses char/4 heuristic)
        min_chunk_length: Minimum character length for a chunk to be kept
        source: Extraction method label (e.g. "html", "grobid", "abstract")
        pre_structured: Pre-processed (paragraphs, section_headings) from HTML extraction.
            When provided, skips heading detection and paragraph splitting.

    Returns:
        List of Chunk objects
    """
    chunks: list[Chunk] = []
    metadata_prefix = _build_paper_metadata_prefix(paper)

    # Determine source text and section assignments
    pre_sections: list[str | None] | None = None

    if pre_structured is not None:
        paragraphs, pre_sections = pre_structured
        if not paragraphs:
            return []
    elif paper.paragraphs:
        paragraphs = paper.paragraphs
    elif paper.full_text:
        paragraphs = _split_into_paragraphs(paper.full_text)
    elif paper.abstract:
        # Fallback: use abstract as single chunk
        paragraphs = [paper.abstract]
    else:
        return []  # No text available

    # Pre-scan: detect reference section by consecutive bibliography entries
    # (fallback when no "References" heading is detected)
    # Skip for pre_structured input since HTML preprocessing already filtered
    ref_cutoff = None if pre_structured else _find_reference_cutoff(paragraphs)

    # Filter and create chunks
    char_offset = 0
    current_section: Optional[str] = None

    for i, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue

        # Update section from pre_structured data
        if pre_sections is not None and i < len(pre_sections):
            if pre_sections[i] is not None:
                current_section = pre_sections[i]

        # Stop if we've reached the detected reference section
        if ref_cutoff is not None and i >= ref_cutoff:
            break

        # Check if this paragraph is a section heading (skip for pre_structured)
        if pre_structured is None and _looks_like_heading(para):
            current_section = para
            # Stop at reference/bibliography sections (always at end of paper)
            if _is_reference_section(current_section):
                break
            # Don't create a chunk for just a heading
            char_offset += len(para) + 2  # +2 for paragraph separator
            continue

        # Skip very short chunks
        if len(para) < min_chunk_length:
            char_offset += len(para) + 2
            continue

        # Skip corrupted text (encoding errors, OCR garbage)
        if _is_corrupted_text(para):
            char_offset += len(para) + 2
            continue

        # Skip figure/table captions
        if _is_figure_or_table_caption(para):
            char_offset += len(para) + 2
            continue

        # Skip boilerplate text (journal notices, copyright, download notices)
        if _is_boilerplate(para):
            char_offset += len(para) + 2
            continue

        # Skip acknowledgment/funding/author contribution paragraphs
        # (only in last 30% of document to avoid filtering content about funding)
        if i > len(paragraphs) * 0.7 and _is_acknowledgment_text(para):
            char_offset += len(para) + 2
            continue

        # Skip equation-heavy or garbled text
        if _is_equation_heavy(para):
            char_offset += len(para) + 2
            continue

        # Skip table data (mostly numbers and short tokens)
        if _is_table_data(para):
            char_offset += len(para) + 2
            continue

        # Skip individual bibliography entries that slipped past section-level detection
        if _is_bibliography_chunk(para):
            char_offset += len(para) + 2
            continue

        # Split long paragraphs if needed
        max_chars = max_tokens * 4  # Rough token-to-char ratio
        if len(para) > max_chars:
            sub_chunks = _split_long_text(para, max_chars, min_chunk_length)
            for j, sub_text in enumerate(sub_chunks):
                chunk_id = f"{paper.id}::chunk_{len(chunks)}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        paper_id=paper.id,
                        text=sub_text,
                        section=current_section,
                        char_offset=char_offset,
                        source=source,
                        context_text=metadata_prefix,
                    )
                )
                char_offset += len(sub_text)
        else:
            chunk_id = f"{paper.id}::chunk_{len(chunks)}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    paper_id=paper.id,
                    text=para,
                    section=current_section,
                    char_offset=char_offset,
                    source=source,
                    context_text=metadata_prefix,
                )
            )

        char_offset += len(para) + 2  # +2 for paragraph separator

    return chunks


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on double-newlines."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split on double newlines (paragraph boundaries)
    paragraphs = re.split(r"\n\s*\n", text)

    # Clean up each paragraph
    result = []
    for para in paragraphs:
        # Collapse internal newlines to spaces (common in PDFs)
        para = re.sub(r"\s*\n\s*", " ", para)
        # Normalize whitespace
        para = re.sub(r"\s+", " ", para)
        para = para.strip()
        if para:
            result.append(para)

    return result


def _looks_like_heading(text: str) -> bool:
    """Check if text looks like a section heading."""
    # Too long to be a heading
    if len(text) > 100:
        return False

    # Too short to be a meaningful heading
    if len(text.strip()) < 5:
        return False

    # Ends with period = probably not a heading
    if text.rstrip().endswith("."):
        return False

    # Reject page metadata: "414 B. Rabe", "123 Author Name"
    if re.match(r"^\d{2,}\s+[A-Z]\.?\s+[A-Z][a-z]+$", text):
        return False

    # Common heading patterns
    heading_patterns = [
        r"^\d+\.?\s+[A-Z][a-z]",  # "1. Introduction" (require lowercase after cap)
        r"^\d+\.?\s+[A-Z]{2,}",  # "1. INTRODUCTION" (all caps word)
        r"^[IVX]+\.?\s+[A-Z]",  # Roman numerals
        (
            r"^(Abstract|Introduction|Background|Methods?|Results?"
            r"|Discussion|Conclusions?|References|Acknowledgments?"
            r"|Related Work|Appendix|Summary|Overview)"
        ),
    ]

    for pattern in heading_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return True

    # All caps and short (must have >1 word to avoid artifacts)
    if len(text) < 50 and text.upper() == text and len(text.split()) > 1:
        return True

    return False


def _is_reference_section(section_name: Optional[str]) -> bool:
    """Check if section name indicates a references/bibliography section.

    Handles variants like:
    - "References"
    - "7. References"
    - "References and Notes"
    - "Literature Cited"
    - "VIII. Bibliography"
    """
    if not section_name:
        return False
    # Strip leading numbering (digits or Roman numerals): "7. References" -> "References"
    stripped = re.sub(r"^[\d.IVX]+\.?\s*", "", section_name).strip()
    # Match STARTS with (not exact match) - handles "References and Notes" etc.
    return bool(
        re.match(
            r"^(references|bibliography|works\s+cited|literature\s+cited|cited\s+literature)",
            stripped,
            re.IGNORECASE,
        )
    )


def _looks_like_bibliography_entry(text: str) -> bool:
    """Detect bibliography entries by content patterns.

    Used as a backup when section-based detection fails (e.g., stale cache,
    inline citations, or papers with non-standard section names).

    Returns True if the text has 3+ bibliographic signals and is short enough
    to be a typical reference entry.
    """
    # Bibliography entries are typically short
    if len(text) > 600:
        return False

    # Common patterns in bibliography entries
    bib_signals = [
        r"\(\d{4}[a-z]?\)",  # Year in parens: (2020) or (2020a)
        r"\d{4}\.\s",  # Year with period: 2020.
        r"doi:\s*10\.",  # DOI
        r"https?://",  # URLs
        r"\bp+\.\s*\d+",  # Page numbers: p. 123 or pp. 123-456
        r"\bVol\.\s*\d+",  # Volume: Vol. 42
        r"Journal of|Proceedings of|Trans\.|Rev\.",  # Publication venues
        r"\bet\s+al\.",  # "et al."
        r"arXiv:\d+\.\d+",  # arXiv IDs
        r"ISBN[\s:-]*[\dX-]+",  # ISBN
    ]

    matches = sum(1 for p in bib_signals if re.search(p, text, re.IGNORECASE))
    return matches >= 3


def _is_bibliography_chunk(text: str) -> bool:
    """Detect individual bibliography entries that slipped past section-level detection.

    Section-level detection (_is_reference_section, _find_reference_cutoff) catches
    the bulk of references, but individual entries leak through when:
    - The "References" heading isn't detected (non-standard formatting, stale cache)
    - Fewer than 3 consecutive entries exist (appendix refs, footnotes)
    - Entries are in the first 60% of the document (outside the scan window)

    This per-chunk filter catches remaining entries via:
    1. Content signals (3+ bibliographic patterns in short text)
    2. Structural patterns (starts with author-year or numbered ref format)
    """
    if len(text) > 600:
        return False

    # Content-based: reuse existing signal detection
    if _looks_like_bibliography_entry(text):
        return True

    # Structural: starts with typical reference entry patterns + at least 1 bib signal
    bib_start_patterns = [
        r"^\[\d+\]\s*[A-Z]",  # [1] Author...
        r"^[A-Z][a-z]+,\s+[A-Z]\.",  # Author, F.
        r"^[A-Z]{2,},\s+[A-Z]\.",  # AUTHOR, F.
        r"^\d+\.\s+[A-Z][a-z]+,\s+[A-Z]",  # 1. Author, F.
    ]

    # Quick signal check (any single signal present?)
    quick_signals = [
        r"\(\d{4}[a-z]?\)",  # (2020)
        r"\d{4}\.\s",  # 2020.
        r"doi:\s*10\.",
        r"https?://",
        r"\bVol\.\s*\d+",
        r"Journal of|Proceedings of",
        r"\bet\s+al\.",
    ]
    has_signal = any(re.search(p, text, re.IGNORECASE) for p in quick_signals)

    if has_signal:
        for pat in bib_start_patterns:
            if re.match(pat, text):
                return True

    return False


def _is_corrupted_text(text: str, min_alpha_ratio: float = 0.30) -> bool:
    """Check if text is corrupted (encoding errors, OCR garbage, pure formulas).

    Returns True if less than min_alpha_ratio of characters are alphabetic.
    """
    if not text:
        return True
    alpha_count = sum(c.isalpha() for c in text)
    return alpha_count / len(text) < min_alpha_ratio


def _find_reference_cutoff(paragraphs: list[str]) -> Optional[int]:
    """Find where the reference section starts by detecting consecutive bib entries.

    If 3+ consecutive paragraphs in the last 40% of the document match bibliography
    patterns, return the index of the first one. Returns None if no reference section
    detected.
    """
    if len(paragraphs) < 10:
        return None

    # Only scan the last 40% of paragraphs (references are always at the end)
    scan_start = int(len(paragraphs) * 0.6)
    consecutive = 0
    gap = 0
    first_bib_idx = None

    for i in range(scan_start, len(paragraphs)):
        para = paragraphs[i].strip()
        if not para:
            continue

        if _looks_like_bibliography_entry(para):
            if consecutive == 0:
                first_bib_idx = i
            consecutive += 1
            gap = 0  # Reset gap counter on bib match
            if consecutive >= 3:
                return first_bib_idx
        else:
            if consecutive >= 2 and gap == 0:
                # Allow one non-bib paragraph gap (headings, page breaks)
                gap = 1
            else:
                consecutive = 0
                gap = 0
                first_bib_idx = None

    return None


def _is_figure_or_table_caption(text: str) -> bool:
    """Skip figure and table captions."""
    return bool(
        re.match(
            r"^(Figure|Fig\.|Table|Supplementary Figure|Supplementary Table|Extended Data)\s+\d",
            text,
            re.IGNORECASE,
        )
    )


def _is_boilerplate(text: str) -> bool:
    """Skip journal boilerplate, copyright, download notices, and HTML artifacts."""
    patterns = [
        r"^This content downloaded from",
        r"^Downloaded from\b",
        r"^All use subject to",
        r"^Copyright\s+(©|\(c\))?\s*\d{4}",
        r"^©\s*\d{4}",
        r"^Licensed under",
        r"^This article is licensed under",
        r"^(Published|Received|Accepted)\s+\d{1,2}\s+\w+\s+\d{4}",
        r"^This is an open.access article",
        r"^Authorized licensed use limited to",
        r"^All rights reserved\.",
        r"^For permissions,?\s+please",
        # HTML-specific boilerplate (safe for PDFs — these never appear in PDF text)
        r"^(Sign in|Log in|Create (an )?account)\b",
        r"^(Share|Tweet|Email|Print)\s+(this|article)",
        r"^(Accept|Reject)\s+(all\s+)?cookies?\b",
        r"^We use cookies",
        r"^(View|Show)\s+(all\s+)?(references|citations|figures|tables)",
        r"^(Cited by|Metrics|Altmetrics)\b",
        r"^Subscribe to\b",
        r"^(Access|Read)\s+the full",
        r"^Author (contributions?|information)\b",
        r"^Data (availability|sharing)\b",
    ]
    for p in patterns:
        if re.search(p, text[:120], re.IGNORECASE):
            return True

    # Journal header patterns: "PNAS 2021 Vol. 118...", "Journal of X 18(3): 298..."
    if re.match(
        r"^(PNAS|Nature|Science|PLoS|Journal of)\b.*\d{4}.*\b(Vol|doi|https?://)",
        text[:150],
        re.IGNORECASE,
    ):
        return True

    # Page-of-page patterns: "1 of 7", "3/15"
    if re.match(r"^\d+\s+(of|/)\s+\d+\b", text[:20]):
        return True

    return False


def _is_acknowledgment_text(text: str) -> bool:
    """Skip acknowledgment, funding, and author contribution paragraphs."""
    return bool(
        re.match(
            r"^(This work was (supported|funded)|"
            r"We (thank|acknowledge|are grateful)|"
            r"The authors? (thank|acknowledge|are grateful|declare|contributed)|"
            r"Author contributions?:?\s|"
            r"Funding\b|"
            r"This research was (funded|supported)|"
            r"Conflicts? of interest|"
            r"Declaration of (competing )?interests?|"
            r"Data availability|"
            r"Supplementary (data|materials?|information)\b)",
            text,
            re.IGNORECASE,
        )
    )


def _is_table_data(text: str) -> bool:
    """Skip chunks that are table data (high ratio of numbers and short tokens)."""
    tokens = text.split()
    if len(tokens) < 5:
        return False
    # Count number-like tokens (digits, percentages, ±, ranges)
    num_tokens = sum(1 for t in tokens if re.match(r"^[\d.,±%<>≤≥$€£]+$", t))
    return num_tokens / len(tokens) > 0.50


def _is_equation_heavy(text: str) -> bool:
    """Skip paragraphs dominated by math symbols or garbled equation fragments."""
    if len(text) < 50:
        return False

    # High density of math/special characters
    math_chars = set("∑∏∫∂∇≈≠≤≥±×÷αβγδεζηθλμνξπρσφψω∀∃∈∉⊂⊃∪∩⟨⟩‖→←↑↓↔⊕⊗¼½¾ð")
    math_count = sum(1 for c in text if c in math_chars)
    if math_count / len(text) > 0.03:
        return True

    # High ratio of single-character tokens (broken equations)
    tokens = text.split()
    if len(tokens) > 10:
        single_char = sum(1 for t in tokens if len(t) == 1)
        if single_char / len(tokens) > 0.20:
            return True

    # High digit ratio (table data, numeric content)
    digit_count = sum(1 for c in text if c.isdigit())
    if len(text) > 100 and digit_count / len(text) > 0.30:
        return True

    return False


def is_bibliography(chunk: Chunk) -> bool:
    """Check if a chunk is from a bibliography section.

    Canonical function for filtering loaded/cached chunks. Checks both
    section name (fast path) and content patterns (fallback). Used by
    loader.py and state.py for post-hoc filtering.

    GROBID chunks: References already separated by ML model, so this
    rarely triggers. Kept as defense-in-depth.

    PyMuPDF chunks: Primary safety net for entries that bypass section-level
    detection during chunking (_find_reference_cutoff, _is_reference_section).
    """
    if chunk.section and _is_reference_section(chunk.section):
        return True
    return _is_bibliography_chunk(chunk.text)


def _split_long_text(
    text: str,
    max_chars: int,
    min_chars: int,
) -> list[str]:
    """Split a long text into smaller chunks at sentence boundaries.

    Args:
        text: Text to split
        max_chars: Maximum characters per chunk
        min_chars: Minimum characters per chunk

    Returns:
        List of text chunks
    """
    # Simple sentence splitting
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length > max_chars and current_chunk:
            # Save current chunk and start new one
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= min_chars:
                chunks.append(chunk_text)
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space

    # Don't forget the last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if len(chunk_text) >= min_chars:
            chunks.append(chunk_text)
        elif chunks:
            # Merge small final chunk with previous
            chunks[-1] = chunks[-1] + " " + chunk_text

    return chunks


def chunk_papers(
    papers: list[Paper],
    max_tokens: int = 512,
    min_chunk_length: int = 150,
    show_progress: bool = True,
) -> list[Chunk]:
    """Chunk multiple papers.

    Args:
        papers: List of Paper objects
        max_tokens: Maximum tokens per chunk
        min_chunk_length: Minimum character length for a chunk
        show_progress: Whether to show progress bar

    Returns:
        List of all Chunk objects from all papers
    """
    from tqdm import tqdm

    all_chunks: list[Chunk] = []

    iterator = papers
    if show_progress:
        iterator = tqdm(papers, desc="Chunking papers")

    for paper in iterator:
        chunks = chunk_paper(paper, max_tokens, min_chunk_length)
        all_chunks.extend(chunks)

    return all_chunks
