"""Chunking module for splitting papers into paragraph-level chunks.

This module provides the main chunking pipeline: splitting paper text into
paragraph-level Chunk objects for retrieval indexing. Supports three input
paths (pre-structured HTML paragraphs, PDF full_text, abstract fallback) and
applies a sequence of quality filters to exclude non-content paragraphs.

Public API:
    chunk_paper: Chunk a single Paper into a list of Chunk objects.
    chunk_papers: Batch-chunk multiple papers with a progress bar.
    is_bibliography: Check if a cached/loaded Chunk is from a bibliography section.

Related modules:
    - incite.corpus.chunk_filters: All filter predicates (bibliography, boilerplate,
      captions, acknowledgments, corrupted text, equations, table data).
    - incite.corpus.grobid_chunking: GROBID-specific XML-aware chunking pipeline.
    - incite.corpus.loader: Loads and post-filters chunks from the cache.
"""

import re
from typing import Optional

from incite.corpus.chunk_filters import (
    _find_reference_cutoff,
    _is_acknowledgment_text,
    _is_bibliography_chunk,
    _is_boilerplate,
    _is_corrupted_text,
    _is_equation_heavy,
    _is_figure_or_table_caption,
    _is_reference_section,
    _is_table_data,
    _looks_like_bibliography_entry,  # noqa: F401 — re-export for backward compat
)
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
    page_numbers: list[int | None] | None = None,
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
        page_numbers: Optional list of 1-indexed page numbers parallel to paragraphs.
            When splitting long paragraphs, sub-chunks inherit the parent's page number.

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

        # Look up page number for this paragraph
        page_num = page_numbers[i] if page_numbers and i < len(page_numbers) else None

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
                        page_number=page_num,
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
                    page_number=page_num,
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
