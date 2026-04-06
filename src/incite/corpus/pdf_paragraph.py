"""PDF text cleaning and paragraph detection heuristics.

Post-processing utilities for raw PDF text extraction: cleaning OCR artifacts,
detecting paragraph boundaries, identifying section headings, and filtering
front matter (author affiliations, correspondence info, etc.).

Related modules:
    - incite.corpus.pdf_extractor: Main PDF extraction (consumer of these utilities).
    - incite.corpus.chunking: Higher-level chunking that operates on extracted paragraphs.
"""

import re


def _clean_text(text: str) -> str:
    """Clean extracted PDF text by removing common artifacts.

    Applies three heuristics:
    1. Normalizes all whitespace sequences to a single space.
    2. Strips lone page number lines (digits only).
    3. Removes hyphenation at line breaks (re-joins split words).

    Args:
        text: Raw text block extracted from a PDF span/block.

    Returns:
        Cleaned text with whitespace normalized and artifacts removed.
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
    """Check if text looks like a section heading.

    Uses two strategies:
    1. Pattern matching against known heading formats (numbered sections,
       Roman numerals, well-known section names like Abstract/Introduction).
    2. All-caps short text with more than one word (e.g. "RELATED WORK").

    Also applies negative filters to reject page metadata that shares
    superficial features with headings (e.g. "414 B. Rabe" — a page number
    followed by an abbreviated author name).

    Args:
        text: A single text block extracted from the PDF.

    Returns:
        True if the text matches heading heuristics, False otherwise.
    """
    # Too short to be a meaningful heading (avoid artifacts like "2 A")
    if len(text.strip()) < 5:
        return False

    # Reject patterns that look like page metadata: "414 B. Rabe", "123 Author"
    # These are page numbers followed by short author names, not section headings.
    # Pattern: 2+ digit number, space, uppercase initial + dot, space, word.
    if re.match(r"^\d{2,}\s+[A-Z]\.\s+\w+$", text):
        return False
    # Reject "414 B. Rabe" style with middle initial (optional dot after initial)
    if re.match(r"^\d{2,}\s+[A-Z]\.?\s+[A-Z][a-z]+$", text):
        return False

    # Common heading patterns — ordered from most specific to most general:
    # - Numbered sections: "1. Introduction" (requires lowercase after the cap to
    #   distinguish from "1. B" abbreviation noise) or "1. INTRODUCTION" (all-caps word)
    # - Roman numerals: "I. Background", "IV. Results"
    # - Standard academic section names (case-insensitive)
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

    # Short text that's all/mostly capitalized (but must have >1 word to avoid
    # single-word noise like "ABSTRACT" appearing as body text)
    if len(text) < 60 and text.upper() == text and len(text.split()) > 1:
        return True

    return False


def _is_front_matter(text: str) -> bool:
    """Check if a paragraph looks like front matter.

    Front matter includes author affiliations, email addresses, ORCID identifiers,
    and correspondence notices that typically appear before the abstract on the
    first page of an academic paper.

    Args:
        text: A paragraph of extracted PDF text.

    Returns:
        True if the paragraph matches front matter patterns, False otherwise.
    """
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

    Only filters paragraphs that appear before the first real section heading
    (Abstract, Introduction, etc.) AND match front matter patterns. Paragraphs
    after the first heading are never filtered.

    If no section headings are found, no filtering is applied.

    Args:
        paragraphs: List of extracted paragraph strings.
        section_headings: List of detected section heading strings (used to
            locate the first body section boundary).

    Returns:
        Filtered list with front matter paragraphs removed.
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


def _filter_front_matter_with_pages(
    paragraphs: list[str],
    paragraph_pages: list[int | None],
    section_headings: list[str],
) -> tuple[list[str], list[int | None]]:
    """Remove front matter paragraphs, keeping paragraph_pages in sync.

    Identical filtering logic to `_filter_front_matter`, but also filters the
    parallel `paragraph_pages` list so paragraph-to-page mappings stay correct
    after removal.

    Args:
        paragraphs: List of extracted paragraph strings.
        paragraph_pages: Parallel list of page numbers (1-indexed) for each
            paragraph. May be shorter than paragraphs if some pages are unknown.
        section_headings: List of detected section heading strings.

    Returns:
        Tuple of (filtered_paragraphs, filtered_pages) with front matter removed
        from both lists.
    """
    if not paragraphs:
        return paragraphs, paragraph_pages

    # Find where body content starts (first section heading)
    body_start = 0
    for i, para in enumerate(paragraphs):
        if para in section_headings or _looks_like_heading(para):
            body_start = i
            break
    else:
        return paragraphs, paragraph_pages

    filtered_paras = []
    filtered_pages = []
    for i, para in enumerate(paragraphs):
        if i < body_start and _is_front_matter(para):
            continue
        filtered_paras.append(para)
        filtered_pages.append(paragraph_pages[i] if i < len(paragraph_pages) else None)

    return filtered_paras, filtered_pages


def _starts_new_paragraph(text: str, current_paragraph: list[str]) -> bool:
    """Heuristically determine if text starts a new paragraph.

    A new paragraph is signalled when:
    - There is no current paragraph (first block), OR
    - The new text starts with an uppercase letter AND the previous block
      ended with a sentence-terminal punctuation mark (., ?, !, :).

    Args:
        text: The incoming text block to evaluate.
        current_paragraph: The list of text blocks accumulated so far in the
            current paragraph.

    Returns:
        True if text should begin a new paragraph, False if it continues the
        current one.
    """
    if not current_paragraph:
        return True

    # Starts with common paragraph indicators
    if re.match(r"^[A-Z]", text):
        last_text = current_paragraph[-1] if current_paragraph else ""
        # Previous block ended with period/question mark
        if last_text.rstrip().endswith((".", "?", "!", ":")):
            return True

    return False
