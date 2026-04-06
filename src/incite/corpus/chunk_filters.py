"""Chunk quality filters and content classifiers.

Predicate functions that identify chunks to exclude from the retrieval index:
bibliography entries, boilerplate text, figure captions, acknowledgments,
corrupted OCR output, equation-heavy passages, and table data.

Used by chunk_paper() in chunking.py during the paragraph chunking pipeline.
Each predicate returns True if the chunk should be EXCLUDED.

Related modules:
    - incite.corpus.chunking: Main chunking pipeline (consumer of these filters).
    - incite.corpus.grobid_chunking: GROBID-specific chunking (may use is_bibliography).
"""

import re
from typing import Optional


def _is_reference_section(section_name: Optional[str]) -> bool:
    """Check if section name indicates a references/bibliography section.

    Returns True (exclude) for section headings that indicate the document has
    transitioned into its reference list. Handles common variants including
    numbered headings ("7. References"), compound names ("References and Notes"),
    and alternative terminology ("Literature Cited", "Works Cited").

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

    Returns True (exclude) if the text matches 3+ bibliographic signals and is
    short enough to be a typical reference entry. Used as a backup when
    section-based detection fails (e.g., stale cache, inline citations, or
    papers with non-standard section names).
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

    Returns True (exclude) for chunks that are individual reference list entries.
    Section-level detection (_is_reference_section, _find_reference_cutoff) catches
    the bulk of references, but individual entries leak through when:
    - The "References" heading isn't detected (non-standard formatting, stale cache)
    - Fewer than 3 consecutive entries exist (appendix refs, footnotes)
    - Entries are in the first 60% of the document (outside the scan window)

    Detection uses two strategies:
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

    Returns True (exclude) if less than min_alpha_ratio of characters are alphabetic.
    Corrupted text typically arises from PDF encoding failures, heavy OCR noise,
    or formula-only paragraphs that contain no readable prose.
    """
    if not text:
        return True
    alpha_count = sum(c.isalpha() for c in text)
    return alpha_count / len(text) < min_alpha_ratio


def _find_reference_cutoff(paragraphs: list[str]) -> Optional[int]:
    """Find where the reference section starts by detecting consecutive bib entries.

    Returns the index of the first bibliography paragraph if 3+ consecutive
    paragraphs in the last 40% of the document match bibliography patterns.
    Returns None if no reference section is detected.

    Used as a pre-scan fallback when no "References" heading is found (e.g.,
    heading detection missed the section marker or the document uses a
    non-standard section name).
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
    """Check if text is a figure or table caption.

    Returns True (exclude) for captions that start with standard figure/table
    labels followed by a number. Captions add retrieval noise without conveying
    substantive citable content, and their short length already filters most via
    min_chunk_length, but this catches longer multi-sentence captions.
    """
    return bool(
        re.match(
            r"^(Figure|Fig\.|Table|Supplementary Figure|Supplementary Table|Extended Data)\s+\d",
            text,
            re.IGNORECASE,
        )
    )


def _is_boilerplate(text: str) -> bool:
    """Check if text is journal boilerplate, copyright, or download/HTML artifacts.

    Returns True (exclude) for publisher-injected text that is not part of the
    original paper content: copyright notices, download metadata, access banners,
    cookie consent, and journal header/footer lines. These appear in HTML-scraped
    and JSTOR-style PDFs and dilute retrieval quality.
    """
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
    """Check if text is an acknowledgment, funding, or author contribution paragraph.

    Returns True (exclude) for boilerplate end-of-paper sections that describe
    funding sources, contributor roles, competing interests, or supplementary
    material availability. These are non-citable administrative content. Only
    called for paragraphs in the last 30% of the document to avoid false
    positives on content that discusses funding research.
    """
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
    """Check if text is table data with a high ratio of numbers and short tokens.

    Returns True (exclude) when more than 50% of whitespace-delimited tokens are
    purely numeric (digits, percentages, ±, ranges). Table data cells produce
    retrieval noise without substantive prose content.
    """
    tokens = text.split()
    if len(tokens) < 5:
        return False
    # Count number-like tokens (digits, percentages, ±, ranges)
    num_tokens = sum(1 for t in tokens if re.match(r"^[\d.,±%<>≤≥$€£]+$", t))
    return num_tokens / len(tokens) > 0.50


def _is_equation_heavy(text: str) -> bool:
    """Check if text is dominated by math symbols or garbled equation fragments.

    Returns True (exclude) for paragraphs that are primarily mathematical content
    rather than natural language prose. Three signals are checked:
    1. High density of Unicode math characters (>3% of chars)
    2. High ratio of single-character tokens indicating broken equations (>20%)
    3. High digit ratio suggesting numeric/table content (>30%)
    """
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
