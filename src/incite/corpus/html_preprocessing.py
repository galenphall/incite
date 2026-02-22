"""HTML-specific text preprocessing for structured extraction.

Cleans HTML-extracted text and converts structured sections into
(paragraphs, section_headings) pairs for the chunking pipeline.
"""

from __future__ import annotations

import re

# Inline citation markers commonly left in HTML-extracted text
_CITATION_MARKER_RE = re.compile(
    r"\[(\d+(?:[,\s]*\d+)*(?:\s*[-–]\s*\d+)?)\]"  # [1], [1,2], [1-3]
)
_SUPERSCRIPT_DIGITS_RE = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+")

# HTML artifacts
_NBSP_RE = re.compile(r"[\u00a0\u200b\u200c\u200d\ufeff]")


def _clean_html_paragraph(text: str) -> str:
    """Clean a single paragraph extracted from HTML."""
    # Replace non-breaking spaces and zero-width chars with regular spaces
    text = _NBSP_RE.sub(" ", text)
    # Strip inline citation markers: [1], [1,2], [1-3]
    text = _CITATION_MARKER_RE.sub("", text)
    # Strip superscript digit characters (often used as footnote refs)
    text = _SUPERSCRIPT_DIGITS_RE.sub("", text)
    # Collapse multiple spaces
    text = re.sub(r"  +", " ", text)
    # Clean up space before punctuation left by citation removal
    text = re.sub(r" ([.,;:!?])", r"\1", text)
    return text.strip()


def preprocess_html_text(
    full_text: str | None,
    structured_text: dict | None,
) -> tuple[list[str], list[str | None]]:
    """Clean HTML-extracted text and return (paragraphs, section_headings).

    If structured_text is available, uses section boundaries directly.
    Otherwise falls back to splitting full_text on double newlines.
    Returns parallel lists: paragraphs[i] belongs to sections[i].
    """
    paragraphs: list[str] = []
    sections: list[str | None] = []

    if structured_text and isinstance(structured_text.get("sections"), list):
        for section in structured_text["sections"]:
            heading = section.get("heading") or None
            for para_text in section.get("paragraphs", []):
                cleaned = _clean_html_paragraph(para_text)
                if cleaned:
                    paragraphs.append(cleaned)
                    sections.append(heading)
    elif full_text:
        # Fallback: split on double newlines
        raw_paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
        for para in raw_paragraphs:
            cleaned = _clean_html_paragraph(para)
            if cleaned:
                paragraphs.append(cleaned)
                sections.append(None)

    return paragraphs, sections
