"""Sentence-level chunking for finer-grained retrieval.

This module provides sentence-level chunking as an alternative to paragraph-level.
Each sentence chunk includes context via the context_text field:
- Format: "{paper_title} | {section_heading} | {previous_sentence}"

This follows Wilson Lin's approach of preserving semantic context without
requiring a trained dependency classifier.

Note: Sentence-level chunking is implemented and tested but excluded from
production scope. The spaCy dependency is heavy for distribution and sentence-
level produces ~7x more chunks, significantly increasing indexing time and
storage. Revisit when infrastructure budget allows.
"""

import re
from typing import Optional

from incite.corpus.chunking import (
    _find_reference_cutoff,
    _is_boilerplate,
    _is_corrupted_text,
    _is_figure_or_table_caption,
    _is_reference_section,
    _looks_like_heading,
    _split_into_paragraphs,
)
from incite.models import Chunk, Paper

# Lazy-loaded spaCy model for sentence segmentation
_nlp = None
_nlp_available: Optional[bool] = None


def _get_spacy_model():
    """Get spaCy model for sentence segmentation.

    Lazy-loads en_core_web_sm with only tok2vec and parser enabled.
    Returns None if spaCy is unavailable.
    """
    global _nlp, _nlp_available

    if _nlp_available is not None:
        return _nlp if _nlp_available else None

    try:
        import spacy

        _nlp = spacy.load("en_core_web_sm")
        # Only enable components needed for sentence segmentation
        _nlp.select_pipes(enable=["tok2vec", "parser"])
        _nlp_available = True
        return _nlp
    except (ImportError, OSError):
        _nlp_available = False
        return None


def _split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Split text into sentences with character offsets.

    Uses spaCy if available, otherwise falls back to regex.

    Args:
        text: Text to split

    Returns:
        List of (start_offset, end_offset, sentence_text) tuples
    """
    nlp = _get_spacy_model()

    if nlp is not None:
        doc = nlp(text)
        sentences = []
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if sent_text:
                sentences.append((sent.start_char, sent.end_char, sent_text))
        return sentences

    # Fallback: regex-based splitting
    return _split_sentences_regex(text)


def _split_sentences_regex(text: str) -> list[tuple[int, int, str]]:
    """Fallback regex-based sentence splitting.

    Less accurate than spaCy - may incorrectly split on abbreviations
    like "et al.", "Fig.", "Eq.", etc.

    Returns:
        List of (start_offset, end_offset, sentence_text) tuples
    """
    # Split on sentence-ending punctuation followed by space and capital letter
    pattern = r"(?<=[.!?])\s+(?=[A-Z])"

    sentences = []
    pos = 0
    parts = re.split(pattern, text)

    for part in parts:
        part_stripped = part.strip()
        if part_stripped:
            # Find the actual start position in original text
            start = text.find(part, pos)
            if start == -1:
                start = pos
            end = start + len(part)
            sentences.append((start, end, part_stripped))
            pos = end

    return sentences


def _build_sentence_context(
    paper: Paper,
    section: Optional[str],
    prev_sentence: Optional[str],
) -> str:
    """Build context string to prepend to sentence for embedding.

    Format: "{title} | {section} | {prev_sentence}"

    Args:
        paper: Parent paper for title
        section: Current section heading (if any)
        prev_sentence: Previous sentence text (if any)

    Returns:
        Context string for the context_text field
    """
    parts = [paper.title]
    if section:
        parts.append(section)
    if prev_sentence:
        parts.append(prev_sentence)
    return " | ".join(parts)


def _is_short_reference(text: str) -> bool:
    """Check if text is a short figure/table reference like 'See Figure 1.'"""
    # Match patterns like "See Figure 1.", "(See Fig. 2)", "Table 3 shows..."
    patterns = [
        r"^(See\s+)?(Figure|Fig\.|Table|Equation|Eq\.)\s+\d",
        r"^\([^)]{0,30}\)\.?$",  # Very short parenthetical
    ]
    for pattern in patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    return False


def chunk_paper_sentences(
    paper: Paper,
    min_chunk_length: int = 50,
    include_context: bool = True,
) -> list[Chunk]:
    """Split a paper into sentence-level chunks with context.

    Each chunk includes:
    - text: The sentence itself
    - context_text: "{title} | {section} | {prev_sentence}" (if include_context=True)
    - section: Current section heading
    - char_offset: Position in full text

    Args:
        paper: Paper object to chunk
        min_chunk_length: Minimum character length for a sentence to be kept (default 50)
        include_context: Whether to include context_text (default True)

    Returns:
        List of Chunk objects
    """
    chunks: list[Chunk] = []

    # Determine source text
    if paper.paragraphs:
        paragraphs = paper.paragraphs
    elif paper.full_text:
        paragraphs = _split_into_paragraphs(paper.full_text)
    elif paper.abstract:
        # Fallback: use abstract as single chunk
        paragraphs = [paper.abstract]
    else:
        return []  # No text available

    # Pre-scan: detect reference section by consecutive bibliography entries
    ref_cutoff = _find_reference_cutoff(paragraphs)

    # Track state across paragraphs
    char_offset = 0
    current_section: Optional[str] = None
    prev_sentence: Optional[str] = None

    for i, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue

        # Stop if we've reached the detected reference section
        if ref_cutoff is not None and i >= ref_cutoff:
            break

        # Check if this paragraph is a section heading
        if _looks_like_heading(para):
            current_section = para
            # Stop at reference/bibliography sections
            if _is_reference_section(current_section):
                break
            # Don't create chunks for headings, but update offset
            char_offset += len(para) + 2  # +2 for paragraph separator
            # Reset prev_sentence at section boundaries
            prev_sentence = None
            continue

        # Skip boilerplate paragraphs
        if _is_boilerplate(para):
            char_offset += len(para) + 2
            continue

        # Split paragraph into sentences
        sentences = _split_sentences(para)

        for start, end, sent_text in sentences:
            # Skip very short sentences
            if len(sent_text) < min_chunk_length:
                prev_sentence = sent_text  # Still use as context for next sentence
                continue

            # Skip corrupted text
            if _is_corrupted_text(sent_text):
                prev_sentence = None
                continue

            # Skip figure/table captions
            if _is_figure_or_table_caption(sent_text):
                prev_sentence = sent_text
                continue

            # Skip short references like "See Figure 1."
            if _is_short_reference(sent_text):
                prev_sentence = sent_text
                continue

            # Create chunk
            chunk_id = f"{paper.id}::chunk_{len(chunks)}"

            context_text = None
            if include_context:
                context_text = _build_sentence_context(paper, current_section, prev_sentence)

            chunks.append(
                Chunk(
                    id=chunk_id,
                    paper_id=paper.id,
                    text=sent_text,
                    section=current_section,
                    char_offset=char_offset + start,
                    context_text=context_text,
                    parent_text=para,  # Store parent paragraph for highlighted display
                )
            )

            # Update prev_sentence for next iteration
            prev_sentence = sent_text

        char_offset += len(para) + 2  # +2 for paragraph separator

    return chunks


def chunk_papers_sentences(
    papers: list[Paper],
    min_chunk_length: int = 50,
    include_context: bool = True,
    show_progress: bool = True,
) -> list[Chunk]:
    """Chunk multiple papers into sentences.

    Args:
        papers: List of Paper objects
        min_chunk_length: Minimum character length for a sentence
        include_context: Whether to include context_text
        show_progress: Whether to show progress bar

    Returns:
        List of all Chunk objects from all papers
    """
    from tqdm import tqdm

    all_chunks: list[Chunk] = []

    iterator = papers
    if show_progress:
        iterator = tqdm(papers, desc="Chunking papers (sentences)")

    for paper in iterator:
        chunks = chunk_paper_sentences(paper, min_chunk_length, include_context)
        all_chunks.extend(chunks)

    return all_chunks
