"""GROBID-based chunking for papers with PDF sources.

This module provides an alternative chunking strategy that uses GROBID
for ML-based structure detection. Unlike regex-based chunking, GROBID:

1. Uses trained ML models to detect section boundaries (~90% accuracy)
2. Separates references into a dedicated back-matter section (no regex needed)
3. Properly handles figures, tables, and formulas
4. Works with complex document layouts

Requires GROBID service running:
    docker run --rm -p 8070:8070 grobid/grobid:0.8.0

Falls back to standard paragraph chunking for papers without PDF sources.
"""

import logging
from pathlib import Path
from typing import Optional

from incite.corpus.chunking import _build_paper_metadata_prefix
from incite.corpus.chunking import chunk_paper as paragraph_chunk_paper
from incite.corpus.grobid import GROBIDClient, GROBIDResult
from incite.models import Chunk, Paper

logger = logging.getLogger(__name__)


def chunk_paper_grobid(
    paper: Paper,
    client: Optional[GROBIDClient] = None,
    max_tokens: int = 512,
    min_chunk_length: int = 100,
) -> list[Chunk]:
    """Chunk a paper using GROBID for structure detection.

    If the paper has a source_file (PDF path) and GROBID is available,
    uses GROBID for extraction. Otherwise falls back to paragraph chunking.

    Args:
        paper: Paper object (should have source_file for GROBID path)
        client: Optional GROBIDClient instance (will create one if needed)
        max_tokens: Maximum tokens per chunk (approximate)
        min_chunk_length: Minimum character length for a chunk

    Returns:
        List of Chunk objects
    """
    # Check if we can use GROBID
    pdf_path = paper.source_file
    if not pdf_path or not Path(pdf_path).exists():
        # No PDF available - fall back to paragraph chunking
        return paragraph_chunk_paper(paper, max_tokens, min_chunk_length)

    # Create client if needed
    if client is None:
        client = GROBIDClient()

    # Check GROBID availability
    if not client.is_available():
        logger.warning("GROBID not available, falling back to paragraph chunking")
        return paragraph_chunk_paper(paper, max_tokens, min_chunk_length)

    # Extract with GROBID
    try:
        result = client.extract_pdf(pdf_path)
    except Exception as e:
        logger.warning(f"GROBID extraction failed for {paper.id}: {e}")
        return paragraph_chunk_paper(paper, max_tokens, min_chunk_length)

    return _grobid_result_to_chunks(
        paper=paper,
        result=result,
        max_tokens=max_tokens,
        min_chunk_length=min_chunk_length,
    )


def _grobid_result_to_chunks(
    paper: Paper,
    result: GROBIDResult,
    max_tokens: int = 512,
    min_chunk_length: int = 100,
) -> list[Chunk]:
    """Convert GROBID extraction result to chunks.

    GROBID separates references into the back-matter, so we never
    create chunks from bibliography entries.

    Args:
        paper: Original paper object
        result: GROBID extraction result
        max_tokens: Maximum tokens per chunk
        min_chunk_length: Minimum character length for a chunk

    Returns:
        List of Chunk objects (no bibliography chunks)
    """
    from incite.corpus.chunking import _is_corrupted_text, _split_long_text

    chunks: list[Chunk] = []
    metadata_prefix = _build_paper_metadata_prefix(paper)
    max_chars = max_tokens * 4  # Rough token-to-char ratio
    char_offset = 0

    # Add abstract as first chunk if present
    if result.abstract and len(result.abstract) >= min_chunk_length:
        if not _is_corrupted_text(result.abstract):
            chunk_id = f"{paper.id}::chunk_{len(chunks)}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    paper_id=paper.id,
                    text=result.abstract,
                    section="Abstract",
                    char_offset=char_offset,
                    context_text=metadata_prefix,
                )
            )
        char_offset += len(result.abstract) + 2

    # Process each section (GROBID already excluded references)
    for section in result.sections:
        section_heading = section.heading

        # Split section text into paragraphs
        paragraphs = section.text.split("\n\n")

        for para in paragraphs:
            para = para.strip()

            # Skip short chunks
            if len(para) < min_chunk_length:
                char_offset += len(para) + 2
                continue

            # Skip corrupted text
            if _is_corrupted_text(para):
                char_offset += len(para) + 2
                continue

            # Split long paragraphs if needed
            if len(para) > max_chars:
                sub_chunks = _split_long_text(para, max_chars, min_chunk_length)
                for sub_text in sub_chunks:
                    chunk_id = f"{paper.id}::chunk_{len(chunks)}"
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            paper_id=paper.id,
                            text=sub_text,
                            section=section_heading,
                            char_offset=char_offset,
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
                        section=section_heading,
                        char_offset=char_offset,
                        context_text=metadata_prefix,
                    )
                )

            char_offset += len(para) + 2

    return chunks


def chunk_papers_grobid(
    papers: list[Paper],
    max_tokens: int = 512,
    min_chunk_length: int = 100,
    show_progress: bool = True,
    grobid_url: str = "http://localhost:8070",
) -> list[Chunk]:
    """Chunk multiple papers using GROBID.

    Papers with PDF source files are processed through GROBID.
    Papers without PDFs fall back to paragraph chunking.

    Args:
        papers: List of Paper objects
        max_tokens: Maximum tokens per chunk
        min_chunk_length: Minimum character length for a chunk
        show_progress: Whether to show progress bar
        grobid_url: URL of GROBID service

    Returns:
        List of all Chunk objects from all papers
    """
    from tqdm import tqdm

    # Create shared client
    client = GROBIDClient(url=grobid_url)

    # Check availability once
    grobid_available = client.is_available()
    if not grobid_available:
        logger.warning(
            "GROBID service not available. "
            "Start with: docker run --rm -p 8070:8070 grobid/grobid:0.8.0"
        )

    all_chunks: list[Chunk] = []
    grobid_count = 0
    fallback_count = 0

    iterator = papers
    if show_progress:
        iterator = tqdm(papers, desc="Chunking papers (GROBID)")

    for paper in iterator:
        pdf_path = paper.source_file

        # Use GROBID if available and paper has PDF
        if grobid_available and pdf_path and Path(pdf_path).exists():
            chunks = chunk_paper_grobid(
                paper,
                client=client,
                max_tokens=max_tokens,
                min_chunk_length=min_chunk_length,
            )
            grobid_count += 1
        else:
            # Fall back to paragraph chunking
            chunks = paragraph_chunk_paper(paper, max_tokens, min_chunk_length)
            fallback_count += 1

        all_chunks.extend(chunks)

    if show_progress:
        logger.info(
            f"Chunked {len(papers)} papers: "
            f"{grobid_count} via GROBID, {fallback_count} via paragraph fallback"
        )

    return all_chunks
