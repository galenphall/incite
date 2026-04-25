"""Helper functions for the InCiteAgent.

Contains utilities that support the agent but don't belong on the class itself.

Related modules:
    - incite.agent: InCiteAgent class.
    - incite.corpus.pdf_extractor: PDF extraction implementation.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)


def _extract_pdfs_for_papers(papers: list, max_workers: int = 8) -> None:
    """Extract PDF text into paper.paragraphs/full_text using PyMuPDF.

    Modifies papers in-place. Papers without source_file or without
    pymupdf installed are silently skipped.

    Threading strategy: uses a ThreadPoolExecutor with up to max_workers
    threads to extract PDFs in parallel. Each paper's PDF is processed
    independently; results are collected via as_completed() to allow
    partial progress even when some extractions fail.

    Error handling: individual PDF extraction failures are caught and
    logged at DEBUG level so that one bad PDF does not abort the whole
    batch. Import errors (pymupdf not installed) short-circuit immediately
    with an INFO log, leaving all papers unchanged.

    Args:
        papers: List of Paper objects to process. Modified in-place —
            paper.full_text and paper.paragraphs are set for papers whose
            PDFs were successfully extracted.
        max_workers: Maximum number of parallel extraction threads.
    """
    try:
        import fitz  # noqa: F401
    except ImportError:
        logger.info("pymupdf not installed, skipping PDF extraction")
        return

    from incite.corpus.pdf_extractor import extract_pdf_text

    papers_with_pdfs = [p for p in papers if p.source_file and Path(p.source_file).exists()]
    if not papers_with_pdfs:
        return

    def _extract_single(paper):
        result = extract_pdf_text(paper.source_file)
        return paper.id, result.full_text or "", result.paragraphs or []

    results_map: dict[str, tuple[str, list[str]]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_extract_single, p): p for p in papers_with_pdfs}
        for future in as_completed(futures):
            try:
                paper_id, full_text, paragraphs = future.result()
                if full_text:
                    results_map[paper_id] = (full_text, paragraphs)
            except Exception:
                logger.debug("PDF extraction failed for %s", futures[future].id, exc_info=True)

    # Update papers in-place
    for paper in papers:
        if paper.id in results_map:
            paper.full_text, paper.paragraphs = results_map[paper.id]

    logger.info(
        "PDF extraction: %d/%d papers had extractable text",
        len(results_map),
        len(papers_with_pdfs),
    )
