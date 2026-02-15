"""Download and extract full text from arXiv LaTeX sources.

Uses the arXiv e-print endpoint to download source files, then
converts LaTeX to plain text paragraphs using pylatexenc.

Rate limit: ~3 seconds between requests (arXiv policy).
"""

import gzip
import io
import re
import tarfile
import time
from typing import Optional

import requests
from tqdm import tqdm

from incite.models import Paper

ARXIV_EPRINT_URL = "https://export.arxiv.org/e-print/{arxiv_id}"
REQUEST_DELAY = 3.0  # seconds between requests (arXiv rate limit)


def _download_source(arxiv_id: str, timeout: int = 30) -> Optional[bytes]:
    """Download arXiv source for a paper.

    Args:
        arxiv_id: arXiv identifier (e.g., 'cond-mat/9803387' or '0808.0242')
        timeout: Request timeout in seconds

    Returns:
        Raw bytes of the source file, or None on failure
    """
    url = ARXIV_EPRINT_URL.format(arxiv_id=arxiv_id)
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.content
        return None
    except requests.RequestException:
        return None


def _extract_latex(content: bytes) -> Optional[str]:
    """Extract LaTeX source from downloaded arXiv content.

    Handles three formats:
    1. tar.gz archive (most common) — finds main .tex file
    2. Plain gzip'd .tex file
    3. Plain .tex file (rare)

    Args:
        content: Raw bytes from arXiv e-print download

    Returns:
        LaTeX source string, or None if extraction fails
    """
    # Skip PDFs
    if content[:5] == b"%PDF-":
        return None

    # Try tar.gz first
    try:
        tar = tarfile.open(fileobj=io.BytesIO(content), mode="r:gz")
        tex_files = [m for m in tar.getmembers() if m.name.endswith(".tex")]

        if not tex_files:
            return None

        # Find main file (the one with \begin{document})
        for tf in tex_files:
            try:
                latex = tar.extractfile(tf).read().decode("utf-8", errors="replace")
                if r"\begin{document}" in latex:
                    return latex
            except Exception:
                continue

        # Fallback: use the largest .tex file
        tex_files.sort(key=lambda x: x.size, reverse=True)
        try:
            return tar.extractfile(tex_files[0]).read().decode("utf-8", errors="replace")
        except Exception:
            return None
    except (tarfile.TarError, EOFError):
        pass

    # Try plain gzip
    try:
        latex = gzip.decompress(content).decode("utf-8", errors="replace")
        # Sanity check: should contain LaTeX commands
        if "\\" in latex:
            return latex
    except (gzip.BadGzipFile, OSError):
        pass

    # Try plain text
    try:
        latex = content.decode("utf-8", errors="replace")
        if "\\" in latex and len(latex) > 100:
            return latex
    except Exception:
        pass

    return None


def _latex_to_paragraphs(
    latex: str,
    min_paragraph_length: int = 50,
    min_alpha_ratio: float = 0.30,
) -> list[str]:
    """Convert LaTeX source to a list of clean text paragraphs.

    Uses pylatexenc for conversion, then splits on paragraph boundaries
    and filters out short/corrupted text.

    Args:
        latex: LaTeX source string
        min_paragraph_length: Minimum characters for a valid paragraph
        min_alpha_ratio: Minimum ratio of alphabetic characters

    Returns:
        List of paragraph text strings
    """
    from pylatexenc.latex2text import LatexNodes2Text

    # Strip everything before \begin{document} (preamble)
    doc_start = latex.find(r"\begin{document}")
    if doc_start >= 0:
        latex = latex[doc_start:]

    # Strip everything after \end{document}
    doc_end = latex.find(r"\end{document}")
    if doc_end >= 0:
        latex = latex[:doc_end]

    # Strip bibliography/references section
    # Common patterns: \bibliography{...}, \begin{thebibliography},
    # or explicit \section*{References}
    for pattern in [
        r"\\bibliography\{",
        r"\\begin\{thebibliography\}",
        r"\\section\*?\{References\}",
        r"\\section\*?\{Bibliography\}",
    ]:
        match = re.search(pattern, latex, re.IGNORECASE)
        if match:
            latex = latex[: match.start()]

    # Convert to text
    converter = LatexNodes2Text()
    try:
        text = converter.latex_to_text(latex)
    except Exception:
        # Fallback: crude regex-based stripping
        text = _crude_latex_strip(latex)

    # Split into paragraphs on double newlines
    raw_paragraphs = re.split(r"\n\s*\n", text)

    # Clean and filter
    paragraphs = []
    for para in raw_paragraphs:
        # Collapse internal newlines to spaces
        para = re.sub(r"\s*\n\s*", " ", para)
        para = re.sub(r"\s+", " ", para)
        para = para.strip()

        if not para:
            continue

        # Skip short paragraphs
        if len(para) < min_paragraph_length:
            continue

        # Skip corrupted text (low alpha ratio)
        alpha_count = sum(c.isalpha() for c in para)
        if alpha_count / len(para) < min_alpha_ratio:
            continue

        paragraphs.append(para)

    return paragraphs


def _crude_latex_strip(latex: str) -> str:
    """Crude fallback: strip LaTeX commands with regex.

    Used when pylatexenc fails to parse.
    """
    # Remove comments
    text = re.sub(r"(?<!\\)%.*$", "", latex, flags=re.MULTILINE)
    # Remove common environments we don't want
    for env in ["equation", "align", "eqnarray", "figure", "table", "tabular"]:
        text = re.sub(
            rf"\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}",
            "",
            text,
            flags=re.DOTALL,
        )
    # Remove \command{arg} — keep arg
    text = re.sub(r"\\(?:textbf|textit|emph|text|mathrm)\{([^}]*)\}", r"\1", text)
    # Remove \command with no arg
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})*", "", text)
    # Remove remaining braces
    text = re.sub(r"[{}]", "", text)
    # Remove $ math delimiters
    text = re.sub(r"\$[^$]*\$", "", text)
    return text


def fetch_arxiv_fulltext(
    corpus_papers: list[Paper],
    openalex_to_arxiv: dict[str, str],
    show_progress: bool = True,
    save_interval: int = 50,
    corpus_path: Optional[str] = None,
) -> dict:
    """Download and extract full text from arXiv for corpus papers.

    Updates Paper objects in-place with paragraphs and full_text.

    Args:
        corpus_papers: List of Paper objects (modified in-place)
        openalex_to_arxiv: Mapping from OpenAlex ID → arXiv ID
        show_progress: Whether to show progress bars
        save_interval: Save corpus every N papers (for crash recovery)
        corpus_path: Path to save corpus for incremental saving

    Returns:
        Stats dict
    """
    paper_by_id = {p.id: p for p in corpus_papers}

    # Filter to papers we need to process
    to_process = []
    for oa_id, arxiv_id in openalex_to_arxiv.items():
        paper = paper_by_id.get(oa_id)
        if paper and not paper.paragraphs:
            to_process.append((oa_id, arxiv_id))

    stats = {
        "total": len(corpus_papers),
        "with_arxiv_id": len(openalex_to_arxiv),
        "already_have": len(openalex_to_arxiv) - len(to_process),
        "attempted": 0,
        "extracted": 0,
        "download_failed": 0,
        "parse_failed": 0,
        "no_paragraphs": 0,
    }

    if not to_process:
        if show_progress:
            print("All papers with arXiv IDs already have paragraphs")
        return stats

    if show_progress:
        print(f"Downloading LaTeX from arXiv for {len(to_process)} papers...")
        print(f"Estimated time: {len(to_process) * REQUEST_DELAY / 60:.0f} minutes")

    iterator = to_process
    if show_progress:
        iterator = tqdm(to_process, desc="Fetching arXiv sources")

    for i, (oa_id, arxiv_id) in enumerate(iterator):
        stats["attempted"] += 1

        # Download
        content = _download_source(arxiv_id)
        if content is None:
            stats["download_failed"] += 1
            if show_progress:
                tqdm.write(f"  Download failed: {arxiv_id}")
            time.sleep(REQUEST_DELAY)
            continue

        # Extract LaTeX
        latex = _extract_latex(content)
        if latex is None:
            stats["parse_failed"] += 1
            if show_progress:
                tqdm.write(f"  Parse failed (PDF or no .tex): {arxiv_id}")
            time.sleep(REQUEST_DELAY)
            continue

        # Convert to paragraphs
        paragraphs = _latex_to_paragraphs(latex)
        if not paragraphs:
            stats["no_paragraphs"] += 1
            time.sleep(REQUEST_DELAY)
            continue

        # Update paper
        paper = paper_by_id[oa_id]
        paper.paragraphs = paragraphs
        paper.full_text = "\n\n".join(paragraphs)
        stats["extracted"] += 1

        # Incremental save
        if corpus_path and save_interval and (i + 1) % save_interval == 0:
            from incite.corpus.loader import save_corpus

            save_corpus(corpus_papers, corpus_path)
            if show_progress:
                tqdm.write(f"  Saved progress ({stats['extracted']} extracted so far)")

        # Rate limit
        time.sleep(REQUEST_DELAY)

    return stats
