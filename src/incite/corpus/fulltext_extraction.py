"""Extract full text from unarXiv for corpus papers.

Given a mapping of OpenAlex ID → arXiv ID, finds the corresponding papers
in the unarXiv dataset and extracts their body_text paragraphs.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from incite.models import Paper


def _arxiv_id_to_prefix(arxiv_id: str) -> str:
    """Extract YYMM prefix from an arXiv ID.

    Handles both new-style (0808.0242, 2103.13020) and
    old-style (gr-qc/9808081, hep-th/0301150) IDs.
    """
    if "." in arxiv_id and "/" not in arxiv_id:
        return arxiv_id.split(".")[0]  # '0808'
    elif "/" in arxiv_id:
        numeric = arxiv_id.split("/")[1]
        return numeric[:4]  # '9808'
    return arxiv_id[:4]


def _find_files_for_prefixes(data_dir: Path, prefixes: set[str]) -> dict[str, list[Path]]:
    """Find JSONL files matching given YYMM prefixes.

    Returns dict mapping prefix → list of JSONL file paths.
    """
    prefix_to_files: dict[str, list[Path]] = defaultdict(list)

    for jsonl_path in sorted(data_dir.rglob("arXiv_src_*.jsonl")):
        file_prefix = jsonl_path.stem.split("_")[2]  # 'arXiv_src_0808_001' → '0808'
        if file_prefix in prefixes:
            prefix_to_files[file_prefix].append(jsonl_path)

    return prefix_to_files


def _extract_paragraphs_from_body_text(body_text: list[dict]) -> list[str]:
    """Extract clean paragraph texts from unarXiv body_text.

    Filters out reference sections, very short paragraphs, and
    corrupted text. Preserves section structure.

    Args:
        body_text: List of paragraph dicts from unarXiv format

    Returns:
        List of paragraph text strings
    """
    paragraphs = []
    in_references = False

    for para in body_text:
        section = para.get("section", "")
        text = para.get("text", "").strip()

        if not text:
            continue

        # Stop at references section
        section_lower = section.lower().strip()
        # Strip leading numbering
        section_stripped = re.sub(r"^[\d.]+\s*", "", section_lower)
        if section_stripped in ("references", "bibliography", "works cited", "literature cited"):
            in_references = True
        if in_references:
            continue

        # Replace citation markers with placeholder
        # unarXiv cite_spans reference bib entries but we want clean text
        cite_spans = para.get("cite_spans", [])
        if cite_spans:
            # Sort spans in reverse order to preserve offsets during replacement
            sorted_spans = sorted(cite_spans, key=lambda s: s.get("start", 0), reverse=True)
            for span in sorted_spans:
                start = span.get("start", 0)
                end = span.get("end", start)
                if 0 <= start < end <= len(text):
                    text = text[:start] + text[end:]

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Skip very short paragraphs
        if len(text) < 50:
            continue

        # Skip corrupted text (low alpha ratio)
        alpha_count = sum(c.isalpha() for c in text)
        if alpha_count / len(text) < 0.30:
            continue

        paragraphs.append(text)

    return paragraphs


def extract_fulltext_from_unarxiv(
    corpus_papers: list[Paper],
    openalex_to_arxiv: dict[str, str],
    data_dir: Path,
    show_progress: bool = True,
) -> dict[str, int]:
    """Extract full text from unarXiv for corpus papers.

    Updates Paper objects in-place with paragraphs and full_text.

    Args:
        corpus_papers: List of Paper objects (will be modified in-place)
        openalex_to_arxiv: Mapping from OpenAlex ID → arXiv ID
        data_dir: Root unarXiv data directory
        show_progress: Whether to show progress bars

    Returns:
        Stats dict with counts
    """
    # Build paper lookup
    paper_by_id = {p.id: p for p in corpus_papers}

    # Build arxiv_id → openalex_id reverse mapping
    arxiv_to_openalex: dict[str, str] = {
        arxiv_id: oa_id for oa_id, arxiv_id in openalex_to_arxiv.items() if oa_id in paper_by_id
    }

    if not arxiv_to_openalex:
        print("No corpus papers have arXiv IDs — nothing to extract")
        return {"total": len(corpus_papers), "extracted": 0, "skipped": 0}

    # Group target arXiv IDs by prefix
    prefix_to_arxiv_ids: dict[str, set[str]] = defaultdict(set)
    for arxiv_id in arxiv_to_openalex:
        prefix = _arxiv_id_to_prefix(arxiv_id)
        prefix_to_arxiv_ids[prefix].add(arxiv_id)

    # Find matching files
    all_prefixes = set(prefix_to_arxiv_ids.keys())
    prefix_to_files = _find_files_for_prefixes(data_dir, all_prefixes)

    files_to_scan = []
    for prefix, files in prefix_to_files.items():
        for f in files:
            files_to_scan.append((f, prefix_to_arxiv_ids[prefix]))

    if show_progress:
        print(f"Searching {len(files_to_scan)} files for {len(arxiv_to_openalex)} papers...")

    extracted = 0
    remaining = set(arxiv_to_openalex.keys())

    iterator = files_to_scan
    if show_progress:
        iterator = tqdm(files_to_scan, desc="Scanning files")

    for jsonl_path, target_ids in iterator:
        # Only look for IDs we haven't found yet
        targets = target_ids & remaining
        if not targets:
            continue

        with open(jsonl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                paper_id = data.get("paper_id", "")

                if paper_id not in targets:
                    continue

                # Found a target paper — extract body text
                body_text = data.get("body_text", [])
                paragraphs = _extract_paragraphs_from_body_text(body_text)

                if paragraphs:
                    oa_id = arxiv_to_openalex[paper_id]
                    paper = paper_by_id[oa_id]
                    paper.paragraphs = paragraphs
                    paper.full_text = "\n\n".join(paragraphs)
                    extracted += 1

                remaining.discard(paper_id)

                if not remaining:
                    break

        if not remaining:
            break

    if show_progress:
        print(f"\nExtracted full text for {extracted} papers")
        if remaining:
            print(f"Not found in unarXiv files: {len(remaining)}")

    return {
        "total": len(corpus_papers),
        "with_arxiv_id": len(arxiv_to_openalex),
        "extracted": extracted,
        "not_found": len(remaining),
    }
