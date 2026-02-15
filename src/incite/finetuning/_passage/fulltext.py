"""Generate passage-level training data from full-text corpora (unarXiv / S2ORC).

Supports two corpus formats:
  - **unarXiv**: body_text is a list of {section, text} paragraph dicts.
  - **S2ORC v2** (Datasets API): content.text is a single string with
    content.annotations containing span-based paragraph/sectionheader offsets.

Both are converted to Paper+Chunk objects and fed into the existing passage
generation pipeline (select_passages, generate_passage_contexts_batch, etc.).

For passage generation we only need the paper's metadata + body_text --
no cite_spans, bib resolution, or OpenAlex IDs needed.
"""

import json
import re
from pathlib import Path
from typing import Iterator, Optional

from incite.models import Chunk, Paper, format_paper_metadata_prefix

from .generation import SKIP_SECTIONS

# Extra patterns beyond clean_citation_markers -- fulltext-specific markup
_FULLTEXT_CLEANUP = [
    re.compile(r"\{\{cite:[a-f0-9]*\}?\}?"),  # {{cite:hash}}
    re.compile(r"\{\{formula:[a-f0-9\-]*\}?\}?"),  # {{formula:uuid}}
    re.compile(r"\{\{ref:[a-f0-9\-]*\}?\}?"),  # {{ref:...}}
    re.compile(r"\{\{table:[a-f0-9\-]*\}?\}?"),  # {{table:...}}
    re.compile(r"\{\{figure:[a-f0-9\-]*\}?\}?"),  # {{figure:...}}
]

# Reference section patterns (paragraph text signals)
_REF_SECTION = re.compile(
    r"^(references|bibliography|works cited|references and notes)\s*$",
    re.IGNORECASE,
)


def _clean_fulltext(text: str) -> str:
    """Remove fulltext markup (cite/formula/ref/table/figure tags)."""
    result = text
    for pat in _FULLTEXT_CLEANUP:
        result = pat.sub(" ", result)
    # Normalize whitespace
    result = re.sub(r"\s+", " ", result).strip()
    return result


def _extract_authors(data: dict) -> list[str]:
    """Extract author names from unarXiv or S2ORC metadata.

    unarXiv: metadata.authors_parsed = [["Last", "First", ""], ...]
    S2ORC:   authors = [{"first": "...", "last": "..."}, ...]
    """
    # unarXiv format
    meta = data.get("metadata", {})
    authors_parsed = meta.get("authors_parsed")
    if authors_parsed and isinstance(authors_parsed, list):
        names = []
        for entry in authors_parsed:
            if isinstance(entry, list) and len(entry) >= 2:
                last, first = entry[0], entry[1]
                if first and last:
                    names.append(f"{first} {last}")
                elif last:
                    names.append(last)
        if names:
            return names

    # S2ORC format
    authors = data.get("authors")
    if authors and isinstance(authors, list):
        names = []
        for a in authors:
            if isinstance(a, dict):
                first = a.get("first", "")
                last = a.get("last", "")
                if first and last:
                    names.append(f"{first} {last}")
                elif last:
                    names.append(last)
        if names:
            return names

    # Fallback: metadata.authors as comma-separated string
    authors_str = meta.get("authors", "")
    if authors_str and isinstance(authors_str, str):
        return [a.strip() for a in authors_str.split(",") if a.strip()]

    return []


def _extract_abstract(data: dict) -> str:
    """Extract abstract text from unarXiv or S2ORC metadata."""
    # unarXiv top-level abstract is a dict: {"section": "Abstract", "text": "..."}
    abstract_field = data.get("abstract")
    if isinstance(abstract_field, dict):
        return abstract_field.get("text", "").strip()
    if isinstance(abstract_field, str) and abstract_field.strip():
        return abstract_field.strip()

    # Fallback: metadata.abstract (string)
    meta = data.get("metadata", {})
    meta_abstract = meta.get("abstract", "")
    if isinstance(meta_abstract, str):
        return meta_abstract.strip()

    return ""


def _extract_title(data: dict) -> str:
    """Extract title from unarXiv or S2ORC metadata."""
    meta = data.get("metadata", {})
    title = meta.get("title", "")
    if title:
        return title.strip()
    # S2ORC top-level title
    return data.get("title", "").strip()


def _extract_year(data: dict) -> Optional[int]:
    """Extract publication year from metadata."""
    meta = data.get("metadata", {})
    # Try update_date field (unarXiv)
    update_date = meta.get("update_date", "")
    if update_date and len(update_date) >= 4:
        try:
            return int(update_date[:4])
        except ValueError:
            pass
    # Try year field (S2ORC)
    year = data.get("year") or meta.get("year")
    if year is not None:
        try:
            return int(year)
        except (ValueError, TypeError):
            pass
    return None


def _is_reference_section(section: str) -> bool:
    """Check if section name indicates a reference/bibliography section."""
    return bool(_REF_SECTION.match(section.strip()))


def _is_s2orc_v2(data: dict) -> bool:
    """Detect S2ORC v2 (Datasets API) format: has content + corpusid."""
    return "content" in data and "corpusid" in data


def _parse_s2orc_annotation(annotations: dict, key: str) -> list[dict]:
    """Parse a JSON-encoded annotation list from S2ORC v2."""
    raw = annotations.get(key)
    if raw and isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, list):
        return raw
    return []


def _safe_span(text: str, span: dict) -> str:
    """Extract text from a span dict, handling non-int offsets gracefully."""
    try:
        start = int(span["start"])
        end = int(span["end"])
        return text[start:end].strip()
    except (KeyError, ValueError, TypeError):
        return ""


def _convert_s2orc_v2(data: dict) -> dict | None:
    """Convert S2ORC v2 span-based record to unarXiv-compatible format.

    S2ORC v2 stores the full paper as a single string (content.text) with
    span-based annotations (content.annotations.paragraph, sectionheader, etc.).
    This converts it to the body_text=[{section, text}] format that
    parse_fulltext_record() expects.

    Returns:
        unarXiv-compatible dict, or None if the record lacks usable content.
    """
    content = data.get("content", {})
    if not isinstance(content, dict):
        return None

    text = content.get("text", "")
    if not text or len(text) < 500:
        return None

    annotations = content.get("annotations", {})
    if not annotations:
        return None

    # Extract title
    titles = _parse_s2orc_annotation(annotations, "title")
    title = _safe_span(text, titles[0]) if titles else ""
    if not title:
        return None

    # Extract abstract
    abstracts = _parse_s2orc_annotation(annotations, "abstract")
    abstract = _safe_span(text, abstracts[0]) if abstracts else ""
    if not abstract or len(abstract) < 50:
        return None

    # Extract authors
    author_spans = _parse_s2orc_annotation(annotations, "author")
    authors_parsed = []
    for a in author_spans:
        name = _safe_span(text, a)
        if name and len(name) > 1:
            parts = name.split()
            if len(parts) >= 2:
                # Store as [Last, First, ""] to match unarXiv format
                authors_parsed.append([parts[-1], " ".join(parts[:-1]), ""])
            elif parts:
                authors_parsed.append([parts[0], "", ""])

    # Build section header lookup: sorted by start offset
    section_spans = _parse_s2orc_annotation(annotations, "sectionheader")
    section_spans = [s for s in section_spans if isinstance(s.get("start"), (int, float))]
    section_spans.sort(key=lambda s: int(s["start"]))

    # Build body_text from paragraph spans
    para_spans = _parse_s2orc_annotation(annotations, "paragraph")
    para_spans = [p for p in para_spans if isinstance(p.get("start"), (int, float))]
    para_spans.sort(key=lambda p: int(p["start"]))

    body_text = []
    sec_idx = 0
    current_section = ""

    for para in para_spans:
        p_start = int(para["start"])
        para_text = _safe_span(text, para)
        if not para_text:
            continue

        # Advance section pointer: find the last sectionheader before this paragraph
        while sec_idx < len(section_spans) and int(section_spans[sec_idx]["start"]) <= p_start:
            current_section = _safe_span(text, section_spans[sec_idx])
            sec_idx += 1

        body_text.append(
            {
                "section": current_section,
                "text": para_text,
            }
        )

    if not body_text:
        return None

    # Build unarXiv-compatible record
    return {
        "paper_id": str(data.get("corpusid", "")),
        "metadata": {
            "title": title,
            "authors_parsed": authors_parsed,
        },
        "abstract": abstract,
        "body_text": body_text,
    }


def parse_fulltext_record(
    data: dict,
    min_chunks: int = 5,
    min_chunk_length: int = 150,
) -> tuple[Paper, list[Chunk]] | None:
    """Parse a raw JSONL record into Paper + Chunks.

    Handles both unarXiv and S2ORC formats transparently.

    Args:
        data: Raw JSON record (one JSONL line)
        min_chunks: Minimum usable chunks for a paper to be included
        min_chunk_length: Minimum character length for a chunk after cleaning

    Returns:
        (Paper, list[Chunk]) or None if paper has too few usable chunks
    """
    # Auto-detect and convert S2ORC v2 format
    if _is_s2orc_v2(data):
        converted = _convert_s2orc_v2(data)
        if converted is None:
            return None
        data = converted

    # Extract metadata
    title = _extract_title(data)
    if not title:
        return None

    abstract = _extract_abstract(data)
    if not abstract:
        return None

    paper_id = data.get("paper_id", "") or data.get("corpusid", "") or ""
    if not paper_id:
        # Generate from title hash as fallback
        paper_id = f"ft_{hash(title) & 0xFFFFFFFF:08x}"

    paper = Paper(
        id=str(paper_id),
        title=title,
        abstract=abstract,
        authors=_extract_authors(data),
        year=_extract_year(data),
    )

    # Parse body_text into chunks
    body_text = data.get("body_text", [])
    if not body_text:
        return None

    chunks: list[Chunk] = []
    chunk_idx = 0

    # Compute metadata prefix once per paper (same for all chunks)
    context_text = format_paper_metadata_prefix(
        title=title,
        author_lastnames=paper.author_lastnames,
        year=paper.year,
        journal=paper.journal,
    )

    for para in body_text:
        if not isinstance(para, dict):
            continue

        section = para.get("section") or ""
        text = para.get("text", "")
        if not text:
            continue

        # Skip reference/acknowledgment sections
        section_lower = section.lower().strip()
        if section_lower in SKIP_SECTIONS or _is_reference_section(section):
            continue

        # Clean fulltext markup
        cleaned = _clean_fulltext(text)
        if len(cleaned) < min_chunk_length:
            continue

        chunk = Chunk(
            id=f"{paper.id}::chunk_{chunk_idx}",
            paper_id=paper.id,
            text=cleaned,
            section=section,
            context_text=context_text,
        )
        chunks.append(chunk)
        chunk_idx += 1

    if len(chunks) < min_chunks:
        return None

    return paper, chunks


def iter_fulltext_papers(
    data_dir: Path,
    target_papers: int = 5000,
    min_chunks: int = 5,
    show_progress: bool = True,
) -> Iterator[tuple[Paper, list[Chunk]]]:
    """Iterate over full-text papers from JSONL files.

    Works for both unarXiv and S2ORC directory layouts.

    Args:
        data_dir: Directory containing JSONL files (searched recursively)
        target_papers: Stop after this many valid papers
        min_chunks: Minimum chunks per paper
        show_progress: Show progress counter

    Yields:
        (Paper, list[Chunk]) tuples
    """
    data_dir = Path(data_dir)
    jsonl_files = sorted(data_dir.glob("**/*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {data_dir}")
        return

    print(f"Found {len(jsonl_files)} JSONL files in {data_dir}")

    yielded = 0
    scanned = 0
    skipped = 0

    for jsonl_path in jsonl_files:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                scanned += 1
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                result = parse_fulltext_record(data, min_chunks=min_chunks)
                if result is None:
                    skipped += 1
                    continue

                yield result
                yielded += 1

                if show_progress and yielded % 100 == 0:
                    print(
                        f"  Papers: {yielded}/{target_papers} | "
                        f"Scanned: {scanned} | Skipped: {skipped}"
                    )

                if yielded >= target_papers:
                    print(
                        f"Reached target: {yielded} papers (scanned {scanned}, skipped {skipped})"
                    )
                    return

    print(f"Exhausted all files: {yielded} papers (scanned {scanned}, skipped {skipped})")


def generate_passage_data(
    data_dir: Path,
    target_papers: int = 5000,
    max_per_paper: int = 3,
    output_dir: Path = Path("data/finetuning"),
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    dev_fraction: float = 0.1,
    eval_fraction: float = 0.05,
    dry_run: bool = False,
    show_progress: bool = True,
    output_prefix: Optional[str] = None,
) -> dict:
    """Full pipeline: parse full-text papers, select passages, generate via LLM.

    Orchestrates:
      1. iter_fulltext_papers() -> scan papers
      2. select_passages() -> pick diverse passages
      3. generate_passage_contexts_batch() -> Haiku Batch API
      4. contexts_to_training_examples() -> TrainingExamples
      5. split_passage_data() -> train/dev/eval splits

    Args:
        data_dir: Directory with JSONL files (unarXiv or S2ORC)
        target_papers: Number of papers to process
        max_per_paper: Maximum passages per paper for LLM generation
        output_dir: Directory for output files
        api_key: Anthropic API key (or ANTHROPIC_API_KEY env)
        model: LLM model to use (default from utils)
        dev_fraction: Fraction for dev split
        eval_fraction: Fraction for eval split
        dry_run: If True, show stats without calling API
        show_progress: Show progress
        output_prefix: Prefix for output filenames (default: derived from data_dir name).
            E.g. prefix="s2orc" produces s2orc_passage_train.jsonl.

    Returns:
        Stats dict with processing summary
    """
    from incite.utils import DEFAULT_LLM_MODEL

    from .generation import (
        contexts_to_training_examples,
        generate_passage_contexts_batch,
        select_passages,
        split_passage_data,
    )

    if model is None:
        model = DEFAULT_LLM_MODEL

    output_dir = Path(output_dir)

    # Derive output prefix from data directory if not specified
    if output_prefix is None:
        # e.g. "data/raw/unarxiv" -> "unarxiv", "data/raw/s2orc" -> "s2orc"
        output_prefix = Path(data_dir).name.lower().replace(" ", "_")

    # Build output paths with prefix to prevent clobbering across sources
    train_path = output_dir / f"{output_prefix}_passage_train.jsonl"
    dev_path = output_dir / f"{output_prefix}_passage_dev.jsonl"
    eval_path = output_dir / f"{output_prefix}_passage_test_set.jsonl"

    # Refuse to overwrite existing files
    for p in [train_path, dev_path, eval_path]:
        if p.exists():
            count = sum(1 for line in open(p) if line.strip())
            print(f"\nERROR: {p} already exists ({count} examples).")
            print("Delete it first or use --output-prefix to choose a different name.")
            return {"error": f"Output file already exists: {p}"}

    # Step 1: Scan papers
    print(f"\n=== Step 1: Scanning full-text papers from {data_dir} ===")
    papers: list[Paper] = []
    chunks_by_paper: dict[str, list[Chunk]] = {}

    for paper, chunks in iter_fulltext_papers(
        data_dir, target_papers=target_papers, show_progress=show_progress
    ):
        papers.append(paper)
        chunks_by_paper[paper.id] = chunks

    total_chunks = sum(len(c) for c in chunks_by_paper.values())
    print(f"Collected {len(papers)} papers with {total_chunks} total chunks")

    if not papers:
        return {"error": "No valid papers found", "papers": 0}

    # Step 2: Select passages
    print(f"\n=== Step 2: Selecting diverse passages (max {max_per_paper}/paper) ===")
    all_chunks = []
    for pid, chunks in chunks_by_paper.items():
        all_chunks.extend(chunks)

    selected = select_passages(all_chunks, max_per_paper=max_per_paper)
    papers_with_passages = len({c.paper_id for c in selected})

    # Section distribution
    section_counts: dict[str, int] = {}
    for c in selected:
        sec = (c.section or "Unknown").strip()
        section_counts[sec] = section_counts.get(sec, 0) + 1

    print(f"  Papers with chunks:    {len(papers)}")
    print(f"  Passages selected:     {len(selected)}")
    print(f"  Papers represented:    {papers_with_passages}")
    print(f"  Avg per paper:         {len(selected) / max(1, papers_with_passages):.1f}")
    avg_length = sum(len(c.text) for c in selected) / max(1, len(selected))
    print(f"  Avg passage length:    {avg_length:.0f} chars")
    print("  Top sections:")
    for sec, count in sorted(section_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {sec}: {count}")

    stats: dict = {
        "papers_scanned": len(papers),
        "total_chunks": total_chunks,
        "passages_selected": len(selected),
        "papers_with_passages": papers_with_passages,
    }

    if dry_run:
        print("\nDry run complete. Use without --dry-run to generate contexts.")
        stats["dry_run"] = True
        return stats

    # Step 3: Generate via Batch API
    print("\n=== Step 3: Generating citation contexts via Batch API ===")
    papers_by_id = {p.id: p for p in papers}
    contexts, gen_stats = generate_passage_contexts_batch(
        papers=papers,
        chunks_by_paper=chunks_by_paper,
        api_key=api_key,
        model=model,
        max_per_paper=max_per_paper,
    )
    stats.update(gen_stats)

    print(f"  Passages processed:  {gen_stats['generated']}")
    print(f"  Contexts created:    {gen_stats['contexts_created']}")
    print(f"  Failed:              {gen_stats['failed']}")

    if not contexts:
        print("No contexts generated. Exiting.")
        stats["error"] = "No contexts generated"
        return stats

    # Step 4: Convert to training examples
    print("\n=== Step 4: Converting to training examples ===")
    examples = contexts_to_training_examples(contexts, papers_by_id, chunks_by_paper)
    # Tag source so these are distinguishable from Zotero-based passage data
    for ex in examples:
        ex.source = "fulltext_passage"
    print(f"  Training examples: {len(examples)}")
    stats["training_examples"] = len(examples)

    # Step 5: Split and save
    print("\n=== Step 5: Splitting and saving ===")
    train, dev, eval_set = split_passage_data(
        examples,
        dev_fraction=dev_fraction,
        eval_fraction=eval_fraction,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex.to_dict()) + "\n")

    with open(dev_path, "w") as f:
        for ex in dev:
            f.write(json.dumps(ex.to_dict()) + "\n")

    from incite.evaluation.passage_metrics import save_passage_test_set

    save_passage_test_set(eval_set, eval_path)

    stats["train_examples"] = len(train)
    stats["dev_examples"] = len(dev)
    stats["eval_examples"] = len(eval_set)

    print("\nOutput files:")
    print(f"  Train: {train_path} ({len(train)} examples)")
    print(f"  Dev:   {dev_path} ({len(dev)} examples)")
    print(f"  Eval:  {eval_path} ({len(eval_set)} examples)")

    return stats
