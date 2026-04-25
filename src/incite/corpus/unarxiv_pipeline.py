"""Pipeline orchestrator for processing unarXiv dataset directories.

Incrementally processes arXiv papers until a target count is reached,
fetching metadata from OpenAlex, building citation contexts, and
merging results into corpus and test set files.

The pipeline is designed to be resumable: existing corpus and test set
files are loaded at startup, and already-processed source papers are
skipped. This allows safe interruption and re-running without duplicating
work or incurring redundant OpenAlex API calls.

Output files:
    - corpus.jsonl: flat list of cited papers (Paper objects) with metadata
    - test_set.jsonl: citation contexts (CitationContext objects) linking
      source paper text to cited papers, used for evaluation

Related modules:
    - incite.corpus.unarxiv: UnarXivProcessor class (core extraction logic).
    - incite.corpus.openalex: OpenAlex API client for metadata enrichment.
    - incite.cli.data: CLI commands that invoke this pipeline.
"""

import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from incite.corpus.openalex import OpenAlexClient
from incite.corpus.unarxiv import UnarXivProcessor
from incite.models import CitationContext, Paper

logger = logging.getLogger(__name__)


def process_unarxiv_directory(
    data_dir: Path | str,
    output_corpus: Path | str,
    output_test_set: Path | str,
    openalex_email: Optional[str] = None,
    min_coverage: float = 0.3,
    min_reference_set_size: int = 15,
    skip_existing: bool = True,
    target_source_papers: int = 100,
) -> dict:
    """Process unarXiv JSONL files incrementally until target is reached.

    Processes files one at a time and stops early once we have enough
    qualifying source papers. This is much faster than scanning everything.

    Incremental strategy:
        1. Load existing corpus and test set (if skip_existing=True).
        2. Collect already-processed source paper IDs to avoid re-work.
        3. Iterate JSONL files in sorted order, processing paper by paper.
        4. For each candidate paper, fetch only the OpenAlex metadata that
           isn't already in the local cache (persists across files).
        5. Apply coverage and reference-set-size filters; skip failures.
        6. Accumulate new papers and contexts until target is reached.
        7. Merge new results with existing data and write output files.

    Merge logic:
        New corpus papers are appended to the existing list, deduplicated by
        OpenAlex ID via ``corpus_ids``. New citation contexts are appended to
        the existing test set without deduplication (each context has a unique
        ``id`` of the form ``{source_paper_id}_cite_{n}``).

    Args:
        data_dir: Directory containing unarXiv JSONL files (searched recursively).
        output_corpus: Path to output corpus.jsonl (created or updated).
        output_test_set: Path to output test_set.jsonl (created or updated).
        openalex_email: Email for OpenAlex polite pool (improves rate limits).
        min_coverage: Minimum fraction of citations with abstracts to include
            a source paper (default 0.3).
        min_reference_set_size: Minimum number of resolved references (papers
            with abstracts) to include a source paper (default 15).
        skip_existing: If True, skip papers already present in the test set
            and reuse the existing corpus cache (default True).
        target_source_papers: Stop after accumulating this many total source
            papers (existing + new). Default 100.

    Returns:
        Stats dict with keys: files_processed, papers_scanned, papers_included,
        papers_skipped_coverage, papers_skipped_ref_size, contexts_included,
        references_fetched, corpus_total, contexts_total, source_papers_total.
    """
    # Deferred import to avoid circular dependency at module load time:
    # loader imports from models, which is fine, but keeping it local
    # here mirrors the original pattern and avoids a hard top-level dep.
    from incite.corpus.loader import load_corpus, load_test_set, save_corpus, save_test_set

    data_dir = Path(data_dir)
    output_corpus = Path(output_corpus)
    output_test_set = Path(output_test_set)

    # Find all JSONL files in the directory tree, sorted for deterministic order
    jsonl_files = sorted(data_dir.glob("**/*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {data_dir}")
        return {"error": "No JSONL files found"}

    print(f"Found {len(jsonl_files)} JSONL files")

    # --- Load existing data to support incremental (resumable) processing ---
    existing_corpus: list[Paper] = []
    existing_contexts: list[CitationContext] = []
    skip_paper_ids: set[str] = set()
    corpus_ids: set[str] = set()

    if skip_existing:
        if output_corpus.exists():
            existing_corpus = load_corpus(output_corpus)
            corpus_ids = {p.id for p in existing_corpus}
            print(f"Loaded {len(existing_corpus)} existing corpus papers")

        if output_test_set.exists():
            existing_contexts = load_test_set(output_test_set)
            # Build skip set from source paper IDs recorded in each context
            skip_paper_ids = {
                ctx.source_paper_id for ctx in existing_contexts if ctx.source_paper_id
            }
            print(
                f"Loaded {len(existing_contexts)} existing contexts "
                f"from {len(skip_paper_ids)} papers"
            )

    # --- Initialize processor and shared metadata cache ---
    client = OpenAlexClient(email=openalex_email)
    processor = UnarXivProcessor(openalex_client=client)

    # papers_cache persists across all JSONL files to avoid redundant API calls.
    # Pre-populate with existing corpus papers so their metadata is free.
    papers_cache: dict[str, Paper] = {p.id: p for p in existing_corpus}

    new_papers: list[Paper] = []
    new_contexts: list[CitationContext] = []
    source_papers_added = 0

    stats: dict = {
        "files_processed": 0,
        "papers_scanned": 0,
        "papers_included": 0,
        "papers_skipped_coverage": 0,
        "papers_skipped_ref_size": 0,
        "contexts_included": 0,
        "references_fetched": 0,
    }

    # How many more source papers we need beyond what already exists
    target_remaining = target_source_papers - len(skip_paper_ids)
    print(f"Target: {target_source_papers} source papers (have {len(skip_paper_ids)} existing)")

    if target_remaining <= 0:
        print("Already have enough source papers!")
        return stats

    # --- Incremental file processing loop ---
    pbar = tqdm(jsonl_files, desc="Processing files")
    for jsonl_path in pbar:
        stats["files_processed"] += 1

        for source in processor.iter_papers(jsonl_path):
            # Skip papers we've already processed in a prior run
            if source.paper_id in skip_paper_ids:
                continue
            # Skip papers with no citation contexts (nothing useful to extract)
            if not source.citation_contexts:
                continue

            stats["papers_scanned"] += 1

            ref_ids = source.reference_openalex_ids

            # Pre-filter: skip immediately if we can't possibly meet the
            # reference-set-size threshold (fast path, no API call needed)
            if len(ref_ids) < min_reference_set_size:
                stats["papers_skipped_ref_size"] += 1
                continue

            # Fetch only the references not already in the local cache
            to_fetch = [rid for rid in ref_ids if rid not in papers_cache]
            if to_fetch:
                fetched = processor._fetch_papers_batch(to_fetch, show_progress=False)
                papers_cache.update(fetched)
                stats["references_fetched"] += len(fetched)

            # Calculate coverage: fraction of references with abstracts
            resolved = [
                rid for rid in ref_ids if rid in papers_cache and papers_cache[rid].abstract
            ]
            coverage = len(resolved) / len(ref_ids) if ref_ids else 0

            if coverage < min_coverage:
                stats["papers_skipped_coverage"] += 1
                continue

            # Post-fetch filter: recheck resolved count with actual abstracts
            if len(resolved) < min_reference_set_size:
                stats["papers_skipped_ref_size"] += 1
                continue

            # Paper qualifies — add it to the output
            stats["papers_included"] += 1
            source_papers_added += 1
            skip_paper_ids.add(source.paper_id)

            # Add referenced papers to corpus (dedup by OpenAlex ID)
            for ref_id in resolved:
                if ref_id not in corpus_ids:
                    new_papers.append(papers_cache[ref_id])
                    corpus_ids.add(ref_id)

            # Build citation contexts for this source paper.
            # cite_num is a 1-based counter used to generate unique context IDs.
            cite_num = 0
            for ctx in source.citation_contexts:
                openalex_id = ctx["openalex_id"]
                # Only include contexts where the ground-truth paper is resolved
                if openalex_id not in resolved:
                    continue

                cite_num += 1
                new_contexts.append(
                    CitationContext(
                        id=f"{source.paper_id}_cite_{cite_num}",
                        local_context=ctx["text"],
                        narrow_context=ctx.get("narrow", ""),
                        broad_context=ctx.get("broad", ""),
                        section_context=ctx["section"],
                        global_context=source.title,
                        source_paper_id=source.paper_id,
                        ground_truth_ids=[openalex_id],
                        reference_set_ids=resolved,
                        mentioned_authors=ctx.get("mentioned_authors", []),
                        mentioned_years=ctx.get("mentioned_years", []),
                    )
                )
                stats["contexts_included"] += 1

            pbar.set_postfix(
                {"sources": source_papers_added, "contexts": stats["contexts_included"]}
            )

            # Early exit once we've reached the per-run target
            if source_papers_added >= target_remaining:
                print(f"\nReached target of {target_source_papers} source papers!")
                break

        # Also break out of the file loop once target is met
        if source_papers_added >= target_remaining:
            break

    # --- Merge new results with existing data and persist ---
    merged_corpus = existing_corpus + new_papers
    merged_contexts = existing_contexts + new_contexts

    save_corpus(merged_corpus, output_corpus)
    save_test_set(merged_contexts, output_test_set)

    # Annotate stats with final totals for caller reporting
    stats["corpus_total"] = len(merged_corpus)
    stats["contexts_total"] = len(merged_contexts)
    stats["source_papers_total"] = len(skip_paper_ids)

    print("\nProcessing complete:")
    print(f"  Files processed: {stats['files_processed']}/{len(jsonl_files)}")
    print(f"  Papers scanned: {stats['papers_scanned']}")
    print(f"  Papers included: {stats['papers_included']}")
    print(f"  Papers skipped (low coverage): {stats['papers_skipped_coverage']}")
    print(f"  Papers skipped (small ref set): {stats['papers_skipped_ref_size']}")
    print(f"  References fetched: {stats['references_fetched']}")
    print(f"  New contexts: {stats['contexts_included']}")
    print(f"  Total corpus: {stats['corpus_total']}")
    print(f"  Total contexts: {stats['contexts_total']}")
    print(f"  Total source papers: {stats['source_papers_total']}")

    return stats
