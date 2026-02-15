"""LLM-powered commands: enrich-llm, generate-synthetic, reformulate, enrich-chunks."""

import os
import sys
from pathlib import Path

from incite.cli._shared import EMBEDDER_CHOICES
from incite.utils import DEFAULT_LLM_MODEL


def register(subparsers):
    """Register LLM commands."""
    _register_enrich_llm(subparsers)
    _register_generate_synthetic(subparsers)
    _register_reformulate(subparsers)
    _register_enrich_chunks(subparsers)


def _register_enrich_llm(subparsers):
    p = subparsers.add_parser("enrich-llm", help="Generate LLM descriptions for corpus papers")
    p.add_argument(
        "--corpus",
        "-c",
        type=str,
        default="data/processed/corpus.jsonl",
        help="Path to corpus JSONL file (read and updated in-place)",
    )
    p.add_argument(
        "--api-key", type=str, help="Anthropic API key (or use ANTHROPIC_API_KEY env var)"
    )
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help=f"Model to use (default: {DEFAULT_LLM_MODEL})",
    )
    p.add_argument(
        "--workers", type=int, default=5, help="Number of parallel API calls (default: 5)"
    )
    p.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-generate descriptions for papers that already have them",
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Use Anthropic Batch API (50%% cheaper, async processing)",
    )
    p.set_defaults(func=cmd_enrich_llm)


def _register_generate_synthetic(subparsers):
    p = subparsers.add_parser(
        "generate-synthetic", help="Generate synthetic citation contexts from Zotero library"
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/processed/synthetic_test_set.jsonl",
        help="Output JSONL path (default: data/processed/synthetic_test_set.jsonl)",
    )
    p.add_argument(
        "--db",
        type=str,
        default="data/processed/synthetic_contexts.db",
        help="SQLite database path (default: data/processed/synthetic_contexts.db)",
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Use Anthropic Batch API (50%% cheaper, async processing)",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Process only first N papers (for testing)"
    )
    p.add_argument(
        "--ref-set-size", type=int, default=50, help="K-NN reference set size (default: 50)"
    )
    p.add_argument(
        "--export-only", action="store_true", help="Skip generation, export existing DB to JSONL"
    )
    p.add_argument(
        "--embedder",
        type=str,
        choices=EMBEDDER_CHOICES,
        default="minilm",
        help="Embedder for reference sets (default: minilm)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help=f"LLM model (default: {DEFAULT_LLM_MODEL})",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel API calls for threaded mode (default: 5)",
    )
    p.add_argument(
        "--api-key", type=str, help="Anthropic API key (or use ANTHROPIC_API_KEY env var)"
    )
    p.add_argument(
        "--difficulty",
        type=str,
        choices=["standard", "moderate"],
        default="standard",
        help="standard (5 easy) or moderate (3 hard) contexts",
    )
    p.set_defaults(func=cmd_generate_synthetic)


def _register_reformulate(subparsers):
    p = subparsers.add_parser("reformulate", help="Reformulate test set queries using LLM (HyDE)")
    p.add_argument(
        "--test-set",
        type=str,
        default="data/processed/test_set.jsonl",
        help="Path to test set JSONL file (read and updated in-place)",
    )
    p.add_argument(
        "--api-key", type=str, help="Anthropic API key (or use ANTHROPIC_API_KEY env var)"
    )
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help=f"Model to use (default: {DEFAULT_LLM_MODEL})",
    )
    p.add_argument(
        "--source-scale",
        type=str,
        choices=["narrow", "broad"],
        default="narrow",
        help="Context scale to use as LLM input (default: narrow)",
    )
    p.add_argument(
        "--workers", type=int, default=5, help="Number of parallel API calls (default: 5)"
    )
    p.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-reformulate queries that already have reformulated text",
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Use Anthropic Batch API (50%% cheaper, async processing)",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Limit to first N queries (for quick testing)"
    )
    p.set_defaults(func=cmd_reformulate)


def _register_enrich_chunks(subparsers):
    p = subparsers.add_parser("enrich-chunks", help="Generate LLM context for chunks")
    p.add_argument(
        "--corpus",
        "-c",
        type=str,
        default="data/processed/corpus.jsonl",
        help="Path to corpus JSONL file",
    )
    p.add_argument(
        "--chunks",
        type=str,
        default="data/processed/chunks.jsonl",
        help="Path to chunks JSONL file (will create if not exists)",
    )
    p.add_argument(
        "--api-key", type=str, help="Anthropic API key (or use ANTHROPIC_API_KEY env var)"
    )
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help=f"Model to use (default: {DEFAULT_LLM_MODEL})",
    )
    p.add_argument(
        "--no-skip", action="store_true", help="Re-generate context for chunks that already have it"
    )
    p.add_argument(
        "--estimate", action="store_true", help="Just estimate cost, don't actually process"
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Use Anthropic Batch API (50%% cheaper, async processing)",
    )
    p.set_defaults(func=cmd_enrich_chunks)


# --- Command handlers ---


def cmd_enrich_llm(args):
    """Generate LLM descriptions for corpus papers."""
    from incite.corpus.loader import load_corpus, save_corpus

    print(f"Loading corpus from {args.corpus}...")
    papers = load_corpus(args.corpus)
    print(f"Loaded {len(papers)} papers")

    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")

    if args.batch:
        from incite.corpus.llm_enrichment import enrich_corpus_batch

        stats = enrich_corpus_batch(
            papers=papers,
            api_key=api_key,
            model=args.model,
            skip_existing=not args.no_skip,
        )
    else:
        from incite.corpus.llm_enrichment import enrich_corpus

        stats = enrich_corpus(
            papers=papers,
            api_key=api_key,
            model=args.model,
            max_workers=args.workers,
            skip_existing=not args.no_skip,
        )

    save_corpus(papers, args.corpus)

    print("\nEnrichment complete:")
    print(f"  Total papers:        {stats['total']}")
    print(f"  Newly enriched:      {stats['enriched']}")
    print(f"  Skipped (existing):  {stats['skipped_existing']}")
    print(f"  Failed:              {stats['failed']}")
    print(f"\nCorpus saved to {args.corpus}")
    print("Run 'incite index' to rebuild the search index with enriched descriptions.")


def cmd_generate_synthetic(args):
    """Generate synthetic citation contexts from Zotero library."""
    from incite.corpus.synthetic_contexts import (
        build_reference_sets,
        export_to_jsonl,
        generate_synthetic_batch,
        generate_synthetic_threaded,
    )
    from incite.corpus.synthetic_db import SyntheticDB
    from incite.corpus.zotero_reader import find_zotero_data_dir, read_zotero_library

    zotero_dir = find_zotero_data_dir()
    if not zotero_dir:
        print("Error: Could not find Zotero data directory.")
        print("Expected ~/Zotero with zotero.sqlite inside.")
        sys.exit(1)

    print(f"Reading Zotero library from {zotero_dir}...")
    papers = read_zotero_library(zotero_dir, show_progress=True)
    print(f"Loaded {len(papers)} papers")

    papers_with_abs = [p for p in papers if p.abstract and len(p.abstract) >= 50]
    print(f"Papers with abstracts (>= 50 chars): {len(papers_with_abs)}")

    if args.limit:
        papers_with_abs = papers_with_abs[: args.limit]
        print(f"Limited to first {args.limit} papers")

    db = SyntheticDB(args.db)

    if not args.export_only:
        api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")

        if args.batch:
            print(f"\nGenerating {args.difficulty} contexts via Batch API ({args.model})...")
            stats = generate_synthetic_batch(
                papers=papers_with_abs,
                db=db,
                api_key=api_key,
                model=args.model,
                difficulty=args.difficulty,
            )
        else:
            print(
                f"\nGenerating {args.difficulty} contexts via threaded API "
                f"({args.model}, {args.workers} workers)..."
            )
            stats = generate_synthetic_threaded(
                papers=papers_with_abs,
                db=db,
                api_key=api_key,
                model=args.model,
                max_workers=args.workers,
                difficulty=args.difficulty,
            )

        print("\nGeneration complete:")
        print(f"  Total papers:        {stats['total']}")
        print(f"  Processed:           {stats['to_process']}")
        print(f"  Generated:           {stats['generated']}")
        print(f"  Contexts created:    {stats['contexts_created']}")
        print(f"  Skipped (existing):  {stats['skipped_existing']}")
        print(f"  Failed:              {stats['failed']}")

        print(f"\nBuilding {args.ref_set_size}-NN reference sets...")
        ref_sets = build_reference_sets(
            papers=papers,
            embedder_type=args.embedder,
            k=args.ref_set_size,
        )
        db.insert_reference_sets(ref_sets)
    else:
        print("Export-only mode: skipping generation")
        ref_sets = db.get_all_reference_sets()
        if not ref_sets:
            print("Warning: No reference sets in DB. Run without --export-only first.")

    output_path = Path(args.output)
    print(f"\nExporting to {output_path}...")
    num_exported = export_to_jsonl(db, ref_sets, output_path)

    db_stats = db.stats()
    print("\nFinal summary:")
    print(f"  Contexts in DB:      {db_stats['total_contexts']}")
    print(f"  Papers in DB:        {db_stats['total_papers']}")
    print(f"  By type:             {db_stats['by_type']}")
    print(f"  Papers with ref sets: {db_stats['papers_with_reference_sets']}")
    print(f"  Exported to JSONL:   {num_exported}")
    print(f"\nOutput: {output_path}")
    print(f"Database: {args.db}")
    print(f"\nTo evaluate: incite evaluate --test-set {args.output} --method hybrid")

    db.close()


def cmd_reformulate(args):
    """Reformulate test set queries using LLM (HyDE)."""
    from incite.corpus.loader import load_test_set, save_test_set

    print(f"Loading test set from {args.test_set}...")
    all_contexts = load_test_set(args.test_set)
    print(f"Loaded {len(all_contexts)} queries")

    if args.limit:
        to_process = all_contexts[: args.limit]
        print(f"Processing first {len(to_process)} queries (--limit {args.limit})")
    else:
        to_process = all_contexts

    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")

    if args.batch:
        from incite.corpus.query_reformulation import reformulate_queries_batch

        stats = reformulate_queries_batch(
            contexts=to_process,
            api_key=api_key,
            model=args.model,
            source_scale=args.source_scale,
            skip_existing=not args.no_skip,
        )
    else:
        from incite.corpus.query_reformulation import reformulate_queries

        stats = reformulate_queries(
            contexts=to_process,
            api_key=api_key,
            model=args.model,
            source_scale=args.source_scale,
            max_workers=args.workers,
            skip_existing=not args.no_skip,
        )

    save_test_set(all_contexts, args.test_set)

    print("\nReformulation complete:")
    print(f"  Total queries:       {stats['total']}")
    print(f"  Newly reformulated:  {stats['reformulated']}")
    print(f"  Skipped (existing):  {stats['skipped_existing']}")
    print(f"  Failed:              {stats['failed']}")
    print(f"\nTest set saved to {args.test_set}")
    print("Run 'incite evaluate --scale reformulated' to evaluate.")


def cmd_enrich_chunks(args):
    """Generate LLM context for chunks (Anthropic contextual retrieval)."""
    from incite.corpus.chunking import chunk_papers
    from incite.corpus.contextual_enrichment import (
        enrich_chunks_sync,
        estimate_enrichment_cost,
    )
    from incite.corpus.loader import load_chunks, load_corpus, save_chunks

    print(f"Loading corpus from {args.corpus}...")
    papers = load_corpus(args.corpus)
    paper_dict = {p.id: p for p in papers}
    print(f"Loaded {len(papers)} papers")

    chunks_path = Path(args.chunks)
    if chunks_path.exists():
        print(f"Loading existing chunks from {args.chunks}...")
        chunks = load_chunks(args.chunks)
    else:
        print("Creating chunks from papers...")
        chunks = chunk_papers(papers, show_progress=True)
        save_chunks(chunks, args.chunks)

    print(f"Total chunks: {len(chunks)}")

    if args.estimate:
        estimate = estimate_enrichment_cost(papers, chunks, args.model)
        print("\nCost estimate:")
        print(f"  Papers: {estimate['total_papers']}")
        print(f"  Chunks: {estimate['total_chunks']}")
        print(f"  Avg chunks per paper: {estimate['avg_chunks_per_paper']:.1f}")
        print(f"  Estimated document tokens: {estimate['estimated_doc_tokens']:,}")
        print(f"  Estimated chunk tokens: {estimate['estimated_chunk_tokens']:,}")
        print(f"  Estimated output tokens: {estimate['estimated_output_tokens']:,}")
        print("\nEstimated cost breakdown:")
        print(f"  Cache write: ${estimate['cache_write_cost']:.4f}")
        print(f"  Cached reads: ${estimate['cached_read_cost']:.4f}")
        print(f"  Chunk input: ${estimate['chunk_input_cost']:.4f}")
        print(f"  Output: ${estimate['output_cost']:.4f}")
        print(f"  Total: ${estimate['total_estimated_cost']:.2f}")
        return

    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    if args.batch:
        from incite.corpus.contextual_enrichment import enrich_chunks_batch

        print(f"\nEnriching chunks with {args.model} (Batch API)...")
        stats = enrich_chunks_batch(
            papers=paper_dict,
            chunks=chunks,
            api_key=api_key,
            model=args.model,
            skip_existing=not args.no_skip,
        )
    else:
        print(f"\nEnriching chunks with {args.model}...")
        stats = enrich_chunks_sync(
            papers=paper_dict,
            chunks=chunks,
            api_key=api_key,
            model=args.model,
            skip_existing=not args.no_skip,
        )

    save_chunks(chunks, args.chunks)

    print("\nEnrichment complete:")
    print(f"  Total chunks:    {stats['total']}")
    print(f"  Enriched:        {stats['enriched']}")
    print(f"  Skipped:         {stats['skipped']}")
    print(f"  Failed:          {stats['failed']}")
    print(f"\nChunks saved to {args.chunks}")
    print("Run 'incite index-chunks' to build the chunk index.")
