"""Agent commands: agent {recommend, batch, stats, extract-pdfs}."""

import sys
from pathlib import Path

from incite.cli._shared import EMBEDDER_CHOICES


def register(subparsers):
    """Register agent commands."""
    agent_parser = subparsers.add_parser(
        "agent", help="Agent-friendly interface for programmatic testing"
    )
    agent_subparsers = agent_parser.add_subparsers(dest="agent_command", help="Agent subcommands")

    _register_recommend(agent_subparsers)
    _register_batch(agent_subparsers)
    _register_stats(agent_subparsers)
    _register_extract(agent_subparsers)

    agent_parser.set_defaults(func=cmd_agent)


def _register_recommend(subparsers):
    p = subparsers.add_parser("recommend", help="Get recommendations with JSON output")
    p.add_argument("query", type=str, help="Text to get citations for")
    p.add_argument("--top-k", "-k", type=int, default=10, help="Number of recommendations")
    p.add_argument("--json", action="store_true", help="Output as JSON (default: human-readable)")
    p.add_argument(
        "--corpus",
        type=str,
        default="data/processed/corpus.jsonl",
        help="Path to corpus JSONL file",
    )
    p.add_argument("--index", type=str, default="data/processed/index", help="Path to FAISS index")
    p.add_argument(
        "--method",
        type=str,
        choices=["neural", "bm25", "hybrid"],
        default="hybrid",
        help="Retrieval method to use (default: hybrid)",
    )
    p.add_argument(
        "--embedder",
        type=str,
        choices=EMBEDDER_CHOICES,
        default="minilm",
        help="Embedder model to use (default: minilm)",
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["paper", "paragraph"],
        default="paper",
        help="Retrieval mode: paper (title+abstract) or paragraph (PDF chunks)",
    )
    p.add_argument(
        "--chunks",
        type=str,
        default="data/processed/chunks.jsonl",
        help="Path to chunks file (for --mode paragraph)",
    )
    p.add_argument(
        "--chunk-index",
        type=str,
        default="data/processed/chunk_index",
        help="Path to chunk index (for --mode paragraph)",
    )
    p.add_argument(
        "--author-boost",
        type=float,
        default=1.0,
        help="Boost papers whose authors appear in query (1.0 = no boost, 1.2 = 20%% boost)",
    )


def _register_batch(subparsers):
    p = subparsers.add_parser("batch", help="Batch recommendations from file")
    p.add_argument("queries_file", type=str, help="File with one query per line")
    p.add_argument(
        "--top-k", "-k", type=int, default=10, help="Number of recommendations per query"
    )
    p.add_argument("--json", action="store_true", help="Output as JSON (default: summary)")
    p.add_argument(
        "--corpus",
        type=str,
        default="data/processed/corpus.jsonl",
        help="Path to corpus JSONL file",
    )
    p.add_argument("--index", type=str, default="data/processed/index", help="Path to FAISS index")
    p.add_argument(
        "--method",
        type=str,
        choices=["neural", "bm25", "hybrid"],
        default="hybrid",
        help="Retrieval method to use",
    )
    p.add_argument(
        "--embedder",
        type=str,
        choices=EMBEDDER_CHOICES,
        default="minilm",
        help="Embedder model to use",
    )
    p.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Run queries in parallel (default: True)",
    )
    p.add_argument("--no-parallel", action="store_true", help="Disable parallel execution")


def _register_stats(subparsers):
    p = subparsers.add_parser("stats", help="Get corpus statistics")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    p.add_argument(
        "--corpus",
        type=str,
        default="data/processed/corpus.jsonl",
        help="Path to corpus JSONL file",
    )


def _register_extract(subparsers):
    p = subparsers.add_parser("extract-pdfs", help="Extract PDFs for paragraph mode")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    p.add_argument(
        "--corpus",
        type=str,
        default="data/processed/corpus.jsonl",
        help="Path to corpus JSONL file",
    )
    p.add_argument("--workers", type=int, default=8, help="Number of parallel workers")


# --- Command handlers ---


def cmd_agent(args):
    """Dispatch agent subcommands."""
    if args.agent_command == "recommend":
        cmd_agent_recommend(args)
    elif args.agent_command == "batch":
        cmd_agent_batch(args)
    elif args.agent_command == "stats":
        cmd_agent_stats(args)
    elif args.agent_command == "extract-pdfs":
        cmd_agent_extract(args)
    else:
        print("Usage: incite agent {recommend|batch|stats|extract-pdfs}")


def cmd_agent_recommend(args):
    """Get recommendations with timing and optional JSON output."""

    from incite.agent import InCiteAgent

    agent = InCiteAgent.from_corpus(
        corpus_path=args.corpus,
        index_path=args.index if Path(args.index).exists() else None,
        method=args.method,
        embedder_type=args.embedder,
        mode=args.mode,
        chunks_path=args.chunks if args.mode == "paragraph" else None,
        chunk_index_path=args.chunk_index if args.mode == "paragraph" else None,
    )

    response = agent.recommend(
        query=args.query,
        k=args.top_k,
        author_boost=args.author_boost,
    )

    if args.json:
        print(response.to_json())
    else:
        print(f'\nQuery: "{args.query[:100]}{"..." if len(args.query) > 100 else ""}"\n')
        print(f"Mode: {response.mode} | Method: {response.method} | Embedder: {response.embedder}")
        print(f"Corpus size: {response.corpus_size} papers")
        print(f"Total time: {response.timing.total_ms:.1f}ms")
        print(f"  - Embed query: {response.timing.embed_query_ms:.1f}ms")
        print(f"  - Vector search: {response.timing.vector_search_ms:.1f}ms")
        if response.timing.bm25_search_ms is not None:
            print(f"  - BM25 search: {response.timing.bm25_search_ms:.1f}ms")
        if response.timing.fusion_ms is not None:
            print(f"  - Fusion: {response.timing.fusion_ms:.1f}ms")
        print()

        print(f"Top {len(response.recommendations)} recommendations:")
        for rec in response.recommendations:
            print(f"\n{rec.rank}. [{rec.score:.3f}] {rec.title}")
            if rec.authors:
                authors = ", ".join(rec.authors[:3])
                if len(rec.authors) > 3:
                    authors += " et al."
                print(f"   {authors} ({rec.year})")
            if rec.matched_paragraph:
                preview = rec.matched_paragraph[:150]
                if len(rec.matched_paragraph) > 150:
                    preview += "..."
                print(f'   Matched: "{preview}"')


def cmd_agent_batch(args):
    """Run batch recommendations from file."""
    import json

    from incite.agent import InCiteAgent

    queries_path = Path(args.queries_file)
    if not queries_path.exists():
        print(f"Error: Queries file not found: {args.queries_file}")
        sys.exit(1)

    with open(queries_path) as f:
        queries = [line.strip() for line in f if line.strip()]

    if not queries:
        print("Error: No queries found in file")
        sys.exit(1)

    print(f"Loaded {len(queries)} queries from {args.queries_file}")

    agent = InCiteAgent.from_corpus(
        corpus_path=args.corpus,
        index_path=args.index if Path(args.index).exists() else None,
        method=args.method,
        embedder_type=args.embedder,
    )

    parallel = not args.no_parallel
    responses = agent.batch_recommend(
        queries=queries,
        k=args.top_k,
        parallel=parallel,
    )

    if args.json:
        output = [r.to_dict() for r in responses]
        print(json.dumps(output, indent=2))
    else:
        total_time = sum(r.timing.total_ms for r in responses)
        avg_time = total_time / len(responses)

        print(f"\nProcessed {len(responses)} queries")
        print(f"Total time: {total_time:.1f}ms")
        print(f"Average time per query: {avg_time:.1f}ms")
        print(f"Parallel: {parallel}")
        print()

        print("Sample results (first 3):")
        for i, response in enumerate(responses[:3]):
            print(f'\n[{i + 1}] Query: "{response.query[:60]}..."')
            if response.recommendations:
                top = response.recommendations[0]
                print(f"    Top result: {top.title[:60]}...")


def cmd_agent_stats(args):
    """Get corpus statistics."""
    import json

    from incite.corpus.loader import load_corpus

    papers = load_corpus(args.corpus)

    stats = {
        "corpus_size": len(papers),
        "corpus_path": str(args.corpus),
        "papers_with_abstract": sum(1 for p in papers if p.abstract),
        "papers_with_year": sum(1 for p in papers if p.year),
        "papers_with_authors": sum(1 for p in papers if p.authors),
        "papers_with_doi": sum(1 for p in papers if p.doi),
        "papers_with_full_text": sum(1 for p in papers if p.full_text),
        "papers_with_llm_description": sum(1 for p in papers if p.llm_description),
    }

    years = [p.year for p in papers if p.year]
    if years:
        stats["year_min"] = min(years)
        stats["year_max"] = max(years)

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(f"\nCorpus: {args.corpus}")
        print(f"Total papers: {stats['corpus_size']}")
        print(f"  With abstract: {stats['papers_with_abstract']}")
        print(f"  With year: {stats['papers_with_year']}")
        print(f"  With authors: {stats['papers_with_authors']}")
        print(f"  With DOI: {stats['papers_with_doi']}")
        print(f"  With full text: {stats['papers_with_full_text']}")
        print(f"  With LLM description: {stats['papers_with_llm_description']}")
        if years:
            print(f"  Year range: {stats['year_min']} - {stats['year_max']}")


def cmd_agent_extract(args):
    """Extract PDFs for paragraph mode."""
    import json

    from incite.corpus.loader import load_corpus, save_corpus
    from incite.webapp.state import extract_and_save_pdfs

    papers = load_corpus(args.corpus)

    def progress_callback(current, total, message):
        if not args.json:
            print(f"\r{message} ({current}/{total})", end="", flush=True)

    stats = extract_and_save_pdfs(
        papers=papers,
        progress_callback=progress_callback if not args.json else None,
        max_workers=args.workers,
    )

    save_corpus(papers, args.corpus)

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print("\n\nExtraction complete:")
        print(f"  Found PDFs: {stats['found_pdfs']}")
        print(f"  Extracted: {stats['extracted']}")
        print(f"  Total papers: {stats['total']}")
        print(f"\nCorpus saved to {args.corpus}")
        if stats["extracted"] > 0:
            print("\nYou can now use paragraph mode:")
            print("  incite agent recommend 'query' --mode paragraph")
