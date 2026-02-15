"""Data acquisition commands: fetch, enrich, expand, fetch-fulltext, extract-pdfs."""

import os
import sys
from pathlib import Path


def register(subparsers):
    """Register data commands."""
    _register_fetch(subparsers)
    _register_enrich(subparsers)
    _register_expand(subparsers)
    _register_fetch_fulltext(subparsers)
    _register_extract_pdfs(subparsers)


def _register_fetch(subparsers):
    p = subparsers.add_parser("fetch", help="Fetch papers from Semantic Scholar")
    p.add_argument("query", type=str, help="Search query")
    p.add_argument("--limit", "-n", type=int, default=100, help="Number of papers to fetch")
    p.add_argument(
        "--output", "-o", type=str, default="data/processed/corpus.jsonl", help="Output path"
    )
    p.add_argument("--api-key", type=str, help="Semantic Scholar API key")
    p.set_defaults(func=cmd_fetch)


def _register_enrich(subparsers):
    p = subparsers.add_parser("enrich", help="Enrich BibTeX file with metadata from APIs")
    p.add_argument("bibtex_file", type=str, help="Path to .bib file")
    p.add_argument(
        "--output", "-o", type=str, default="data/processed/corpus.jsonl", help="Output corpus path"
    )
    p.add_argument(
        "--s2-key",
        type=str,
        help="Semantic Scholar API key (or use SEMANTIC_SCHOLAR_API_KEY env var)",
    )
    p.add_argument(
        "--email", type=str, help="Email for OpenAlex polite pool (or use OPENALEX_EMAIL env var)"
    )
    p.add_argument("--no-skip", action="store_true", help="Re-process entries already in corpus")
    p.set_defaults(func=cmd_enrich)


def _register_expand(subparsers):
    p = subparsers.add_parser("expand", help="Expand test set from unarXiv data")
    p.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/unarxiv",
        help="Directory containing unarXiv JSONL files",
    )
    p.add_argument(
        "--corpus", "-c", type=str, default="data/processed/corpus.jsonl", help="Output corpus path"
    )
    p.add_argument(
        "--test-set",
        "-t",
        type=str,
        default="data/processed/test_set.jsonl",
        help="Output test set path",
    )
    p.add_argument(
        "--min-coverage",
        type=float,
        default=0.3,
        help="Minimum citation coverage to include a paper (0.0-1.0)",
    )
    p.add_argument(
        "--min-ref-size",
        type=int,
        default=15,
        help="Minimum reference set size to include a paper (default: 15)",
    )
    p.add_argument(
        "--email", type=str, help="Email for OpenAlex polite pool (or use OPENALEX_EMAIL env var)"
    )
    p.add_argument("--no-skip", action="store_true", help="Re-process papers already in test set")
    p.add_argument(
        "--target",
        type=int,
        default=100,
        help="Target number of source papers (default: 100, stops early when reached)",
    )
    p.set_defaults(func=cmd_expand)


def _register_fetch_fulltext(subparsers):
    p = subparsers.add_parser("fetch-fulltext", help="Download full text from arXiv LaTeX sources")
    p.add_argument(
        "--corpus",
        "-c",
        type=str,
        default="data/processed/corpus.jsonl",
        help="Path to corpus JSONL file (updated in-place with paragraphs)",
    )
    p.add_argument(
        "--mapping",
        type=str,
        default="data/processed/openalex_to_arxiv_all.json",
        help="Path to OpenAlex->arXiv ID mapping JSON",
    )
    p.add_argument(
        "--source",
        "-s",
        type=str,
        choices=["arxiv", "unarxiv"],
        default="arxiv",
        help="Source for full text: arxiv (download LaTeX) or unarxiv (local files)",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/unarxiv",
        help="Directory containing unarXiv JSONL files (for --source unarxiv)",
    )
    p.add_argument("--limit", type=int, default=None, help="Limit to first N papers (for testing)")
    p.set_defaults(func=cmd_fetch_fulltext)


def _register_extract_pdfs(subparsers):
    p = subparsers.add_parser("extract-pdfs", help="Extract text from PDFs in Zotero storage")
    p.add_argument(
        "--zotero-dir",
        type=str,
        default="~/Zotero/storage",
        help="Zotero storage directory (default: ~/Zotero/storage)",
    )
    p.add_argument(
        "--corpus",
        "-c",
        type=str,
        default="data/processed/corpus.jsonl",
        help="Path to corpus JSONL file (will be updated with full_text)",
    )
    p.set_defaults(func=cmd_extract_pdfs)


# --- Command handlers ---


def cmd_fetch(args):
    """Fetch papers from Semantic Scholar."""
    from incite.corpus.loader import save_corpus
    from incite.corpus.semantic_scholar import SemanticScholarClient

    api_key = args.api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    client = SemanticScholarClient(api_key=api_key)

    print(f"Searching for: {args.query}")
    papers = client.search_papers(args.query, limit=args.limit)

    print(f"Found {len(papers)} papers with abstracts")

    save_corpus(papers, args.output)
    print(f"Saved to {args.output}")


def cmd_enrich(args):
    """Enrich BibTeX file with metadata from APIs."""
    from incite.corpus.enrichment import enrich_bibtex_to_corpus

    s2_key = args.s2_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    email = args.email or os.getenv("OPENALEX_EMAIL")

    stats = enrich_bibtex_to_corpus(
        bibtex_path=args.bibtex_file,
        output_path=args.output,
        s2_api_key=s2_key,
        openalex_email=email,
        skip_existing=not args.no_skip,
    )

    print("\nEnrichment complete:")
    print(f"  Total papers in corpus: {stats['total']}")
    print(f"  New papers added:       {stats['new']}")
    print(f"  Skipped (already in):   {stats['skipped']}")
    print(f"  With DOI:               {stats['doi_found']}")
    print(f"  With abstract:          {stats['abstract_found']}")
    print(f"  API-enriched:           {stats['api_enriched']}")


def cmd_expand(args):
    """Expand test set from unarXiv data."""
    from incite.corpus.unarxiv import process_unarxiv_directory

    email = args.email or os.getenv("OPENALEX_EMAIL")

    stats = process_unarxiv_directory(
        data_dir=args.data_dir,
        output_corpus=args.corpus,
        output_test_set=args.test_set,
        openalex_email=email,
        min_coverage=args.min_coverage,
        min_reference_set_size=args.min_ref_size,
        skip_existing=not args.no_skip,
        target_source_papers=args.target,
    )

    if "error" not in stats:
        print("\nRun 'incite index' to rebuild the search index with the new corpus.")


def cmd_fetch_fulltext(args):
    """Download full text from arXiv LaTeX sources."""
    import json

    from incite.corpus.loader import load_corpus, save_corpus

    print(f"Loading corpus from {args.corpus}...")
    papers = load_corpus(args.corpus)
    print(f"Loaded {len(papers)} papers")

    already_have = sum(1 for p in papers if p.paragraphs)
    print(f"Papers already with paragraphs: {already_have}")

    mapping_path = Path(args.mapping)
    if not mapping_path.exists():
        print(f"No mapping file found at {mapping_path}")
        print("Run: python scripts/assess_fulltext_coverage.py first")
        sys.exit(1)

    with open(mapping_path) as f:
        openalex_to_arxiv = json.load(f)
    print(f"Loaded {len(openalex_to_arxiv)} OpenAlex->arXiv mappings")

    if args.limit:
        limited = {}
        count = 0
        paper_ids_with_paras = {p.id for p in papers if p.paragraphs}
        for oa_id, arxiv_id in openalex_to_arxiv.items():
            if oa_id not in paper_ids_with_paras:
                limited[oa_id] = arxiv_id
                count += 1
                if count >= args.limit:
                    break
        openalex_to_arxiv = limited
        print(f"Limited to {len(openalex_to_arxiv)} papers")

    if args.source == "arxiv":
        from incite.corpus.arxiv_fulltext import fetch_arxiv_fulltext

        stats = fetch_arxiv_fulltext(
            corpus_papers=papers,
            openalex_to_arxiv=openalex_to_arxiv,
            show_progress=True,
            corpus_path=args.corpus,
        )

        print("\nExtraction complete:")
        print(f"  Total papers:       {stats['total']}")
        print(f"  With arXiv ID:      {stats['with_arxiv_id']}")
        print(f"  Already had text:   {stats['already_have']}")
        print(f"  Attempted:          {stats['attempted']}")
        print(f"  Extracted:          {stats['extracted']}")
        print(f"  Download failed:    {stats['download_failed']}")
        print(f"  Parse failed:       {stats['parse_failed']}")
        print(f"  No paragraphs:      {stats['no_paragraphs']}")

    else:  # unarxiv
        from incite.corpus.fulltext_extraction import extract_fulltext_from_unarxiv

        stats = extract_fulltext_from_unarxiv(
            corpus_papers=papers,
            openalex_to_arxiv=openalex_to_arxiv,
            data_dir=Path(args.data_dir),
            show_progress=True,
        )

        print("\nExtraction complete:")
        print(f"  Total papers:       {stats['total']}")
        print(f"  With arXiv ID:      {stats['with_arxiv_id']}")
        print(f"  Extracted:          {stats['extracted']}")
        print(f"  Not found in files: {stats['not_found']}")

    save_corpus(papers, args.corpus)

    new_with_text = sum(1 for p in papers if p.paragraphs)
    print(f"\nCorpus saved to {args.corpus}")
    print(f"Papers with paragraphs: {already_have} -> {new_with_text}")

    if new_with_text > already_have:
        print("\nNext steps:")
        print("1. Run 'incite enrich-chunks' to create chunks")
        print("2. Run 'incite index-chunks' to build the chunk index")
        print("3. Run 'incite evaluate --paragraph' to benchmark")


def cmd_extract_pdfs(args):
    """Extract text from PDFs in Zotero storage."""
    from incite.corpus.loader import load_corpus, save_corpus
    from incite.corpus.pdf_extractor import extract_pdfs_for_corpus

    print(f"Loading corpus from {args.corpus}...")
    papers = load_corpus(args.corpus)
    print(f"Loaded {len(papers)} papers")

    without_text = sum(1 for p in papers if not p.full_text)
    print(f"Papers without full text: {without_text}")

    if without_text == 0:
        print("All papers already have full text. Use --force to re-extract.")
        return

    zotero_dir = Path(args.zotero_dir).expanduser()
    print(f"Extracting text from PDFs in {zotero_dir}...")

    results = extract_pdfs_for_corpus(papers, zotero_dir, show_progress=True)

    updated = 0
    for paper in papers:
        if paper.id in results:
            result = results[paper.id]
            if result.full_text:
                paper.full_text = result.full_text
                paper.paragraphs = result.paragraphs
                updated += 1

    print(f"\nExtracted text from {updated} PDFs")

    save_corpus(papers, args.corpus)
    print(f"Saved updated corpus to {args.corpus}")

    if updated > 0:
        print("\nNext steps:")
        print("1. Run 'incite enrich-chunks' to create chunks with LLM context")
        print("2. Run 'incite index-chunks' to build the chunk index")
