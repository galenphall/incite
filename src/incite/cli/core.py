"""Core retrieval commands: index, recommend, index-chunks."""

import sys
from pathlib import Path

from incite.cli._shared import EMBEDDER_CHOICES


def register(subparsers):
    """Register core commands."""
    _register_index(subparsers)
    _register_recommend(subparsers)
    _register_index_chunks(subparsers)


def _register_index(subparsers):
    p = subparsers.add_parser("index", help="Build search index from corpus")
    p.add_argument(
        "--corpus",
        "-c",
        type=str,
        default="data/processed/corpus.jsonl",
        help="Path to corpus JSONL file",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/processed/index",
        help="Path to save index (or base dir for multi-scale)",
    )
    p.add_argument(
        "--embedder",
        type=str,
        choices=EMBEDDER_CHOICES,
        default="minilm",
        help="Embedder model to use (default: minilm)",
    )
    p.add_argument(
        "--multi-scale", action="store_true", help="Build paper, paragraph, and sentence indexes"
    )
    p.set_defaults(func=cmd_index)


def _register_recommend(subparsers):
    p = subparsers.add_parser("recommend", help="Get citation recommendations")
    p.add_argument("query", type=str, help="Text to get citations for")
    p.add_argument("--top-k", "-k", type=int, default=10, help="Number of recommendations")
    p.add_argument("--index", type=str, default="data/processed/index", help="Path to search index")
    p.add_argument(
        "--corpus",
        type=str,
        default="data/processed/corpus.jsonl",
        help="Path to corpus JSONL file",
    )
    p.add_argument(
        "--method",
        type=str,
        choices=["neural", "bm25", "hybrid"],
        default="neural",
        help="Retrieval method to use",
    )
    p.add_argument(
        "--fusion",
        type=str,
        choices=["rrf", "weighted"],
        default="rrf",
        help="Fusion method for hybrid retrieval",
    )
    p.add_argument(
        "--embedder",
        type=str,
        choices=EMBEDDER_CHOICES,
        default="minilm",
        help="Embedder model to use (default: minilm)",
    )
    p.add_argument(
        "--paragraph",
        action="store_true",
        help="Use paragraph-level retrieval (requires chunks index)",
    )
    p.add_argument(
        "--chunks",
        type=str,
        default="data/processed/chunks.jsonl",
        help="Path to chunks JSONL file (for --paragraph mode)",
    )
    p.add_argument(
        "--chunk-index",
        type=str,
        default="data/processed/chunk_index",
        help="Path to chunk index (for --paragraph mode)",
    )
    p.add_argument(
        "--no-deduplicate", action="store_true", help="Disable title-based deduplication of results"
    )
    p.add_argument(
        "--no-evidence",
        action="store_true",
        help="Disable paragraph evidence snippets (paper mode only)",
    )
    p.add_argument(
        "--multi-scale",
        action="store_true",
        help="Use multi-scale retrieval (paper + paragraph + sentence)",
    )
    p.add_argument(
        "--two-stage",
        action="store_true",
        help="Use two-stage retrieval (paper ranking + paragraph reranking)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Paper vs chunk score blend weight for two-stage retrieval (default: 0.6)",
    )
    p.set_defaults(func=cmd_recommend)


def _register_index_chunks(subparsers):
    p = subparsers.add_parser("index-chunks", help="Build FAISS index from chunks")
    p.add_argument(
        "--chunks",
        type=str,
        default="data/processed/chunks.jsonl",
        help="Path to chunks JSONL file",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/processed/chunk_index",
        help="Path to save chunk index",
    )
    p.add_argument(
        "--embedder",
        type=str,
        choices=EMBEDDER_CHOICES,
        default="nomic",
        help="Embedder model to use (default: nomic for 8K context)",
    )
    p.set_defaults(func=cmd_index_chunks)


# --- Command handlers ---


def cmd_index(args):
    """Build search index."""
    from incite.corpus.loader import load_corpus
    from incite.retrieval.factory import build_index, build_multi_scale_index

    print(f"Loading corpus from {args.corpus}...")
    papers = load_corpus(args.corpus)
    print(f"Loaded {len(papers)} papers")

    print(f"Building index with embedder: {args.embedder}...")

    if args.multi_scale:
        build_multi_scale_index(
            papers=papers,
            output_dir=args.output,
            embedder_type=args.embedder,
            show_progress=True,
        )
        print(f"Saved multi-scale indexes to {args.output}")
    else:
        build_index(
            papers=papers,
            output_path=args.output,
            embedder_type=args.embedder,
            show_progress=True,
        )
        print(f"Saved index to {args.output}")
    print("Done!")


def cmd_recommend(args):
    """Get recommendations for a query."""
    from incite.corpus.loader import load_corpus
    from incite.retrieval.factory import create_retriever

    papers = load_corpus(args.corpus)
    paper_dict = {p.id: p for p in papers}
    deduplicate = not args.no_deduplicate

    if args.two_stage:
        from incite.corpus.loader import load_chunks
        from incite.embeddings.chunk_store import ChunkStore
        from incite.retrieval.factory import create_two_stage_retriever

        # Try to find chunk index
        cache_dir = Path.home() / ".incite"
        chunk_index_path = cache_dir / f"zotero_chunks_{args.embedder}"
        if not (chunk_index_path / "index.faiss").exists():
            chunk_index_path = Path(args.chunk_index)
        if not (chunk_index_path / "index.faiss").exists():
            print("Error: No chunk index found. Build with 'incite index-chunks' first.")
            sys.exit(1)

        chunk_store = ChunkStore()
        chunk_store.load(chunk_index_path)

        # Load chunk texts
        chunk_dict = {}
        chunks_jsonl = cache_dir / "zotero_chunks_paragraph.jsonl"
        if chunks_jsonl.exists():
            raw_chunks = load_chunks(str(chunks_jsonl))
            chunk_dict = {c.id: c for c in raw_chunks}

        retriever = create_two_stage_retriever(
            papers=papers,
            chunk_store=chunk_store,
            chunks=chunk_dict,
            embedder_type=args.embedder,
            index_path=Path(args.index) if Path(args.index).exists() else None,
            alpha=args.alpha,
            show_progress=False,
        )

    elif args.multi_scale:
        from incite.retrieval.factory import create_multi_scale_retriever

        # args.index serves as base_dir for multi-scale
        base_dir = Path(args.index)
        if not base_dir.exists():
            print(f"Error: Multi-scale index directory not found: {base_dir}")
            print("Run 'incite index --multi-scale' first.")
            sys.exit(1)

        retriever = create_multi_scale_retriever(
            papers=papers,
            base_dir=base_dir,
            embedder_type=args.embedder,
            show_progress=False,
        )

    elif args.paragraph:
        from incite.corpus.loader import load_chunks
        from incite.retrieval.factory import create_paragraph_retriever

        if not Path(args.chunks).exists():
            print(f"Error: Chunks file not found: {args.chunks}")
            print("Run 'incite enrich-chunks' first to create chunks.")
            sys.exit(1)

        chunks = load_chunks(args.chunks)
        retriever = create_paragraph_retriever(
            chunks=chunks,
            papers=papers,
            embedder_type=args.embedder,
            index_path=Path(args.chunk_index) if Path(args.chunk_index).exists() else None,
            method=args.method if args.method != "bm25" else "hybrid",
            show_progress=False,
        )
    else:
        retriever = create_retriever(
            papers=papers,
            method=args.method,
            embedder_type=args.embedder,
            fusion=args.fusion,
            show_progress=False,
        )

    results, timing = retriever.retrieve(
        args.query,
        k=args.top_k,
        papers=paper_dict,
        deduplicate=deduplicate,
        return_timing=True,
    )

    # Attach paragraph evidence in paper mode (same logic as agent.py).
    # Skip for two-stage (evidence already attached) and paragraph modes.
    if (
        not getattr(args, "two_stage", False)
        and not args.paragraph
        and not args.multi_scale
        and not args.no_evidence
    ):
        _try_attach_evidence(results, args, retriever, timing)

    print(f"\nTop {args.top_k} recommendations for:")
    print(f'  "{args.query[:100]}{"..." if len(args.query) > 100 else ""}"\n')

    for i, result in enumerate(results, 1):
        paper = paper_dict.get(result.paper_id)
        if paper:
            confidence_label = (
                "HIGH"
                if result.confidence >= 0.55
                else "MED"
                if result.confidence >= 0.35
                else "LOW"
            )
            print(f"{i}. [{result.score:.3f}] {paper.title}  ({confidence_label})")
            if paper.authors:
                authors = ", ".join(paper.authors[:3])
                if len(paper.authors) > 3:
                    authors += " et al."
                print(f"   {authors} ({paper.year})")

            if result.matched_paragraph:
                preview = result.matched_paragraph[:200]
                if len(result.matched_paragraph) > 200:
                    preview += "..."
                print(f'   Evidence: "{preview}"')

            print()

    # Print timing summary
    total_ms = sum(v for v in timing.values() if isinstance(v, (int, float)))
    parts = []
    if "embed_query_ms" in timing:
        parts.append(f"embed={timing['embed_query_ms']:.0f}ms")
    if "vector_search_ms" in timing:
        parts.append(f"search={timing['vector_search_ms']:.0f}ms")
    if "bm25_search_ms" in timing:
        parts.append(f"bm25={timing['bm25_search_ms']:.0f}ms")
    if "evidence_ms" in timing:
        parts.append(f"evidence={timing['evidence_ms']:.0f}ms")
    if parts:
        print(f"  Timing: {' | '.join(parts)} | total={total_ms:.0f}ms")


def _try_attach_evidence(results, args, retriever, timing):
    """Try to load chunk index and attach evidence snippets to paper-mode results."""
    import time

    from incite.agent import InCiteAgent

    # Look for chunk index: check ~/.incite/ cache first, then explicit path
    cache_dir = Path.home() / ".incite"
    chunk_index_path = cache_dir / f"zotero_chunks_{args.embedder}"

    if not (chunk_index_path / "index.faiss").exists():
        # Fall back to explicit --chunk-index if it has a FAISS index
        explicit = Path(args.chunk_index)
        if (explicit / "index.faiss").exists():
            chunk_index_path = explicit
        else:
            return  # No chunk index available

    try:
        from incite.embeddings.chunk_store import ChunkStore
        from incite.retrieval.factory import get_embedder
        from incite.retrieval.paragraph import _highlight_sentence_in_parent

        # Load chunk store
        chunk_store = ChunkStore()
        chunk_store.load(chunk_index_path)

        # Load chunk texts for evidence display
        chunk_dict = {}
        chunks_jsonl = cache_dir / "zotero_chunks_paragraph.jsonl"
        if chunks_jsonl.exists():
            from incite.corpus.loader import load_chunks

            chunks = load_chunks(str(chunks_jsonl))
            chunk_dict = {c.id: c for c in chunks}

        # Get embedder and compute query embedding
        embedder = _find_embedder(retriever)
        if embedder is None:
            embedder = get_embedder(args.embedder)
        query_embedding = embedder.embed_query(args.query)

        # Search and attach (same logic as InCiteAgent._attach_evidence)
        evidence_start = time.perf_counter()
        n_chunks = max(100, len(results) * 10)
        chunk_results = chunk_store.search_with_papers(query_embedding, k=n_chunks)

        best_chunks: dict[str, tuple[str, float]] = {}
        for chunk_id, paper_id, score in chunk_results:
            if paper_id not in best_chunks or score > best_chunks[paper_id][1]:
                best_chunks[paper_id] = (chunk_id, score)

        for result in results:
            if result.paper_id in best_chunks:
                chunk_id, score = best_chunks[result.paper_id]
                if score >= InCiteAgent.EVIDENCE_THRESHOLD and chunk_id in chunk_dict:
                    chunk = chunk_dict[chunk_id]
                    result.matched_paragraph = _highlight_sentence_in_parent(chunk)
                    result.score_breakdown["best_chunk_score"] = score

        timing["evidence_ms"] = (time.perf_counter() - evidence_start) * 1000
    except Exception as e:
        import logging

        logging.getLogger(__name__).debug("Evidence lookup failed: %s", e)


def _find_embedder(retriever):
    """Extract the neural embedder from a retriever chain."""
    if hasattr(retriever, "embedder"):
        return retriever.embedder
    if hasattr(retriever, "retrievers"):
        for sub, _ in retriever.retrievers:
            if hasattr(sub, "embedder"):
                return sub.embedder
    return None


def cmd_index_chunks(args):
    """Build FAISS index from chunks."""
    from incite.corpus.loader import load_chunks
    from incite.retrieval.factory import build_chunk_index

    print(f"Loading chunks from {args.chunks}...")
    chunks = load_chunks(args.chunks)
    print(f"Loaded {len(chunks)} chunks")

    with_context = sum(1 for c in chunks if c.context_text)
    print(f"Chunks with LLM context: {with_context}")

    print(f"Building index with embedder: {args.embedder}...")
    build_chunk_index(
        chunks=chunks,
        output_path=Path(args.output),
        embedder_type=args.embedder,
        show_progress=True,
    )

    print(f"Saved chunk index to {args.output}")
    print("\nDone! You can now use paragraph-level retrieval:")
    print("  incite recommend 'your query' --paragraph")
