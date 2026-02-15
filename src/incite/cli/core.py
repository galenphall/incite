"""Core retrieval commands: index, recommend, evaluate, index-chunks."""

import sys
from pathlib import Path

from incite.cli._shared import EMBEDDER_CHOICES


def register(subparsers):
    """Register core commands."""
    _register_index(subparsers)
    _register_recommend(subparsers)
    _register_evaluate(subparsers)
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


def _register_evaluate(subparsers):
    p = subparsers.add_parser("evaluate", help="Evaluate retrieval quality")
    p.add_argument(
        "--test-set",
        type=str,
        default="data/processed/test_set.jsonl",
        help="Path to test set JSONL file",
    )
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
    p.add_argument("--name", type=str, help="Optional name/notes for this experiment run")
    p.add_argument("--no-log", action="store_true", help="Skip logging this run")
    p.add_argument(
        "--reranker",
        type=str,
        choices=[
            "bge",
            "bge-base",
            "ms-marco",
            "ms-marco-l12",
            "jina",
            "gte",
            "citation-ft",
            "citation-ft-v4",
            "citation-ft-v5",
        ],
        default=None,
        help="Cross-encoder reranker (optional, adds second stage)",
    )
    p.add_argument(
        "--initial-k",
        type=int,
        default=100,
        help="Number of candidates for reranking (default: 100)",
    )
    p.add_argument(
        "--blend-alpha",
        type=float,
        default=0.0,
        help="Blend retrieval+CE scores: 0=pure CE, 0.5=equal, 1=pure retrieval (default: 0)",
    )
    p.add_argument(
        "--reranker-full-text",
        action="store_true",
        help="Pass full paper text (title+authors+year+abstract) to reranker instead of raw abstract",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to first N queries (for quick testing)",
    )
    p.add_argument(
        "--scale",
        type=str,
        choices=["local", "narrow", "broad", "section", "global", "reformulated"],
        default="narrow",
        help="Context scale to use for queries (default: narrow)",
    )
    p.add_argument(
        "--prefix-section",
        action="store_true",
        help="Prepend section heading to queries (e.g., 'Related Work: <context>')",
    )
    p.add_argument(
        "--paragraph",
        action="store_true",
        help="Use paragraph-level retrieval (requires chunks and chunk index)",
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
        "--aggregation",
        type=str,
        choices=["max", "mean", "sum", "weighted_max", "top_k_mean", "coverage", "log_normalized"],
        default="max",
        help="Chunk-to-paper aggregation method for --paragraph mode (default: max)",
    )
    p.add_argument(
        "--macro-average",
        action="store_true",
        help="Average within each source paper first, then across papers (corrects for skew)",
    )
    p.add_argument(
        "--sweep",
        type=str,
        choices=["scales"],
        default=None,
        help="Run evaluation across multiple settings (e.g. 'scales' sweeps context scales)",
    )
    p.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip test set cleaning (degenerate GT removal, dedup, mismatch detection)",
    )
    p.add_argument(
        "--evidence-metrics",
        action="store_true",
        help="Compute evidence quality metrics (needs chunks)",
    )
    p.add_argument(
        "--no-reference-sets",
        action="store_true",
        help="Disable reference set filtering (search full corpus, like real product use)",
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
        help="Paper vs chunk score blend weight for two-stage (default: 0.6)",
    )
    p.set_defaults(func=cmd_evaluate)


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


def _evaluate_cursor_weighted(
    retriever,
    test_set,
    k: int = 50,
    prefix_section: bool = False,
    macro_average: bool = False,
    focus_decay: float = 0.5,
):
    """Evaluate retrieval using cursor-weighted embeddings.

    Uses the narrow context, splits into sentences, and computes a weighted
    embedding focused on the last sentence (simulating cursor position).
    Falls back to standard narrow-scale evaluation if no embedder is available.

    Args:
        retriever: Retriever to evaluate
        test_set: List of CitationContext objects with ground_truth_ids
        k: Number of results to retrieve per query
        prefix_section: If True, prepend section heading to queries
        macro_average: If True, macro-average by source paper
        focus_decay: Exponential decay rate for sentence weighting
    """
    import re
    from collections import defaultdict

    from incite.evaluation.metrics import _get_embedder, evaluate_single
    from incite.models import EvaluationResult, QueryResult, clean_citation_markers

    embedder = _get_embedder(retriever)
    if embedder is None:
        from incite.evaluation import evaluate_retrieval

        print("  (no embedder found, falling back to narrow scale)")
        return evaluate_retrieval(
            retriever,
            test_set,
            k=k,
            scale="narrow",
            prefix_section=prefix_section,
            macro_average=macro_average,
        )

    # Sentence splitting (same regex as InCiteAgent._split_sentences)
    _ABBREV_RE = re.compile(
        r"\b(?:Dr|Mr|Mrs|Ms|Prof|Inc|Ltd|Jr|Sr|St|vs|etc|e\.g|i\.e|al|Fig|Eq|No|Vol)\."
    )

    def _split_sentences(text: str) -> list[str]:
        placeholder = "\x00"
        processed = _ABBREV_RE.sub(lambda m: re.sub(r"\.\s*", placeholder, m.group()), text)
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"])', processed)
        return [s.replace(placeholder, ". ").strip() for s in parts if s.strip()]

    query_results: list[QueryResult] = []

    for context in test_set:
        if not context.ground_truth_ids:
            continue

        # Use narrow context (same text as narrow scale)
        text = context.narrow_context if context.narrow_context else context.local_context
        text = clean_citation_markers(text)

        # Split into sentences and compute weighted embedding
        sentences = _split_sentences(text)
        if len(sentences) > 1:
            focus_index = len(sentences) - 1
            query_embedding = embedder.embed_query_weighted(
                sentences,
                focus_index,
                decay=focus_decay,
            )
        else:
            query_embedding = embedder.embed_query(text)

        # Retrieve with pre-computed embedding
        retrieve_k = k
        if context.reference_set_ids:
            retrieve_k = min(k * 3, 200)

        results = retriever.retrieve(text, k=retrieve_k, query_embedding=query_embedding)
        # Unpack if retriever returns (results, timing)
        if isinstance(results, tuple):
            results = results[0]

        # Filter to reference set
        if context.reference_set_ids:
            ref_set = set(context.reference_set_ids)
            results = [r for r in results if r.paper_id in ref_set]

        ref_set_size = len(context.reference_set_ids) if context.reference_set_ids else None
        scores = evaluate_single(results, context.ground_truth_ids, ref_set_size)

        first_rank = None
        gt_set = set(context.ground_truth_ids)
        for rank, r in enumerate(results, 1):
            if r.paper_id in gt_set:
                first_rank = rank
                break

        query_results.append(
            QueryResult(
                context_id=context.id,
                source_paper_id=context.source_paper_id,
                ground_truth_ids=context.ground_truth_ids,
                scores=scores,
                first_relevant_rank=first_rank,
            )
        )

    if not query_results:
        return EvaluationResult(num_queries=0)

    metric_keys = [
        "recall@1",
        "recall@5",
        "recall@10",
        "recall@20",
        "recall@50",
        "mrr",
        "ndcg@10",
        "concordance",
        "skill_mrr",
    ]

    if macro_average:
        by_paper: dict = defaultdict(list)
        for qr in query_results:
            by_paper[qr.source_paper_id].append(qr)
        paper_means: dict[str, list[float]] = {k_: [] for k_ in metric_keys}
        for paper_qrs in by_paper.values():
            for key in metric_keys:
                paper_mean = sum(qr.scores.get(key, 0.0) for qr in paper_qrs) / len(paper_qrs)
                paper_means[key].append(paper_mean)
        avg = {k_: sum(v) / len(v) for k_, v in paper_means.items()}
    else:
        n = len(query_results)
        avg = {k_: sum(qr.scores.get(k_, 0.0) for qr in query_results) / n for k_ in metric_keys}

    return EvaluationResult(
        recall_at_1=avg["recall@1"],
        recall_at_5=avg["recall@5"],
        recall_at_10=avg["recall@10"],
        recall_at_20=avg["recall@20"],
        recall_at_50=avg["recall@50"],
        mrr=avg["mrr"],
        ndcg_at_10=avg["ndcg@10"],
        concordance=avg.get("concordance", 0.0),
        skill_mrr=avg.get("skill_mrr", 0.0),
        num_queries=len(query_results),
        per_query=query_results,
    )


def cmd_evaluate(args):
    """Evaluate retrieval quality."""
    from incite.corpus.loader import load_corpus, load_test_set
    from incite.evaluation import (
        ExperimentConfig,
        ExperimentLogger,
        clean_test_set,
        compute_file_hash,
        evaluate_retrieval,
        evaluate_retrieval_stratified,
        evaluate_with_reranking,
    )
    from incite.retrieval.factory import create_retriever

    papers = load_corpus(args.corpus)
    paper_dict = {p.id: p for p in papers}

    test_set = load_test_set(args.test_set)
    if args.limit:
        test_set = test_set[: args.limit]
        print(f"Loaded {len(test_set)} test cases (limited from full set)")
    else:
        print(f"Loaded {len(test_set)} test cases")

    # Clean test set (remove degenerate GT, domain mismatches, duplicates)
    if not args.no_clean:
        test_set, cleaning_stats = clean_test_set(test_set, paper_dict)
        if cleaning_stats.total_removed > 0:
            print(cleaning_stats)

    if args.method == "bm25":
        model_name = "bm25"
    else:
        model_name = args.embedder

    if getattr(args, "two_stage", False):
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

        chunk_dict = {}
        chunks_jsonl = cache_dir / "zotero_chunks_paragraph.jsonl"
        if chunks_jsonl.exists():
            raw_chunks = load_chunks(str(chunks_jsonl))
            chunk_dict = {c.id: c for c in raw_chunks}
            print(f"Loaded {len(chunk_dict)} chunks for two-stage retrieval (alpha={args.alpha})")

        retriever = create_two_stage_retriever(
            papers=papers,
            chunk_store=chunk_store,
            chunks=chunk_dict,
            embedder_type=args.embedder,
            index_path=Path(args.index) if Path(args.index).exists() else None,
            alpha=args.alpha,
            show_progress=True,
        )
        model_name = f"{model_name}-two-stage-a{args.alpha}"
    elif args.paragraph:
        from incite.corpus.loader import load_chunks
        from incite.retrieval.factory import create_paragraph_retriever

        if not Path(args.chunks).exists():
            print(f"Error: Chunks file not found: {args.chunks}")
            print("Run 'incite enrich-chunks' first to create chunks.")
            sys.exit(1)

        chunks = load_chunks(args.chunks)
        print(f"Loaded {len(chunks)} chunks (paragraph mode, aggregation={args.aggregation})")
        retriever = create_paragraph_retriever(
            chunks=chunks,
            papers=papers,
            embedder_type=args.embedder,
            index_path=Path(args.chunk_index) if Path(args.chunk_index).exists() else None,
            method=args.method if args.method != "bm25" else "hybrid",
            aggregation=args.aggregation,
            show_progress=True,
        )
        model_name = f"{model_name}-paragraph-{args.aggregation}"
    else:
        retriever = create_retriever(
            papers=papers,
            method=args.method,
            embedder_type=args.embedder,
            fusion=args.fusion,
            show_progress=True,
        )

    # Sweep mode: run across all context scales
    if getattr(args, "sweep", None) == "scales":
        scales = ["local", "narrow", "broad", "section", "global", "cursor"]
        sweep_results = {}
        for scale in scales:
            print(f"\n--- Scale: {scale} ---")
            if scale == "cursor":
                result = _evaluate_cursor_weighted(
                    retriever,
                    test_set,
                    k=50,
                    prefix_section=args.prefix_section,
                    macro_average=args.macro_average,
                )
            else:
                result = evaluate_retrieval(
                    retriever,
                    test_set,
                    k=50,
                    scale=scale,
                    prefix_section=args.prefix_section,
                    macro_average=args.macro_average,
                )
            sweep_results[scale] = result
            print(result)

        # Print comparison table
        print("\n=== Sweep Summary ===")
        metrics = ["recall@1", "recall@5", "recall@10", "recall@20", "mrr", "ndcg@10"]
        header = f"{'Scale':<10}" + "".join(f"{m:<12}" for m in metrics)
        print(header)
        print("-" * len(header))
        for scale in scales:
            r = sweep_results[scale]
            d = r.to_dict()
            row = f"{scale:<10}" + "".join(f"{d[m]:<12.3f}" for m in metrics)
            print(row)

        if not args.no_log:
            extra = {
                "sweep": "scales",
                "sweep_results": {s: r.to_dict() for s, r in sweep_results.items()},
            }
            if args.macro_average:
                extra["macro_average"] = True
            config = ExperimentConfig(
                method=args.method,
                model_name=model_name,
                fusion=args.fusion if args.method == "hybrid" else None,
                k=50,
                scale="sweep",
                dataset_name=args.test_set,
                dataset_hash=compute_file_hash(args.test_set),
                extra=extra,
            )
            logger = ExperimentLogger()
            # Use narrow result as the primary result for the log
            run = logger.log(config, sweep_results["narrow"], notes=args.name)
            print(f"\nLogged as run: {run.id}")
        return

    if args.reranker:
        from incite.retrieval import get_reranker

        print(f"\nLoading reranker: {args.reranker}...")
        reranker = get_reranker(args.reranker)

        blend_label = f", blend_alpha={args.blend_alpha}" if args.blend_alpha > 0 else ""
        ft_label = ", full_text" if args.reranker_full_text else ""
        print(
            f"Evaluating with {args.method} retrieval + {args.reranker} reranking{blend_label}{ft_label}..."
        )
        print(f"Initial candidates: {args.initial_k}, final k: 50")
        use_ref = not getattr(args, "no_reference_sets", False)
        result = evaluate_with_reranking(
            retriever=retriever,
            reranker=reranker,
            papers=paper_dict,
            test_set=test_set,
            initial_k=args.initial_k,
            final_k=50,
            scale=args.scale,
            use_reference_sets=use_ref,
            prefix_section=args.prefix_section,
            show_progress=True,
            blend_alpha=args.blend_alpha,
            use_full_text=args.reranker_full_text,
        )
        model_name = f"{model_name}+{args.reranker}"
    else:
        prefix_label = " + section prefix" if args.prefix_section else ""
        para_label = f" [paragraph, agg={args.aggregation}]" if args.paragraph else ""
        macro_label = " [macro-avg]" if args.macro_average else ""
        print(
            f"Evaluating with {args.method} retrieval (embedder: {args.embedder}, "
            f"scale: {args.scale}{prefix_label}{para_label}{macro_label})..."
        )
        use_ref = not getattr(args, "no_reference_sets", False)
        result = evaluate_retrieval(
            retriever,
            test_set,
            k=50,
            scale=args.scale,
            use_reference_sets=use_ref,
            prefix_section=args.prefix_section,
            macro_average=args.macro_average,
        )

    print(result)

    if args.paragraph and not args.reranker:
        print("\nRunning stratified evaluation (full-text vs abstract-only ground truth)...")
        evaluate_retrieval_stratified(
            retriever,
            test_set,
            paper_dict,
            k=50,
            scale=args.scale,
            prefix_section=args.prefix_section,
        )

    if not args.no_log:
        extra = {}
        if args.paragraph:
            extra["paragraph"] = True
            extra["aggregation"] = args.aggregation
        if args.macro_average:
            extra["macro_average"] = True
        config = ExperimentConfig(
            method=args.method if not args.reranker else f"{args.method}+rerank",
            model_name=model_name,
            fusion=args.fusion if args.method == "hybrid" else None,
            k=50,
            scale=args.scale,
            dataset_name=args.test_set,
            dataset_hash=compute_file_hash(args.test_set),
            extra=extra,
        )
        logger = ExperimentLogger()
        run = logger.log(config, result, notes=args.name)
        print(f"\nLogged as run: {run.id}")

        # Save per-query results
        if result.per_query:
            pq_path = logger.save_per_query(run.id, result.per_query)
            print(f"Saved per-query data: {pq_path}")

    # Evidence quality metrics — run automatically when passage test set exists.
    # For two-stage retriever, evidence is already attached; for others, attach via chunk index.
    _run_evidence_metrics(args, retriever, paper_dict, result)


def _find_passage_test_set() -> Path | None:
    """Find a passage-level test set, checking standard locations.

    Returns the path if found, None otherwise.
    """
    candidates = [
        Path("data/finetuning/passage_test_set.jsonl"),
        Path("data/finetuning/master_eval.jsonl"),
        Path("data/processed/passage_test_set.jsonl"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _is_two_stage_retriever(retriever) -> bool:
    """Check if retriever is a TwoStageRetriever (produces evidence natively)."""
    from incite.retrieval.two_stage import TwoStageRetriever

    return isinstance(retriever, TwoStageRetriever)


def _run_evidence_metrics(args, retriever, paper_dict, eval_result):
    """Run evidence quality metrics on passage test set.

    Runs automatically when a passage test set is found. For TwoStageRetriever,
    evidence is already attached to results. For other retrievers, evidence is
    attached via chunk index lookup.

    Skips silently if no passage test set or chunk index is available.
    """
    from incite.evaluation.passage_metrics import (
        evaluate_evidence_quality,
        load_passage_test_set,
    )

    # Find passage test set — skip silently if none available
    passage_test_path = _find_passage_test_set()
    if passage_test_path is None:
        return

    passage_tests = load_passage_test_set(passage_test_path)
    if not passage_tests:
        return

    is_two_stage = _is_two_stage_retriever(retriever)

    # For non-two-stage retrievers, we need a chunk index to attach evidence
    chunk_store = None
    chunk_dict = {}
    embedder = None
    if not is_two_stage:
        try:
            cache_dir = Path.home() / ".incite"
            chunk_index_path = cache_dir / f"zotero_chunks_{args.embedder}"
            if not (chunk_index_path / "index.faiss").exists():
                return  # No chunk index — skip silently

            from incite.corpus.loader import load_chunks
            from incite.embeddings.chunk_store import ChunkStore
            from incite.retrieval.factory import get_embedder

            chunk_store = ChunkStore()
            chunk_store.load(chunk_index_path)

            chunks_jsonl = cache_dir / "zotero_chunks_paragraph.jsonl"
            if not chunks_jsonl.exists():
                return  # No chunks — skip silently

            chunks = load_chunks(str(chunks_jsonl))
            chunk_dict = {c.id: c for c in chunks}

            embedder = _find_embedder(retriever)
            if embedder is None:
                embedder = get_embedder(args.embedder)
        except Exception:
            return  # Skip silently on any setup error

    print(
        f"\nEvidence quality ({len(passage_tests)} passage examples from {passage_test_path.name})..."
    )

    # Run retrieval + evidence evaluation for each passage test
    results_with_evidence = []
    for test in passage_tests:
        results = retriever.retrieve(test.citation_context, k=50, papers=paper_dict)

        # For non-two-stage, manually attach evidence via chunk index
        if not is_two_stage and chunk_store is not None and embedder is not None:
            from incite.agent import InCiteAgent
            from incite.retrieval.paragraph import _highlight_sentence_in_parent

            query_embedding = embedder.embed_query(test.citation_context)
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

        results_with_evidence.append((results, test.gold_passage, test.cited_paper_id))

    # Evaluate
    evidence_metrics = evaluate_evidence_quality(results_with_evidence)

    n = evidence_metrics["num_queries"]
    prec = evidence_metrics["evidence_precision"]
    rec = evidence_metrics["evidence_recall"]
    f1 = evidence_metrics["evidence_f1"]
    token_f1 = evidence_metrics["mean_evidence_token_f1"]
    correct_papers = evidence_metrics["correct_papers_found"]
    papers_with_ev = evidence_metrics["correct_papers_with_evidence"]

    print(f"\nEvidence Quality (n={n}):")
    print(f"  Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}")
    print(f"  Mean Token F1: {token_f1:.3f}")
    print(f"  Coverage: {papers_with_ev}/{correct_papers} correct papers have evidence")

    # Update eval result for logging
    eval_result.evidence_precision = prec
    eval_result.evidence_recall = rec
    eval_result.evidence_f1 = f1


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
