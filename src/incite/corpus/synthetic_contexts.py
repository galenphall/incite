"""Synthetic citation context generation from Zotero library papers.

Generates diverse citation contexts using Claude Haiku, classified by
citation type (background/methods/results/comparison/motivation).
Supports both batch API and threaded generation modes.

Prompt templates and response parsers live in
``incite.corpus.synthetic_prompts`` — this module handles orchestration only.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from incite.corpus.synthetic_db import SyntheticDB
from incite.corpus.synthetic_prompts import (
    MODERATE_DIFFICULTY_PROMPT,
    SYNTHETIC_PROMPT,
    parse_moderate_response,
    parse_response,
)
from incite.models import CitationContext, Paper
from incite.utils import DEFAULT_LLM_MODEL

logger = logging.getLogger(__name__)


def generate_synthetic_threaded(
    papers: list[Paper],
    db: SyntheticDB,
    api_key: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    max_workers: int = 5,
    skip_existing: bool = True,
    show_progress: bool = True,
    difficulty: str = "standard",
) -> dict:
    """Generate synthetic contexts using threaded API calls.

    Args:
        papers: Papers to generate contexts for
        db: SyntheticDB instance to store results
        api_key: Anthropic API key (or ANTHROPIC_API_KEY env var)
        model: Model to use
        max_workers: Number of parallel API calls
        skip_existing: Skip papers already in DB
        show_progress: Show progress bar
        difficulty: "standard" (5 easy contexts) or "moderate" (3 harder contexts)

    Returns:
        Stats dict with processing summary
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required. Install with: pip install anthropic")

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key parameter."
        )

    client = anthropic.Anthropic(api_key=api_key)

    # Select prompt and parser based on difficulty
    if difficulty == "moderate":
        prompt_template = MODERATE_DIFFICULTY_PROMPT
        parser_fn = parse_moderate_response
    else:
        prompt_template = SYNTHETIC_PROMPT
        parser_fn = parse_response

    # Filter papers — for moderate, only skip papers that already have moderate contexts
    existing_ids = db.get_existing_paper_ids(difficulty=difficulty) if skip_existing else set()
    to_process = [
        p for p in papers if p.abstract and len(p.abstract) >= 50 and p.id not in existing_ids
    ]

    existing_count = len(existing_ids & {p.id for p in papers})
    stats = {
        "total": len(papers),
        "to_process": len(to_process),
        "skipped_existing": existing_count,
        "skipped_no_abstract": len(papers) - len(to_process) - existing_count,
        "generated": 0,
        "contexts_created": 0,
        "failed": 0,
    }

    if not to_process:
        return stats

    def _process_paper(paper: Paper) -> tuple[Paper, list[dict], Optional[str]]:
        try:
            prompt = prompt_template.format(title=paper.title, abstract=paper.abstract)
            response = client.messages.create(
                model=model,
                max_tokens=1500,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            contexts = parser_fn(paper, text)
            return paper, contexts, None
        except Exception as e:
            return paper, [], str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_paper, p): p for p in to_process}

        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(to_process), desc="Generating contexts")

        for future in iterator:
            paper, contexts, error = future.result()
            if error:
                stats["failed"] += 1
                if show_progress:
                    tqdm.write(f"  Failed: {paper.title[:60]}... ({error})")
            elif contexts:
                db.insert_contexts(contexts)
                stats["generated"] += 1
                stats["contexts_created"] += len(contexts)
            else:
                stats["failed"] += 1
                if show_progress:
                    tqdm.write(f"  No valid contexts: {paper.title[:60]}...")

    db.log_run(
        model=model,
        num_papers=stats["generated"],
        num_contexts=stats["contexts_created"],
        notes="threaded generation",
    )

    return stats


def generate_synthetic_batch(
    papers: list[Paper],
    db: SyntheticDB,
    api_key: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    poll_interval: int = 30,
    skip_existing: bool = True,
    difficulty: str = "standard",
) -> dict:
    """Generate synthetic contexts using Anthropic Batch API.

    50% cheaper than threaded mode. Submits all requests as a single batch.

    Args:
        papers: Papers to generate contexts for
        db: SyntheticDB instance to store results
        api_key: Anthropic API key (or ANTHROPIC_API_KEY env var)
        model: Model to use
        poll_interval: Seconds between status checks
        skip_existing: Skip papers already in DB
        difficulty: "standard" (5 easy contexts) or "moderate" (3 harder contexts)

    Returns:
        Stats dict with processing summary
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required. Install with: pip install anthropic")

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key parameter."
        )

    client = anthropic.Anthropic(api_key=api_key)

    # Select prompt and parser based on difficulty
    if difficulty == "moderate":
        prompt_template = MODERATE_DIFFICULTY_PROMPT
        parser_fn = parse_moderate_response
    else:
        prompt_template = SYNTHETIC_PROMPT
        parser_fn = parse_response

    # Filter papers — for moderate, only skip papers that already have moderate contexts
    existing_ids = db.get_existing_paper_ids(difficulty=difficulty) if skip_existing else set()
    to_process = [
        p for p in papers if p.abstract and len(p.abstract) >= 50 and p.id not in existing_ids
    ]

    existing_count = len(existing_ids & {p.id for p in papers})
    stats = {
        "total": len(papers),
        "to_process": len(to_process),
        "skipped_existing": existing_count,
        "skipped_no_abstract": len(papers) - len(to_process) - existing_count,
        "generated": 0,
        "contexts_created": 0,
        "failed": 0,
        "batch_id": None,
    }

    if not to_process:
        return stats

    # Build lookup and requests
    paper_by_custom_id = {}
    requests = []
    for i, paper in enumerate(to_process):
        custom_id = f"synth_{i}"
        paper_by_custom_id[custom_id] = paper
        prompt = prompt_template.format(title=paper.title, abstract=paper.abstract)
        requests.append(
            {
                "custom_id": custom_id,
                "params": {
                    "model": model,
                    "max_tokens": 1500,
                    "temperature": 0.7,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
        )

    print(f"Submitting batch of {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    stats["batch_id"] = batch.id
    print(f"Batch created: {batch.id}")
    print(f"Processing status: {batch.processing_status}")

    # Poll until complete
    while batch.processing_status != "ended":
        time.sleep(poll_interval)
        batch = client.messages.batches.retrieve(batch.id)
        counts = batch.request_counts
        total = (
            counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
        )
        done = counts.succeeded + counts.errored + counts.canceled + counts.expired
        print(
            f"  Status: {batch.processing_status} | "
            f"Done: {done}/{total} | "
            f"Succeeded: {counts.succeeded} | "
            f"Errors: {counts.errored}"
        )

    print("Batch complete! Retrieving results...")

    # Retrieve and parse results
    for entry in client.messages.batches.results(batch.id):
        paper = paper_by_custom_id.get(entry.custom_id)
        if not paper:
            continue

        if entry.result.type == "succeeded":
            text = entry.result.message.content[0].text
            contexts = parser_fn(paper, text)
            if contexts:
                db.insert_contexts([{**c, "batch_id": batch.id} for c in contexts])
                stats["generated"] += 1
                stats["contexts_created"] += len(contexts)
            else:
                stats["failed"] += 1
        else:
            stats["failed"] += 1

    db.log_run(
        model=model,
        num_papers=stats["generated"],
        num_contexts=stats["contexts_created"],
        batch_id=batch.id,
        notes="batch generation",
    )

    return stats


def build_reference_sets(
    papers: list[Paper],
    embedder_type: str = "minilm",
    k: int = 50,
    show_progress: bool = True,
) -> dict[str, list[str]]:
    """Build K-NN reference sets from FAISS index.

    Each paper gets a set of k nearest neighbors as its "distractor" set
    for evaluation. The paper itself is included in its own reference set.

    Args:
        papers: List of papers (must already be indexed or will be indexed)
        embedder_type: Embedder to use for similarity
        k: Number of neighbors per paper (default: 50)
        show_progress: Show progress bar

    Returns:
        Dict mapping paper_id -> list of neighbor paper_ids (including self)
    """
    # Prevent OMP duplicate library crash on macOS (faiss + PyTorch conflict).
    # PyTorch and FAISS each link their own libomp; batch FAISS search triggers
    # OpenMP parallelism which segfaults. Single-threading FAISS avoids this
    # with negligible perf impact (~37ms for 3K vectors).
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import faiss

    faiss.omp_set_num_threads(1)

    from incite.embeddings.stores import FAISSStore
    from incite.retrieval.factory import build_index

    cache_dir = Path.home() / ".incite"

    # Try to load existing index (built by webapp or previous run)
    index_path = cache_dir / f"zotero_index_{embedder_type}"
    store = FAISSStore()

    if (index_path / "index.faiss").exists() and (index_path / "id_map.json").exists():
        print(f"Loading existing index from {index_path}...")
        store.load(index_path)
        # Verify the index has our papers
        indexed_ids = set(store._id_to_idx.keys())
        paper_ids = {p.id for p in papers}
        overlap = len(indexed_ids & paper_ids)
        if overlap < len(paper_ids) * 0.8:
            print(f"  Index has {overlap}/{len(paper_ids)} papers — rebuilding...")
            store = None
        else:
            print(f"  Loaded index with {store.size} vectors ({overlap} match our papers)")
    else:
        store = None

    if store is None:
        # Build index from scratch
        print(f"Building {embedder_type} index for {len(papers)} papers...")
        index_path.mkdir(parents=True, exist_ok=True)
        build_index(papers, index_path, embedder_type=embedder_type, show_progress=show_progress)
        store = FAISSStore()
        store.load(index_path)
        print(f"  Built index with {store.size} vectors")

    # Reconstruct all vectors and do batch K-NN search
    n = store._index.ntotal
    dim = store._index.d
    print(f"Computing {k}-NN for {n} papers...")

    # Reconstruct vectors in batches to avoid memory issues
    all_vectors = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        all_vectors[i] = store._index.reconstruct(i)

    # Search for k+1 neighbors (to exclude self)
    search_k = min(k + 1, n)
    scores, indices = store._index.search(all_vectors, search_k)

    # Build reference sets
    ref_sets: dict[str, list[str]] = {}
    for i in range(n):
        paper_id = store._idx_to_id.get(i)
        if not paper_id:
            continue

        neighbors = []
        for j in range(search_k):
            idx = indices[i][j]
            if idx >= 0:
                neighbor_id = store._idx_to_id.get(idx)
                if neighbor_id and neighbor_id != paper_id:
                    neighbors.append(neighbor_id)
            if len(neighbors) >= k - 1:  # -1 because we'll prepend self
                break

        # Include self as first entry (ground truth is always in ref set)
        ref_sets[paper_id] = [paper_id] + neighbors

    avg_size = np.mean([len(v) for v in ref_sets.values()])
    print(f"  Built reference sets for {len(ref_sets)} papers (avg size: {avg_size:.0f})")
    return ref_sets


def export_to_jsonl(
    db: SyntheticDB,
    ref_sets: dict[str, list[str]],
    output_path: Path,
) -> int:
    """Export synthetic contexts to JSONL test set format.

    Args:
        db: SyntheticDB with generated contexts
        ref_sets: K-NN reference sets (paper_id -> neighbor list)
        output_path: Output path for JSONL file

    Returns:
        Number of contexts exported
    """
    from incite.corpus.loader import save_test_set

    contexts = db.get_contexts()
    citation_contexts = []

    for ctx in contexts:
        paper_id = ctx["paper_id"]
        ref_set = ref_sets.get(paper_id, [paper_id])

        # Ensure the target paper is in the reference set
        if paper_id not in ref_set:
            ref_set = [paper_id] + ref_set

        cc = CitationContext(
            id=ctx["id"],
            local_context=ctx["text"],
            narrow_context=ctx["text"],
            broad_context=ctx["text"],
            section_context=ctx.get("section_hint") or "",
            global_context="",
            source_paper_id=None,
            ground_truth_ids=[paper_id],
            reference_set_ids=ref_set,
            difficulty=ctx.get("difficulty") or "",
        )
        citation_contexts.append(cc)

    save_test_set(citation_contexts, output_path)
    return len(citation_contexts)
