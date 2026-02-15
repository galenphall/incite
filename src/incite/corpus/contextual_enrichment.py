"""Contextual enrichment for chunks using Claude Haiku.

This module implements Anthropic's contextual retrieval approach:
https://www.anthropic.com/news/contextual-retrieval

For each chunk, we generate a short context that situates it within the
overall document. This context is prepended to both the embedding AND
BM25 representations, improving retrieval by 35-49% over baseline.

Key features:
- Uses prompt caching to cache the full document, making per-chunk enrichment cheap
- Generates 50-100 token context per chunk
- Async batch processing for efficiency
"""

import asyncio
import os
from typing import Optional

from incite.models import Chunk, Paper
from incite.utils import DEFAULT_LLM_MODEL

CONTEXT_PROMPT = """<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall document
for the purposes of improving search retrieval of the chunk. Answer only with the
succinct context and nothing else."""


async def enrich_chunk_with_context(
    paper: Paper,
    chunk: Chunk,
    client,  # anthropic.AsyncAnthropic
    model: str = DEFAULT_LLM_MODEL,
    cached_document_tokens: Optional[int] = None,
) -> str:
    """Generate situating context for a single chunk.

    Args:
        paper: The paper containing this chunk
        chunk: The chunk to generate context for
        client: Anthropic async client
        model: Model to use (default: Haiku for speed/cost)
        cached_document_tokens: If provided, assume document is already cached

    Returns:
        Generated context string
    """
    # Build document text for context
    doc_parts = []
    if paper.title:
        doc_parts.append(f"Title: {paper.title}")
    if paper.abstract:
        doc_parts.append(f"Abstract: {paper.abstract}")
    if paper.full_text:
        doc_parts.append(f"Full text:\n{paper.full_text}")
    elif paper.paragraphs:
        doc_parts.append(f"Full text:\n{chr(10).join(paper.paragraphs)}")

    document = "\n\n".join(doc_parts)

    # Build the prompt
    prompt = CONTEXT_PROMPT.format(document=document, chunk=chunk.text)

    try:
        # Use prompt caching for the document portion
        response = await client.messages.create(
            model=model,
            max_tokens=150,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
        )
        return response.content[0].text.strip()
    except Exception as e:
        # Return empty string on error, chunk still usable without context
        print(f"Warning: Failed to generate context for chunk {chunk.id}: {e}")
        return ""


async def enrich_chunks_for_paper(
    paper: Paper,
    chunks: list[Chunk],
    client,  # anthropic.AsyncAnthropic
    model: str = DEFAULT_LLM_MODEL,
    max_concurrent: int = 5,
) -> list[Chunk]:
    """Enrich all chunks for a single paper with contextual information.

    Uses prompt caching: the full document is cached once, then each chunk
    query uses the cached context, reducing cost significantly.

    Args:
        paper: The paper these chunks belong to
        chunks: List of chunks to enrich
        client: Anthropic async client
        model: Model to use
        max_concurrent: Maximum concurrent API calls

    Returns:
        List of chunks with context_text populated
    """
    if not chunks:
        return []

    # Build document text once
    doc_parts = []
    if paper.title:
        doc_parts.append(f"Title: {paper.title}")
    if paper.abstract:
        doc_parts.append(f"Abstract: {paper.abstract}")
    if paper.full_text:
        doc_parts.append(f"Full text:\n{paper.full_text}")
    elif paper.paragraphs:
        doc_parts.append(f"Full text:\n{chr(10).join(paper.paragraphs)}")

    document = "\n\n".join(doc_parts)

    # Semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    async def enrich_single(chunk: Chunk) -> Chunk:
        async with semaphore:
            prompt = CONTEXT_PROMPT.format(document=document, chunk=chunk.text)

            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=150,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                    "cache_control": {"type": "ephemeral"},
                                }
                            ],
                        }
                    ],
                )
                chunk.context_text = response.content[0].text.strip()
            except Exception as e:
                print(f"Warning: Failed to enrich chunk {chunk.id}: {e}")
                chunk.context_text = None

            return chunk

    # Process all chunks concurrently
    enriched = await asyncio.gather(*[enrich_single(c) for c in chunks])
    return list(enriched)


def enrich_chunks_sync(
    papers: dict[str, Paper],
    chunks: list[Chunk],
    api_key: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    max_concurrent: int = 5,
    show_progress: bool = True,
    skip_existing: bool = True,
) -> dict:
    """Synchronous wrapper for enriching chunks with contextual information.

    Groups chunks by paper and enriches each paper's chunks together
    to maximize prompt cache hits.

    Args:
        papers: Dict mapping paper_id -> Paper
        chunks: List of all chunks to enrich
        api_key: Anthropic API key (or uses ANTHROPIC_API_KEY env var)
        model: Model to use
        max_concurrent: Maximum concurrent API calls per paper
        show_progress: Whether to show progress bar
        skip_existing: Skip chunks that already have context_text

    Returns:
        Stats dict with counts of enriched, skipped, failed
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required. Install with: pip install anthropic")

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    # Filter chunks if skipping existing
    if skip_existing:
        chunks_to_process = [c for c in chunks if not c.context_text]
    else:
        chunks_to_process = chunks

    stats = {
        "total": len(chunks),
        "enriched": 0,
        "skipped": len(chunks) - len(chunks_to_process),
        "failed": 0,
    }

    if not chunks_to_process:
        return stats

    # Group chunks by paper
    chunks_by_paper: dict[str, list[Chunk]] = {}
    for chunk in chunks_to_process:
        if chunk.paper_id not in chunks_by_paper:
            chunks_by_paper[chunk.paper_id] = []
        chunks_by_paper[chunk.paper_id].append(chunk)

    # Create async client
    client = anthropic.AsyncAnthropic(api_key=api_key)

    async def process_all():
        from tqdm.asyncio import tqdm

        paper_ids = list(chunks_by_paper.keys())

        if show_progress:
            iterator = tqdm(paper_ids, desc="Enriching papers")
        else:
            iterator = paper_ids

        for paper_id in iterator:
            paper = papers.get(paper_id)
            if not paper:
                stats["failed"] += len(chunks_by_paper[paper_id])
                continue

            paper_chunks = chunks_by_paper[paper_id]
            try:
                await enrich_chunks_for_paper(paper, paper_chunks, client, model, max_concurrent)
                stats["enriched"] += sum(1 for c in paper_chunks if c.context_text)
                stats["failed"] += sum(1 for c in paper_chunks if not c.context_text)
            except Exception as e:
                print(f"Error enriching paper {paper_id}: {e}")
                stats["failed"] += len(paper_chunks)

    # Run async processing
    asyncio.run(process_all())

    return stats


def enrich_chunks_batch(
    papers: dict[str, Paper],
    chunks: list[Chunk],
    api_key: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    skip_existing: bool = True,
    poll_interval: int = 30,
) -> dict:
    """Enrich chunks using Anthropic's Message Batches API (50% cheaper).

    Submits all chunk enrichment requests as a batch. Each request includes
    the full document context and the chunk to situate.

    Note: Batch API doesn't support prompt caching, so each request includes
    the full document. Still 50% cheaper on input tokens than standard API.

    Args:
        papers: Dict mapping paper_id -> Paper
        chunks: List of all chunks to enrich
        api_key: Anthropic API key (or uses ANTHROPIC_API_KEY env var)
        model: Model to use
        skip_existing: Skip chunks that already have context_text
        poll_interval: Seconds between status checks

    Returns:
        Stats dict with counts of enriched, skipped, failed
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required. Install with: pip install anthropic")

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    # Filter chunks
    if skip_existing:
        chunks_to_process = [c for c in chunks if not c.context_text]
    else:
        chunks_to_process = list(chunks)

    stats = {
        "total": len(chunks),
        "enriched": 0,
        "skipped": len(chunks) - len(chunks_to_process),
        "failed": 0,
    }

    if not chunks_to_process:
        return stats

    # Build chunk lookup
    chunk_by_id = {c.id: c for c in chunks_to_process}

    # Build document text cache (one per paper)
    doc_cache: dict[str, str] = {}
    for chunk in chunks_to_process:
        if chunk.paper_id not in doc_cache:
            paper = papers.get(chunk.paper_id)
            if paper:
                doc_parts = []
                if paper.title:
                    doc_parts.append(f"Title: {paper.title}")
                if paper.abstract:
                    doc_parts.append(f"Abstract: {paper.abstract}")
                if paper.full_text:
                    doc_parts.append(f"Full text:\n{paper.full_text}")
                elif paper.paragraphs:
                    doc_parts.append(f"Full text:\n{chr(10).join(paper.paragraphs)}")
                doc_cache[chunk.paper_id] = "\n\n".join(doc_parts)

    # Build batch requests
    requests = []
    for chunk in chunks_to_process:
        document = doc_cache.get(chunk.paper_id, "")
        if not document:
            stats["failed"] += 1
            continue

        prompt = CONTEXT_PROMPT.format(document=document, chunk=chunk.text)
        requests.append(
            {
                "custom_id": chunk.id,
                "params": {
                    "model": model,
                    "max_tokens": 150,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
        )

    if not requests:
        return stats

    # Submit batch (max 10K per batch)
    client = anthropic.Anthropic(api_key=api_key)

    # Split into sub-batches of 10K if needed
    batch_size = 10_000
    all_batch_ids = []

    for i in range(0, len(requests), batch_size):
        sub_requests = requests[i : i + batch_size]
        print(f"Submitting batch {len(all_batch_ids) + 1} ({len(sub_requests)} requests)...")
        batch = client.messages.batches.create(requests=sub_requests)
        all_batch_ids.append(batch.id)
        print(f"  Batch ID: {batch.id}")

    # Poll all batches until complete
    from incite.corpus.batch_utils import poll_batch

    for batch_id in all_batch_ids:
        print(f"\nWaiting for batch {batch_id}...")
        poll_batch(client, batch_id, poll_interval)
        print(f"Batch {batch_id} complete! Retrieving results...")

        # Retrieve results
        for entry in client.messages.batches.results(batch_id):
            chunk_id = entry.custom_id
            chunk = chunk_by_id.get(chunk_id)
            if not chunk:
                continue

            if entry.result.type == "succeeded":
                text = entry.result.message.content[0].text.strip()
                if text:
                    chunk.context_text = text
                    stats["enriched"] += 1
                else:
                    stats["failed"] += 1
            else:
                stats["failed"] += 1

    return stats


def estimate_enrichment_cost(
    papers: list[Paper],
    chunks: list[Chunk],
    model: str = DEFAULT_LLM_MODEL,
) -> dict:
    """Estimate the cost of enriching chunks.

    Uses Anthropic's pricing with prompt caching assumptions.

    Args:
        papers: List of papers
        chunks: List of chunks to enrich
        model: Model to use

    Returns:
        Dict with token counts and estimated costs
    """
    # Rough token estimates (chars / 4)
    total_doc_tokens = 0
    total_chunk_tokens = 0

    for paper in papers:
        doc_tokens = len(paper.title or "") // 4
        doc_tokens += len(paper.abstract or "") // 4
        doc_tokens += len(paper.full_text or "") // 4
        total_doc_tokens += doc_tokens

    for chunk in chunks:
        total_chunk_tokens += len(chunk.text) // 4

    # Group chunks by paper to estimate cache usage
    chunks_per_paper = len(chunks) / len(papers) if papers else 0

    # Haiku pricing (as of 2024):
    # - Input: $0.25/M tokens
    # - Output: $1.25/M tokens
    # - Cached input: $0.03/M tokens (88% discount)
    # - Cache write: $0.30/M tokens

    # With caching, each document is written once, then read cached for each chunk
    cache_write_cost = total_doc_tokens * 0.30 / 1_000_000
    cached_read_cost = total_doc_tokens * chunks_per_paper * 0.03 / 1_000_000
    chunk_input_cost = total_chunk_tokens * 0.25 / 1_000_000
    output_cost = len(chunks) * 75 * 1.25 / 1_000_000  # ~75 tokens per context

    total_cost = cache_write_cost + cached_read_cost + chunk_input_cost + output_cost

    return {
        "total_papers": len(papers),
        "total_chunks": len(chunks),
        "avg_chunks_per_paper": chunks_per_paper,
        "estimated_doc_tokens": total_doc_tokens,
        "estimated_chunk_tokens": total_chunk_tokens,
        "estimated_output_tokens": len(chunks) * 75,
        "cache_write_cost": cache_write_cost,
        "cached_read_cost": cached_read_cost,
        "chunk_input_cost": chunk_input_cost,
        "output_cost": output_cost,
        "total_estimated_cost": total_cost,
    }
