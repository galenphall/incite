"""LLM-based query reformulation for HyDE (Hypothetical Document Embeddings).

Transforms raw citation contexts into hypothetical paper descriptions,
closing the query-document genre gap. The LLM predicts what the cited
paper is about based on surrounding context, producing text that better
matches the title+abstract format of paper embeddings.

Only uses information available during drafting: citation context,
section heading, and title of the paper being written.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Optional

from tqdm import tqdm

from incite.models import CitationContext
from incite.utils import DEFAULT_LLM_MODEL

if TYPE_CHECKING:
    import anthropic

REFORMULATION_PROMPT = """\
You are helping find a specific academic paper. Given context from a paper \
being written, describe the paper that should be cited at the marked location.

Write a short hypothetical description (2-4 sentences) of what the cited \
paper is likely about, including a plausible title or topic, the likely \
research methods or approach, and key concepts and terminology the paper \
would contain.

Write plain text only. Do not use markdown, headers, bold, or bullet points.

{section_line}Paper being written: {global_context}

Citation context:
{citation_context}

Based on this context, the cited paper is most likely about:"""


def build_reformulation_prompt(
    context: CitationContext,
    source_scale: str = "narrow",
) -> str:
    """Build a reformulation prompt from a CitationContext.

    Args:
        context: Citation context to reformulate
        source_scale: Which context scale to use as input ("narrow" or "broad")

    Returns:
        Formatted prompt string
    """
    # Get citation context at requested scale, cleaned of markers
    citation_context = context.get_query(scale=source_scale, clean=True)

    # Section line: include if available
    section_line = ""
    if context.section_context:
        section_line = f"Section: {context.section_context}\n"

    # Global context: source paper title (or empty)
    global_context = context.global_context if context.global_context else "Unknown"

    return REFORMULATION_PROMPT.format(
        section_line=section_line,
        global_context=global_context,
        citation_context=citation_context,
    )


def _reformulate_single(
    context: CitationContext,
    client: "anthropic.Anthropic",
    model: str,
    source_scale: str,
) -> tuple[CitationContext, str, Optional[str]]:
    """Reformulate a single query, returning (context, result, error)."""
    try:
        prompt = build_reformulation_prompt(context, source_scale)
        response = client.messages.create(
            model=model,
            max_tokens=200,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return context, response.content[0].text.strip(), None
    except Exception as e:
        return context, "", str(e)


def reformulate_queries(
    contexts: list[CitationContext],
    api_key: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    source_scale: str = "narrow",
    max_workers: int = 5,
    skip_existing: bool = True,
    show_progress: bool = True,
) -> dict:
    """Reformulate citation contexts into hypothetical paper descriptions.

    Modifies contexts in-place by setting their reformulated_query field.

    Args:
        contexts: List of CitationContext objects to reformulate
        api_key: Anthropic API key (or uses ANTHROPIC_API_KEY env var)
        model: Model to use for generation
        source_scale: Context scale to use as input ("narrow" or "broad")
        max_workers: Number of parallel API calls
        skip_existing: Skip contexts that already have reformulated queries
        show_progress: Show progress bar

    Returns:
        Stats dict with processing summary
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package required for query reformulation. "
            "Install with: pip install incite[llm]"
        )

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key parameter."
        )

    client = anthropic.Anthropic(api_key=api_key)

    # Filter contexts to process
    to_process = []
    for ctx in contexts:
        if skip_existing and ctx.reformulated_query:
            continue
        to_process.append(ctx)

    stats = {
        "total": len(contexts),
        "to_process": len(to_process),
        "skipped_existing": len(contexts) - len(to_process),
        "reformulated": 0,
        "failed": 0,
    }

    if not to_process:
        return stats

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_reformulate_single, ctx, client, model, source_scale): ctx
            for ctx in to_process
        }

        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(to_process), desc="Reformulating queries")

        for future in iterator:
            context, result, error = future.result()
            if error:
                stats["failed"] += 1
                if show_progress:
                    tqdm.write(f"  Failed: {context.id} ({error})")
            elif result:
                context.reformulated_query = result
                stats["reformulated"] += 1

    return stats


def reformulate_queries_batch(
    contexts: list[CitationContext],
    api_key: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    source_scale: str = "narrow",
    skip_existing: bool = True,
    poll_interval: int = 30,
) -> dict:
    """Reformulate queries using Anthropic's Message Batches API.

    50% cheaper on input tokens. Modifies contexts in-place.

    Args:
        contexts: List of CitationContext objects to reformulate
        api_key: Anthropic API key (or uses ANTHROPIC_API_KEY env var)
        model: Model to use for generation
        source_scale: Context scale to use as input ("narrow" or "broad")
        skip_existing: Skip contexts that already have reformulated queries
        poll_interval: Seconds between status checks

    Returns:
        Stats dict with processing summary
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package required for query reformulation. "
            "Install with: pip install incite[llm]"
        )

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key parameter."
        )

    client = anthropic.Anthropic(api_key=api_key)

    # Filter contexts to process
    to_process = []
    for ctx in contexts:
        if skip_existing and ctx.reformulated_query:
            continue
        to_process.append(ctx)

    stats = {
        "total": len(contexts),
        "to_process": len(to_process),
        "skipped_existing": len(contexts) - len(to_process),
        "reformulated": 0,
        "failed": 0,
    }

    if not to_process:
        return stats

    # Build lookup by batch index (context IDs may contain characters
    # invalid for the Batch API's custom_id field)
    ctx_by_batch_id = {}

    # Build batch requests
    requests = []
    for i, ctx in enumerate(to_process):
        batch_id = f"ctx_{i}"
        ctx_by_batch_id[batch_id] = ctx
        prompt = build_reformulation_prompt(ctx, source_scale)
        requests.append(
            {
                "custom_id": batch_id,
                "params": {
                    "model": model,
                    "max_tokens": 200,
                    "temperature": 0.0,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
        )

    print(f"Submitting batch of {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch created: {batch.id}")
    print(f"Processing status: {batch.processing_status}")

    # Poll until complete
    from incite.corpus.batch_utils import poll_batch

    poll_batch(client, batch.id, poll_interval)
    print("Batch complete! Retrieving results...")

    # Retrieve results
    for entry in client.messages.batches.results(batch.id):
        batch_id = entry.custom_id
        ctx = ctx_by_batch_id.get(batch_id)
        if not ctx:
            continue

        if entry.result.type == "succeeded":
            text = entry.result.message.content[0].text.strip()
            if text:
                ctx.reformulated_query = text
                stats["reformulated"] += 1
            else:
                stats["failed"] += 1
        else:
            stats["failed"] += 1

    return stats
