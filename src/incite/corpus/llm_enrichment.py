"""LLM-based corpus enrichment for improved paper representations.

Uses Claude Haiku to generate enriched paper descriptions that capture
key contributions, methods, and topics beyond what title+abstract convey.
These descriptions are appended to the embedding text for better retrieval.

Based on Anthropic's Contextual Retrieval approach:
- https://www.anthropic.com/engineering/contextual-retrieval
- See docs/PROJECT_PLAN.md, "Deferred for Later Versions" section
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Optional

from tqdm import tqdm

from incite.models import Paper
from incite.utils import DEFAULT_LLM_MODEL

if TYPE_CHECKING:
    import anthropic

# Prompt for generating enriched paper descriptions
ENRICHMENT_PROMPT = """\
Given this academic paper's title and abstract, generate a short description \
(2-3 sentences) that captures:
- The core problem or research question
- The key method, approach, or contribution
- The main findings or implications

Focus on specific keywords and concepts that would help someone find this \
paper when searching for related work. Do not repeat the title verbatim.

Title: {title}

Abstract: {abstract}

Description:"""


def generate_description(
    paper: Paper,
    client: "anthropic.Anthropic",
    model: str = DEFAULT_LLM_MODEL,
) -> str:
    """Generate an enriched description for a single paper.

    Args:
        paper: Paper to describe
        client: Anthropic API client
        model: Model to use (default: Haiku for cost efficiency)

    Returns:
        Generated description string
    """
    if not paper.abstract:
        return ""

    prompt = ENRICHMENT_PROMPT.format(title=paper.title, abstract=paper.abstract)

    response = client.messages.create(
        model=model,
        max_tokens=200,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text.strip()


def enrich_corpus(
    papers: list[Paper],
    api_key: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    max_workers: int = 5,
    skip_existing: bool = True,
    show_progress: bool = True,
) -> dict:
    """Generate LLM descriptions for papers in a corpus.

    Modifies papers in-place by setting their llm_description field.

    Args:
        papers: List of Paper objects to enrich
        api_key: Anthropic API key (or uses ANTHROPIC_API_KEY env var)
        model: Model to use for generation
        max_workers: Number of parallel API calls
        skip_existing: Skip papers that already have descriptions
        show_progress: Show progress bar

    Returns:
        Stats dict with processing summary
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package required for LLM enrichment. Install with: pip install incite[llm]"
        )

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key parameter."
        )

    client = anthropic.Anthropic(api_key=api_key)

    # Filter papers to process
    to_process = []
    for paper in papers:
        if skip_existing and paper.llm_description:
            continue
        if not paper.abstract:
            continue
        to_process.append(paper)

    stats = {
        "total": len(papers),
        "to_process": len(to_process),
        "skipped_existing": len(papers) - len(to_process),
        "enriched": 0,
        "failed": 0,
    }

    if not to_process:
        return stats

    def _process_paper(paper: Paper) -> tuple[Paper, str, Optional[str]]:
        """Process a single paper, returning (paper, description, error)."""
        try:
            desc = generate_description(paper, client, model)
            return paper, desc, None
        except Exception as e:
            return paper, "", str(e)

    # Process with thread pool for parallel API calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_paper, p): p for p in to_process}

        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(to_process), desc="Generating descriptions")

        for future in iterator:
            paper, description, error = future.result()
            if error:
                stats["failed"] += 1
                if show_progress:
                    tqdm.write(f"  Failed: {paper.title[:60]}... ({error})")
            elif description:
                paper.llm_description = description
                stats["enriched"] += 1

    return stats


def enrich_corpus_batch(
    papers: list[Paper],
    api_key: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    skip_existing: bool = True,
    poll_interval: int = 30,
) -> dict:
    """Generate LLM descriptions using Anthropic's Message Batches API.

    Submits all requests as a single batch (up to 10K). 50% cheaper on
    input tokens and typically completes within minutes for small batches.

    Modifies papers in-place by setting their llm_description field.

    Args:
        papers: List of Paper objects to enrich
        api_key: Anthropic API key (or uses ANTHROPIC_API_KEY env var)
        model: Model to use for generation
        skip_existing: Skip papers that already have descriptions
        poll_interval: Seconds between status checks

    Returns:
        Stats dict with processing summary
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package required for LLM enrichment. Install with: pip install incite[llm]"
        )

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key parameter."
        )

    client = anthropic.Anthropic(api_key=api_key)

    # Filter papers to process
    to_process = []
    for paper in papers:
        if skip_existing and paper.llm_description:
            continue
        if not paper.abstract:
            continue
        to_process.append(paper)

    stats = {
        "total": len(papers),
        "to_process": len(to_process),
        "skipped_existing": len(papers) - len(to_process),
        "enriched": 0,
        "failed": 0,
    }

    if not to_process:
        return stats

    # Build paper lookup by ID
    paper_by_id = {p.id: p for p in to_process}

    # Build batch requests
    requests = []
    for paper in to_process:
        prompt = ENRICHMENT_PROMPT.format(title=paper.title, abstract=paper.abstract)
        requests.append(
            {
                "custom_id": paper.id,
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
        paper_id = entry.custom_id
        paper = paper_by_id.get(paper_id)
        if not paper:
            continue

        if entry.result.type == "succeeded":
            text = entry.result.message.content[0].text.strip()
            if text:
                paper.llm_description = text
                stats["enriched"] += 1
            else:
                stats["failed"] += 1
        else:
            stats["failed"] += 1

    return stats
