"""Passage-level training data generation using LLM.

Given paragraph chunks from full-text papers, generates citation contexts
that reference specific passages. The passage IS the ground truth — no
matching or validation needed.
"""

import json
import os
import random
import re
import time
from collections import defaultdict
from typing import Optional

from incite.evaluation.passage_metrics import PassageTestExample
from incite.finetuning.types import TrainingExample
from incite.models import (
    Chunk,
    Paper,
    clean_citation_markers,
    format_paper_metadata_prefix,
    format_passage_embedding_text,
)
from incite.utils import DEFAULT_LLM_MODEL

VALID_TYPES = {"background", "methods", "results", "comparison", "motivation"}

# Sections to skip (boilerplate, not useful for citation training)
SKIP_SECTIONS = {
    "acknowledgments",
    "acknowledgements",
    "acknowledgment",
    "acknowledgement",
    "author contributions",
    "competing interests",
    "conflict of interest",
    "conflicts of interest",
    "data availability",
    "funding",
    "supplementary material",
    "supplementary information",
    "appendix",
    "appendices",
    "supporting information",
}

PASSAGE_PROMPT = """\
You are an expert academic writer. Given a passage from a published paper, \
write 2 citation contexts that would appear in OTHER papers citing this \
specific passage.

Rules:
- 2-4 sentences of flowing academic prose per context
- Mark citation point with [CITE]
- MUST reference SPECIFIC content of this passage, not the paper generally
- Paraphrase heavily, no verbatim copying from passage
- Do NOT use the paper title verbatim
- Each context must use a DIFFERENT citation type

Paper: {title} ({year})
Abstract (for context): {abstract_truncated}
Section: {section}

Passage:
{passage_text}

Output JSON:
{{"contexts": [
  {{"type": "<background|methods|results|comparison|motivation>", "text": "..."}},
  {{"type": "<type>", "text": "..."}}
]}}"""


def select_passages(
    chunks: list[Chunk],
    max_per_paper: int = 10,
    min_length: int = 150,
    max_length: int = 2000,
) -> list[Chunk]:
    """Select diverse, informative passages from chunks for generation.

    Filters by length, skips boilerplate sections, and caps per paper
    with section diversity (max 3 from any single section).

    Args:
        chunks: All chunks to select from
        max_per_paper: Maximum passages per paper
        min_length: Minimum chunk text length in characters
        max_length: Maximum chunk text length in characters

    Returns:
        Selected chunks suitable for context generation
    """
    # Group by paper
    by_paper: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_paper[chunk.paper_id].append(chunk)

    selected: list[Chunk] = []

    for paper_id, paper_chunks in by_paper.items():
        # Filter by length and section
        candidates = []
        for chunk in paper_chunks:
            text_len = len(chunk.text.strip())
            if text_len < min_length or text_len > max_length:
                continue

            # Skip boilerplate sections
            section_lower = (chunk.section or "").lower().strip()
            if section_lower in SKIP_SECTIONS:
                continue

            candidates.append(chunk)

        if not candidates:
            continue

        # Enforce section diversity: max 3 from any single section
        section_counts: dict[str, int] = defaultdict(int)
        diverse_candidates: list[Chunk] = []

        for chunk in candidates:
            section_key = (chunk.section or "unknown").lower().strip()
            if section_counts[section_key] < 3:
                diverse_candidates.append(chunk)
                section_counts[section_key] += 1

        # Cap at max_per_paper
        selected.extend(diverse_candidates[:max_per_paper])

    return selected


def parse_passage_response(paper: Paper, chunk: Chunk, response_text: str) -> list[dict]:
    """Parse and validate LLM response into passage context dicts.

    Args:
        paper: The source paper
        chunk: The passage chunk
        response_text: Raw LLM response text

    Returns:
        List of validated context dicts with keys:
        id, paper_id, chunk_id, passage_text, citation_type, text, section
    """
    text = response_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    contexts_raw = data.get("contexts", [])
    if not isinstance(contexts_raw, list):
        return []

    title_lower = paper.title.lower()
    seen_types: set[str] = set()
    results: list[dict] = []

    # Extract chunk index from chunk ID
    try:
        _, chunk_idx = Chunk.parse_id(chunk.id)
    except ValueError:
        chunk_idx = 0

    for ctx in contexts_raw:
        if not isinstance(ctx, dict):
            continue

        ctype = ctx.get("type", "").strip().lower()
        ctext = ctx.get("text", "").strip()

        # Validate citation type
        if ctype not in VALID_TYPES:
            continue

        # Skip duplicate types
        if ctype in seen_types:
            continue

        # Must contain [CITE] marker
        if "[CITE]" not in ctext:
            continue

        # Must be at least 30 chars
        if len(ctext) < 30:
            continue

        # Must not contain paper title verbatim
        if title_lower in ctext.lower():
            continue

        seen_types.add(ctype)
        results.append(
            {
                "id": f"passage_{paper.id}_{chunk_idx}_{ctype}",
                "paper_id": paper.id,
                "chunk_id": chunk.id,
                "passage_text": chunk.text,
                "citation_type": ctype,
                "text": ctext,
                "section": chunk.section or "",
            }
        )

    return results


def generate_passage_contexts_batch(
    papers: list[Paper],
    chunks_by_paper: dict[str, list[Chunk]],
    api_key: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    max_per_paper: int = 10,
    poll_interval: int = 30,
) -> tuple[list[dict], dict]:
    """Generate passage-level citation contexts using Anthropic Batch API.

    One request per passage (clean 1:1 mapping).

    Args:
        papers: Papers to generate contexts for
        chunks_by_paper: Dict mapping paper_id -> list of chunks
        api_key: Anthropic API key (or ANTHROPIC_API_KEY env var)
        model: Model to use
        max_per_paper: Max passages per paper for selection
        poll_interval: Seconds between batch status checks

    Returns:
        Tuple of (contexts list, stats dict)
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

    # Build paper lookup
    papers_by_id = {p.id: p for p in papers}

    # Collect all chunks and select passages
    all_chunks: list[Chunk] = []
    for paper_id, chunks in chunks_by_paper.items():
        if paper_id in papers_by_id:
            all_chunks.extend(chunks)

    passages = select_passages(all_chunks, max_per_paper=max_per_paper)

    stats: dict = {
        "total_papers": len(papers),
        "total_chunks": len(all_chunks),
        "selected_passages": len(passages),
        "generated": 0,
        "contexts_created": 0,
        "failed": 0,
        "batch_id": None,
    }

    if not passages:
        return [], stats

    # Build batch requests
    passage_by_custom_id: dict[str, tuple[Paper, Chunk]] = {}
    requests = []

    for i, chunk in enumerate(passages):
        paper = papers_by_id.get(chunk.paper_id)
        if not paper:
            continue

        custom_id = f"passage_{i}"
        passage_by_custom_id[custom_id] = (paper, chunk)

        abstract_truncated = (paper.abstract or "")[:500]
        prompt = PASSAGE_PROMPT.format(
            title=paper.title,
            year=paper.year or "n.d.",
            abstract_truncated=abstract_truncated,
            section=chunk.section or "Unknown",
            passage_text=chunk.text,
        )

        requests.append(
            {
                "custom_id": custom_id,
                "params": {
                    "model": model,
                    "max_tokens": 512,
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
    all_contexts: list[dict] = []
    for entry in client.messages.batches.results(batch.id):
        pair = passage_by_custom_id.get(entry.custom_id)
        if not pair:
            continue

        paper, chunk = pair

        if entry.result.type == "succeeded":
            text = entry.result.message.content[0].text
            contexts = parse_passage_response(paper, chunk, text)
            if contexts:
                all_contexts.extend(contexts)
                stats["generated"] += 1
                stats["contexts_created"] += len(contexts)
            else:
                stats["failed"] += 1
        else:
            stats["failed"] += 1

    return all_contexts, stats


def contexts_to_training_examples(
    contexts: list[dict],
    papers: dict[str, Paper],
    chunks_by_paper: dict[str, list[Chunk]],
    max_hard_negatives: int = 3,
    seed: int = 42,
) -> list[TrainingExample]:
    """Convert generated passage contexts to TrainingExamples.

    Args:
        contexts: List of context dicts from generate_passage_contexts_batch
        papers: Dict mapping paper_id -> Paper
        chunks_by_paper: Dict mapping paper_id -> list of chunks
        max_hard_negatives: Number of hard negative passages (from same paper)
        seed: Random seed for reproducibility

    Returns:
        List of TrainingExample with passage-level ground truth
    """
    rng = random.Random(seed)
    examples: list[TrainingExample] = []

    for ctx in contexts:
        paper_id = ctx["paper_id"]
        paper = papers.get(paper_id)
        if not paper:
            continue

        # Clean citation markers from generated text
        query = clean_citation_markers(ctx["text"])

        # Build metadata prefix matching production chunk format
        metadata_prefix = format_paper_metadata_prefix(
            title=paper.title,
            author_lastnames=paper.author_lastnames,
            year=paper.year,
            journal=paper.journal,
        )

        # Passage positive: chunk text with metadata prefix (matches Chunk.to_embedding_text)
        raw_passage = ctx["passage_text"]
        passage_positive = format_passage_embedding_text(raw_passage, metadata_prefix)

        # Hard negatives: random other chunks from same paper (with metadata prefix)
        paper_chunks = chunks_by_paper.get(paper_id, [])
        other_chunks = [c for c in paper_chunks if c.id != ctx["chunk_id"]]
        if other_chunks and max_hard_negatives > 0:
            sample_size = min(max_hard_negatives, len(other_chunks))
            hard_neg_chunks = rng.sample(other_chunks, sample_size)
            passage_hard_negatives = [
                format_passage_embedding_text(c.text, metadata_prefix) for c in hard_neg_chunks
            ]
        else:
            passage_hard_negatives = []

        examples.append(
            TrainingExample(
                query=query,
                positive=passage_positive,
                hard_negatives=passage_hard_negatives,
                cited_paper_id=paper_id,
                source="passage_gen",
                passage_positive=passage_positive,
                passage_score=1.0,
                passage_validation=5,
                intent=ctx["citation_type"],
                passage_section=ctx["section"],
            )
        )

    return examples


def split_passage_data(
    examples: list[TrainingExample],
    dev_fraction: float = 0.1,
    eval_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[list[TrainingExample], list[TrainingExample], list[PassageTestExample]]:
    """Split passage data into train/dev/eval by paper ID.

    Splits by paper_id (not example) to prevent data leakage.

    Args:
        examples: All training examples
        dev_fraction: Fraction for dev set
        eval_fraction: Fraction for eval set (converted to PassageTestExample)
        seed: Random seed

    Returns:
        Tuple of (train_examples, dev_examples, eval_passage_test_examples)
    """
    rng = random.Random(seed)

    # Group by paper
    by_paper: dict[str, list[TrainingExample]] = defaultdict(list)
    for ex in examples:
        by_paper[ex.cited_paper_id].append(ex)

    paper_ids = sorted(by_paper.keys())
    rng.shuffle(paper_ids)

    n_papers = len(paper_ids)
    n_eval = max(1, int(n_papers * eval_fraction))
    n_dev = max(1, int(n_papers * dev_fraction))

    eval_papers = set(paper_ids[:n_eval])
    dev_papers = set(paper_ids[n_eval : n_eval + n_dev])
    train_papers = set(paper_ids[n_eval + n_dev :])

    train_examples: list[TrainingExample] = []
    dev_examples: list[TrainingExample] = []
    eval_test_examples: list[PassageTestExample] = []

    for pid in train_papers:
        train_examples.extend(by_paper[pid])

    for pid in dev_papers:
        dev_examples.extend(by_paper[pid])

    for pid in eval_papers:
        for ex in by_paper[pid]:
            eval_test_examples.append(
                PassageTestExample(
                    id=f"eval_{ex.cited_paper_id}_{ex.intent}",
                    citation_context=ex.query,
                    cited_paper_id=ex.cited_paper_id,
                    gold_passage=ex.passage_positive or ex.positive,
                    gold_passage_section=ex.passage_section,
                    intent=ex.intent,
                )
            )

    return train_examples, dev_examples, eval_test_examples
