"""Batch diagnosis pipeline for citation recommendation failure analysis.

Handles retrieval re-runs for failure queries and batch LLM diagnosis
via the Anthropic Batch API.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from incite.evaluation.failure_analysis import DiagnosisInput, DiagnosisResult
    from incite.models import CitationContext, Paper

logger = logging.getLogger(__name__)


def run_retrieval_for_failures(
    inputs: list[DiagnosisInput],
    corpus_papers: dict[str, Paper],
    test_set: dict[str, CitationContext],
    embedder_type: str = "granite-ft",
    method: str = "neural",
    k: int = 5,
    cache_path: Path | None = None,
) -> dict[str, list[dict]]:
    """Re-run retrieval for failure queries to get the model's actual predictions.

    For each failure query (is_success=False), runs the retriever and collects
    the top-k predictions (excluding ground truth).

    Args:
        inputs: List of DiagnosisInput objects (from failure_analysis).
        corpus_papers: Mapping of paper_id -> Paper for all corpus papers.
        test_set: Mapping of context_id -> CitationContext for reference_set_ids.
        embedder_type: Embedder type to use (e.g. "granite-ft").
        method: Retrieval method ("neural", "bm25", "hybrid").
        k: Number of top predictions to collect per failure query.
        cache_path: If provided and exists, load cached predictions.

    Returns:
        Mapping of context_id -> list of {paper_id, title, abstract, score, rank}.
    """
    # Return cached results if available
    if cache_path and cache_path.exists() and cache_path.stat().st_size > 0:
        logger.info("Loading cached predictions from %s", cache_path)
        predictions: dict[str, list[dict]] = {}
        with open(cache_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    predictions[entry["context_id"]] = entry["predictions"]
        logger.info("Loaded predictions for %d queries", len(predictions))
        return predictions

    from incite.retrieval.factory import create_retriever

    # Identify failure queries that need retrieval
    failure_inputs = [inp for inp in inputs if not inp.is_success]
    if not failure_inputs:
        logger.info("No failure queries to re-run retrieval for")
        return {}

    # Build one global retriever over all corpus papers
    all_papers = list(corpus_papers.values())
    logger.info(
        "Building %s retriever (%s) over %d papers...",
        method,
        embedder_type,
        len(all_papers),
    )
    retriever = create_retriever(
        all_papers,
        method=method,
        embedder_type=embedder_type,
        show_progress=True,
    )

    predictions: dict[str, list[dict]] = {}
    for inp in tqdm(failure_inputs, desc="Retrieving for failures"):
        ctx = test_set.get(inp.context_id)
        if ctx is None:
            logger.warning("No CitationContext found for %s, skipping", inp.context_id)
            continue

        # Get query text at narrow scale (matches eval default)
        query_text = ctx.get_query(scale="narrow", clean=True)

        # Retrieve a larger pool so we can filter to reference set
        results = retriever.retrieve(query_text, k=50)
        # Handle tuple return (some retrievers return (results, metadata))
        if isinstance(results, tuple):
            results = results[0]

        ref_set = set(ctx.reference_set_ids)
        gt_set = set(ctx.ground_truth_ids)

        # Filter to reference set, exclude ground truth, take top-k
        filtered = []
        for result in results:
            if result.paper_id in ref_set and result.paper_id not in gt_set:
                paper = corpus_papers.get(result.paper_id)
                filtered.append(
                    {
                        "paper_id": result.paper_id,
                        "title": paper.title if paper else "",
                        "abstract": (paper.abstract or "")[:300] if paper else "",
                        "score": round(result.score, 4),
                        "rank": len(filtered) + 1,
                    }
                )
                if len(filtered) >= k:
                    break

        predictions[inp.context_id] = filtered

    logger.info("Retrieved predictions for %d failure queries", len(predictions))

    # Save to cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            for context_id, preds in predictions.items():
                f.write(json.dumps({"context_id": context_id, "predictions": preds}) + "\n")
        logger.info("Saved predictions cache to %s", cache_path)

    return predictions


def submit_batch_diagnosis(
    inputs: list[DiagnosisInput],
    predictions: dict[str, list[dict]],
    output_path: Path,
    model: str | None = None,
    poll_interval: int = 30,
) -> list[DiagnosisResult]:
    """Submit all queries to Anthropic Batch API for LLM diagnosis.

    If output_path already exists with the expected number of results,
    loads and returns cached results.

    Args:
        inputs: All diagnosis inputs (successes + failures).
        predictions: Mapping of context_id -> model predictions for failures.
        output_path: Path to save/load diagnosis results.
        model: LLM model override (defaults to DEFAULT_LLM_MODEL).
        poll_interval: Seconds between batch status checks.

    Returns:
        List of DiagnosisResult objects.
    """
    from incite.evaluation.failure_analysis import (
        build_diagnosis_prompt,
        load_diagnoses,
        parse_diagnosis_response,
        save_diagnoses,
    )

    # Check for cached results
    if output_path.exists() and output_path.stat().st_size > 0:
        cached = load_diagnoses(output_path)
        if len(cached) >= len(inputs):
            logger.info("Loaded %d cached diagnoses from %s", len(cached), output_path)
            return cached
        logger.info("Cache has %d/%d results, re-running batch", len(cached), len(inputs))

    import anthropic

    from incite.corpus.batch_utils import poll_batch
    from incite.utils import DEFAULT_LLM_MODEL

    if model is None:
        model = DEFAULT_LLM_MODEL

    # Build batch requests
    # custom_id must match ^[a-zA-Z0-9_-]{1,64}$, so sanitize context_ids
    def _sanitize_id(cid: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]", "_", cid)[:64]

    batch_requests = []
    sanitized_to_original: dict[str, str] = {}
    for inp in inputs:
        preds = predictions.get(inp.context_id, [])
        prompt = build_diagnosis_prompt(inp, preds)
        safe_id = _sanitize_id(inp.context_id)
        sanitized_to_original[safe_id] = inp.context_id
        batch_requests.append(
            {
                "custom_id": safe_id,
                "params": {
                    "model": model,
                    "max_tokens": 300,
                    "temperature": 0.0,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
        )

    logger.info("Submitting batch of %d diagnosis requests...", len(batch_requests))
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=batch_requests)
    logger.info("Batch created: %s", batch.id)

    # Poll until complete
    poll_batch(client, batch.id, poll_interval)
    logger.info("Batch complete, retrieving results...")

    # Collect results keyed by context_id
    input_by_id = {inp.context_id: inp for inp in inputs}
    results: list[DiagnosisResult] = []

    for entry in client.messages.batches.results(batch.id):
        context_id = sanitized_to_original.get(entry.custom_id, entry.custom_id)
        inp = input_by_id.get(context_id)
        if inp is None:
            logger.warning("Unknown context_id in batch result: %s", context_id)
            continue

        if entry.result.type == "succeeded":
            text = entry.result.message.content[0].text
            preds = predictions.get(context_id, [])
            result = parse_diagnosis_response(context_id, text, inp.is_success, preds)
            results.append(result)
        else:
            logger.warning("Batch entry failed for %s: %s", context_id, entry.result.type)

    logger.info("Got %d/%d successful diagnoses", len(results), len(inputs))

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_diagnoses(results, output_path)
    logger.info("Saved diagnoses to %s", output_path)

    return results
