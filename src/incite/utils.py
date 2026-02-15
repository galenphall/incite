"""Shared utilities for inCite."""

import math
import re
import unicodedata
from collections import defaultdict

import torch

from incite.models import Paper, RetrievalResult

# Default LLM model for all enrichment tasks
DEFAULT_LLM_MODEL = "claude-haiku-4-5-20251001"


def get_best_device() -> str:
    """Auto-detect the best available device for PyTorch.

    Priority: MPS (Apple Silicon) > CUDA > CPU

    Returns:
        Device string: "mps", "cuda", or "cpu"
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def rrf_fuse(
    ranked_lists: list[tuple[list[RetrievalResult], float]],
    rrf_k: int = 10,
) -> dict[str, dict]:
    """Reciprocal Rank Fusion across multiple ranked lists.

    Args:
        ranked_lists: List of (results, weight) tuples
        rrf_k: RRF constant (smaller = more top-heavy ranking)

    Returns:
        Dict mapping paper_id -> {"total_score": float, "scores": dict}
    """
    all_results: dict[str, dict] = defaultdict(lambda: {"scores": {}, "total_score": 0.0})

    for results, weight in ranked_lists:
        for result in results:
            paper_id = result.paper_id
            rrf_score = weight / (rrf_k + result.rank)
            all_results[paper_id]["total_score"] += rrf_score
            all_results[paper_id]["scores"].update(result.score_breakdown)

    return dict(all_results)


def apply_author_boost(
    all_results: dict[str, dict],
    query: str,
    papers: dict[str, Paper],
    boost: float = 1.2,
) -> None:
    """Apply author-name boosting to fused results (in-place).

    Uses word-boundary matching to avoid false positives from substring
    matches (e.g., "Li" matching "limitations").

    Args:
        all_results: Dict mapping paper_id -> {"total_score": float, "scores": dict}
        query: Query text to search for author names
        papers: Dict mapping paper_id -> Paper
        boost: Score multiplier for author matches (1.0 = no boost)
    """
    if boost <= 1.0:
        return

    query_lower = query.lower()
    for paper_id, data in all_results.items():
        paper = papers.get(paper_id)
        if paper:
            for lastname in paper.author_lastnames:
                # Word-boundary matching to avoid substring false positives
                pattern = r"\b" + re.escape(lastname.lower()) + r"\b"
                if re.search(pattern, query_lower):
                    data["total_score"] *= boost
                    data["scores"]["author_match"] = lastname
                    break  # Only boost once per paper


def _normalize_title(title: str) -> str:
    """Normalize a paper title for deduplication.

    Lowercases, strips leading articles, collapses whitespace,
    removes trailing punctuation, and strips accents.

    Args:
        title: Raw paper title

    Returns:
        Normalized title string for comparison
    """
    # Lowercase
    t = title.lower()
    # Strip accents (é -> e, ü -> u, etc.)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    # Strip leading articles
    t = re.sub(r"^(the|a|an)\s+", "", t)
    # Remove non-alphanumeric (keep spaces)
    t = re.sub(r"[^a-z0-9\s]", "", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def deduplicate_results(
    results: list[RetrievalResult],
    papers: dict[str, Paper],
) -> list[RetrievalResult]:
    """Remove results with duplicate titles, keeping the highest-scored copy.

    When scores are within 5% of each other, prefers the copy with an abstract.

    Args:
        results: Sorted list of RetrievalResult
        papers: Dict mapping paper_id -> Paper

    Returns:
        Filtered list with duplicates removed, preserving order
    """
    seen_titles: dict[str, RetrievalResult] = {}
    output: list[RetrievalResult] = []

    for result in results:
        paper = papers.get(result.paper_id)
        if paper is None:
            output.append(result)
            continue

        norm = _normalize_title(paper.title)
        if not norm:
            output.append(result)
            continue

        if norm not in seen_titles:
            seen_titles[norm] = result
            output.append(result)
        else:
            # We already have a result with this title
            existing = seen_titles[norm]
            existing_paper = papers.get(existing.paper_id)

            # If scores are close, prefer the one with an abstract
            if existing.score > 0 and result.score / existing.score > 0.95:
                has_abstract_new = paper.abstract and len(paper.abstract) > 0
                has_abstract_existing = (
                    existing_paper and existing_paper.abstract and len(existing_paper.abstract) > 0
                )
                if has_abstract_new and not has_abstract_existing:
                    # Replace existing with this better copy
                    output = [r for r in output if r.paper_id != existing.paper_id]
                    output.append(result)
                    seen_titles[norm] = result

    return output


def compute_confidence(
    score_breakdown: dict[str, float],
    mode: str = "hybrid",
) -> float:
    """Compute a confidence score in [0, 1] from the score breakdown.

    For hybrid mode, uses the neural cosine similarity (already 0-1).
    For paragraph mode, uses the best chunk score.
    For BM25-only, applies sigmoid normalization.

    Args:
        score_breakdown: Dict with component scores from retrieval
        mode: "hybrid", "paragraph", or "bm25"

    Returns:
        Confidence float in [0, 1]
    """
    if mode == "paragraph":
        return float(score_breakdown.get("best_chunk_score", 0.0))

    if mode == "two_stage":
        # Blend of paper neural score and best chunk score
        chunk = float(score_breakdown.get("best_chunk_score", 0.0))
        neural = float(score_breakdown.get("neural", 0.0))
        return max(chunk, neural) if chunk > 0 or neural > 0 else 0.0

    if mode == "multi_scale":
        # Prefer best_evidence_score (raw chunk cosine sim, already 0-1).
        # Fall back to paper_score (also raw neural cosine sim).
        # paragraph_score/sentence_score are aggregated paper-level scores
        # from ParagraphRetriever, not raw similarities — don't use for confidence.
        evidence = score_breakdown.get("best_evidence_score", 0.0)
        paper = score_breakdown.get("paper_score", 0.0)
        return max(float(evidence), float(paper))

    neural = score_breakdown.get("neural", 0.0)
    if neural > 0:
        return float(neural)

    # For neural_score from paragraph hybrid mode
    neural_score = score_breakdown.get("neural_score", 0.0)
    if neural_score > 0:
        return float(neural_score)

    # BM25-only: sigmoid normalize (midpoint ~10, typical range 0-30)
    bm25 = score_breakdown.get("bm25", 0.0) or score_breakdown.get("bm25_score", 0.0)
    if bm25 > 0:
        return 1.0 / (1.0 + math.exp(-0.2 * (bm25 - 10)))

    return 0.0
