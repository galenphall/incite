"""Passage-level evaluation metrics for citation retrieval.

Measures both paper-level recall (did we find the right paper?) and
passage-level recall (did we find the right passage within that paper?).
Supports intent-stratified metrics for analyzing performance by citation type.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from incite.models import Chunk


class _Embedder(Protocol):
    """Minimal embedder protocol for passage retrieval."""

    def embed_query(self, text: str) -> np.ndarray: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class PassageTestExample:
    """A test example with passage-level ground truth."""

    id: str
    citation_context: str
    cited_paper_id: str
    gold_passage: str
    gold_passage_section: str
    intent: str  # background/method/result/comparison/motivation
    source_paper_id: str = ""
    reference_set_ids: list[str] = field(default_factory=list)


@dataclass
class PassageEvaluationResult:
    """Aggregated passage-level evaluation metrics."""

    paper_recall_at_10: float = 0.0
    passage_recall_at_1: float = 0.0
    passage_recall_at_5: float = 0.0
    passage_recall_at_10: float = 0.0
    passage_mrr: float = 0.0
    passage_f1: float = 0.0  # Average token F1 between retrieved and gold passage
    by_intent: dict[str, dict[str, float]] = field(default_factory=dict)
    num_queries: int = 0

    def __str__(self) -> str:
        lines = [
            f"Passage Evaluation Results (n={self.num_queries}):",
            f"  Paper Recall@10:   {self.paper_recall_at_10:.3f}",
            f"  Passage Recall@1:  {self.passage_recall_at_1:.3f}",
            f"  Passage Recall@5:  {self.passage_recall_at_5:.3f}",
            f"  Passage Recall@10: {self.passage_recall_at_10:.3f}",
            f"  Passage MRR:       {self.passage_mrr:.3f}",
            f"  Passage F1:        {self.passage_f1:.3f}",
        ]
        if self.by_intent:
            lines.append("  By intent:")
            for intent, metrics in sorted(self.by_intent.items()):
                lines.append(
                    f"    {intent}: R@1={metrics.get('passage_recall_at_1', 0):.3f} "
                    f"R@5={metrics.get('passage_recall_at_5', 0):.3f} "
                    f"F1={metrics.get('passage_f1', 0):.3f} "
                    f"(n={metrics.get('num_queries', 0)})"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "paper_recall_at_10": self.paper_recall_at_10,
            "passage_recall_at_1": self.passage_recall_at_1,
            "passage_recall_at_5": self.passage_recall_at_5,
            "passage_recall_at_10": self.passage_recall_at_10,
            "passage_mrr": self.passage_mrr,
            "passage_f1": self.passage_f1,
            "by_intent": self.by_intent,
            "num_queries": self.num_queries,
        }


def passage_token_f1(predicted: str, gold: str) -> float:
    """Compute token-level F1 between predicted and gold passage.

    Uses whitespace tokenization. Measures overlap between the two passages.

    Args:
        predicted: Retrieved passage text
        gold: Ground-truth passage text

    Returns:
        F1 score in [0, 1]
    """
    pred_tokens = set(predicted.lower().split())
    gold_tokens = set(gold.lower().split())

    if not pred_tokens or not gold_tokens:
        return 0.0

    overlap = pred_tokens & gold_tokens
    if not overlap:
        return 0.0

    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def _passage_matches_gold(predicted: str, gold: str, f1_threshold: float = 0.5) -> bool:
    """Check if a predicted passage matches the gold passage.

    Uses token F1 threshold to determine a match, allowing for
    minor differences in chunking boundaries.
    """
    return passage_token_f1(predicted, gold) >= f1_threshold


def _rank_passages_by_embedding(
    embedder: _Embedder, query: str, chunks: list[Chunk], k: int
) -> list[tuple[str, float]]:
    """Rank passages by cosine similarity using embedder.

    Returns list of (chunk_text, score) sorted by score descending.
    """
    query_emb = embedder.embed_query(query)
    chunk_texts = [c.text for c in chunks]
    chunk_embs = embedder.embed(chunk_texts)

    # Cosine similarity
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    chunk_norms = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-10)
    scores = chunk_norms @ query_norm

    top_indices = np.argsort(scores)[::-1][:k]
    return [(chunk_texts[i], float(scores[i])) for i in top_indices]


def evaluate_passage_retrieval(
    embedder: Optional[_Embedder],
    test_set: list[PassageTestExample],
    chunks_by_paper: dict[str, list[Chunk]],
    k: int = 10,
    f1_threshold: float = 0.5,
) -> PassageEvaluationResult:
    """Evaluate passage-level retrieval on a test set.

    For each query:
    1. Use embedder to find top-k passages from the cited paper by cosine similarity
    2. Check if any retrieved passage matches the gold passage (token F1 >= threshold)
    3. Compute recall@k, MRR, and average F1

    If embedder is None, uses token F1 of each chunk vs query as score
    (useful for unit tests without loading a real model).

    Args:
        embedder: Embedder instance (or None for F1-only mode)
        test_set: List of PassageTestExample with gold passages
        chunks_by_paper: Dict mapping paper_id -> list of chunks
        k: Number of passages to retrieve per query
        f1_threshold: Token F1 threshold for a passage match

    Returns:
        PassageEvaluationResult with aggregated metrics
    """
    if not test_set:
        return PassageEvaluationResult()

    metrics: dict[str, float] = defaultdict(float)
    intent_metrics: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    intent_counts: dict[str, int] = defaultdict(int)

    valid_queries = 0

    for example in test_set:
        chunks = chunks_by_paper.get(example.cited_paper_id, [])
        if not chunks:
            continue

        # Retrieve passages
        if embedder is not None:
            ranked = _rank_passages_by_embedding(embedder, example.citation_context, chunks, k)
        else:
            # F1-only fallback for tests: rank by token F1 with query
            f1_scores = [
                (c.text, passage_token_f1(c.text, example.citation_context)) for c in chunks
            ]
            f1_scores.sort(key=lambda x: x[1], reverse=True)
            ranked = f1_scores[:k]

        if not ranked:
            valid_queries += 1
            continue

        # Check passage recall at various k
        found_at: Optional[int] = None  # 1-indexed rank of first match
        best_f1 = 0.0

        for i, (passage_text, _score) in enumerate(ranked):
            f1 = passage_token_f1(passage_text, example.gold_passage)
            best_f1 = max(best_f1, f1)
            if found_at is None and f1 >= f1_threshold:
                found_at = i + 1

        # Paper-level: always 1.0 since we're given the paper's chunks
        metrics["paper_recall_at_10"] += 1.0

        # Passage recall
        if found_at is not None:
            if found_at <= 1:
                metrics["passage_recall_at_1"] += 1.0
            if found_at <= 5:
                metrics["passage_recall_at_5"] += 1.0
            if found_at <= 10:
                metrics["passage_recall_at_10"] += 1.0
            metrics["passage_mrr"] += 1.0 / found_at
        metrics["passage_f1"] += best_f1

        # Intent-stratified metrics
        intent = example.intent
        if intent:
            intent_counts[intent] += 1
            intent_metrics[intent]["passage_f1"] += best_f1
            if found_at is not None:
                if found_at <= 1:
                    intent_metrics[intent]["passage_recall_at_1"] += 1.0
                if found_at <= 5:
                    intent_metrics[intent]["passage_recall_at_5"] += 1.0

        valid_queries += 1

    if valid_queries == 0:
        return PassageEvaluationResult()

    # Average intent metrics
    by_intent: dict[str, dict[str, float]] = {}
    for intent, intent_m in intent_metrics.items():
        n = intent_counts[intent]
        if n > 0:
            by_intent[intent] = {
                "passage_recall_at_1": intent_m["passage_recall_at_1"] / n,
                "passage_recall_at_5": intent_m["passage_recall_at_5"] / n,
                "passage_f1": intent_m["passage_f1"] / n,
                "num_queries": n,
            }

    return PassageEvaluationResult(
        paper_recall_at_10=metrics["paper_recall_at_10"] / valid_queries,
        passage_recall_at_1=metrics["passage_recall_at_1"] / valid_queries,
        passage_recall_at_5=metrics["passage_recall_at_5"] / valid_queries,
        passage_recall_at_10=metrics["passage_recall_at_10"] / valid_queries,
        passage_mrr=metrics["passage_mrr"] / valid_queries,
        passage_f1=metrics["passage_f1"] / valid_queries,
        by_intent=by_intent,
        num_queries=valid_queries,
    )


def load_passage_test_set(path: str | Path) -> list[PassageTestExample]:
    """Load passage-level test set from JSONL file.

    Args:
        path: Path to passage_test_set.jsonl

    Returns:
        List of PassageTestExample
    """
    examples = []
    path = Path(path)

    with open(path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                examples.append(
                    PassageTestExample(
                        id=data["id"],
                        citation_context=data["citation_context"],
                        cited_paper_id=data["cited_paper_id"],
                        gold_passage=data["gold_passage"],
                        gold_passage_section=data.get("gold_passage_section", ""),
                        intent=data.get("intent", ""),
                        source_paper_id=data.get("source_paper_id", ""),
                        reference_set_ids=data.get("reference_set_ids", []),
                    )
                )

    return examples


def evaluate_evidence_quality(
    results_with_evidence: list[tuple[list, str, str]],
    f1_threshold: float = 0.3,
) -> dict[str, float]:
    """Evaluate the quality of attached evidence snippets.

    Inspired by OpenScholar's "citation F1" metric (Asai et al., 2026), which
    measures whether cited evidence actually supports claims. Our proxy:
    does the returned evidence snippet overlap with the ground truth passage?

    Args:
        results_with_evidence: List of (retrieval_results, gold_passage, cited_paper_id)
            tuples. retrieval_results should have matched_paragraph populated.
        f1_threshold: Token F1 threshold for considering evidence "correct"

    Returns:
        Dict with:
        - evidence_precision: fraction of attached evidence with token F1 >= threshold
        - evidence_recall: fraction of correct papers in top-k that have evidence
        - evidence_f1: harmonic mean of precision and recall
        - mean_evidence_token_f1: average token F1 across all evidence snippets
        - num_queries: number of queries evaluated
    """
    total_evidence_attached = 0
    correct_evidence = 0
    correct_papers_found = 0
    correct_papers_with_evidence = 0
    total_f1 = 0.0
    num_queries = 0

    for results, gold_passage, cited_paper_id in results_with_evidence:
        num_queries += 1

        # Check if cited paper is in results
        paper_found = False
        paper_has_evidence = False

        for result in results:
            if result.paper_id == cited_paper_id:
                paper_found = True

                # Check if evidence is attached
                if result.matched_paragraph:
                    paper_has_evidence = True
                    total_evidence_attached += 1
                    f1 = passage_token_f1(result.matched_paragraph, gold_passage)
                    total_f1 += f1
                    if f1 >= f1_threshold:
                        correct_evidence += 1

                # Also check matched_paragraphs (multi-evidence)
                if hasattr(result, "matched_paragraphs") and result.matched_paragraphs:
                    for para_info in result.matched_paragraphs[1:]:  # Skip first (already counted)
                        text = para_info.get("text", "") if isinstance(para_info, dict) else ""
                        if text:
                            total_evidence_attached += 1
                            f1 = passage_token_f1(text, gold_passage)
                            total_f1 += f1
                            if f1 >= f1_threshold:
                                correct_evidence += 1

                break  # Only check the target paper

        if paper_found:
            correct_papers_found += 1
            if paper_has_evidence:
                correct_papers_with_evidence += 1

    # Compute metrics
    evidence_precision = (
        correct_evidence / total_evidence_attached if total_evidence_attached > 0 else 0.0
    )
    evidence_recall = (
        correct_papers_with_evidence / correct_papers_found if correct_papers_found > 0 else 0.0
    )
    evidence_f1 = (
        2 * evidence_precision * evidence_recall / (evidence_precision + evidence_recall)
        if (evidence_precision + evidence_recall) > 0
        else 0.0
    )
    mean_f1 = total_f1 / total_evidence_attached if total_evidence_attached > 0 else 0.0

    return {
        "evidence_precision": evidence_precision,
        "evidence_recall": evidence_recall,
        "evidence_f1": evidence_f1,
        "mean_evidence_token_f1": mean_f1,
        "num_queries": num_queries,
        "total_evidence_snippets": total_evidence_attached,
        "correct_evidence_snippets": correct_evidence,
        "correct_papers_found": correct_papers_found,
        "correct_papers_with_evidence": correct_papers_with_evidence,
    }


def save_passage_test_set(examples: list[PassageTestExample], path: str | Path) -> None:
    """Save passage-level test set to JSONL file.

    Args:
        examples: List of PassageTestExample
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for ex in examples:
            data = {
                "id": ex.id,
                "citation_context": ex.citation_context,
                "cited_paper_id": ex.cited_paper_id,
                "gold_passage": ex.gold_passage,
                "gold_passage_section": ex.gold_passage_section,
                "intent": ex.intent,
                "source_paper_id": ex.source_paper_id,
                "reference_set_ids": ex.reference_set_ids,
            }
            f.write(json.dumps(data) + "\n")
