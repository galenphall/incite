"""Evaluation metric models for retrieval benchmarking.

Contains EvaluationResult and QueryResult dataclasses used by the evaluation
pipeline to track per-query and aggregate retrieval metrics (MRR, Recall@K,
NDCG, concordance index, skill MRR).

Related modules:
    - incite.models: Core data models (Paper, Chunk, CitationContext)
    - incite.cli.core: Evaluation CLI commands that produce these results
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QueryResult:
    """Per-query evaluation result tracking scores and ranking position."""

    context_id: str
    source_paper_id: Optional[str]
    ground_truth_ids: list[str]
    scores: dict[str, float]  # metric_name -> value
    first_relevant_rank: Optional[int] = None  # 1-indexed, None if not found

    def to_dict(self) -> dict:
        return {
            "context_id": self.context_id,
            "source_paper_id": self.source_paper_id,
            "ground_truth_ids": self.ground_truth_ids,
            "scores": self.scores,
            "first_relevant_rank": self.first_relevant_rank,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QueryResult":
        return cls(
            context_id=data["context_id"],
            source_paper_id=data.get("source_paper_id"),
            ground_truth_ids=data["ground_truth_ids"],
            scores=data["scores"],
            first_relevant_rank=data.get("first_relevant_rank"),
        )


@dataclass
class EvaluationResult:
    """Aggregate evaluation metrics across all queries in a benchmark run."""

    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    recall_at_50: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg_at_10: float = 0.0
    concordance: float = 0.0  # C-index: P(correct ranked above random incorrect)
    skill_mrr: float = 0.0  # MRR normalized to 0=random, 1=perfect
    num_queries: int = 0
    per_query: list[QueryResult] = field(default_factory=list, repr=False)
    # Evidence quality metrics (OpenScholar citation accuracy)
    evidence_precision: float = 0.0
    evidence_recall: float = 0.0
    evidence_f1: float = 0.0
    # Two-stage retrieval metrics
    evidence_coverage: float = 0.0  # Fraction of correct papers with evidence attached
    mean_best_chunk_score: float = 0.0  # Average best_chunk_score for correct papers

    def to_dict(self) -> dict[str, float]:
        # per_query intentionally excluded — backward-compatible with experiment log
        d = {
            "recall@1": self.recall_at_1,
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "recall@20": self.recall_at_20,
            "recall@50": self.recall_at_50,
            "mrr": self.mrr,
            "ndcg@10": self.ndcg_at_10,
            "num_queries": self.num_queries,
        }
        # Include corpus-adjusted metrics only when computed (backward compat)
        if self.concordance > 0 or self.skill_mrr != 0:
            d["concordance"] = self.concordance
            d["skill_mrr"] = self.skill_mrr
        # Include evidence metrics only if populated (backward compat)
        if self.evidence_precision > 0 or self.evidence_recall > 0:
            d["evidence_precision"] = self.evidence_precision
            d["evidence_recall"] = self.evidence_recall
            d["evidence_f1"] = self.evidence_f1
        # Include two-stage metrics only if populated (backward compat)
        if self.evidence_coverage > 0 or self.mean_best_chunk_score > 0:
            d["evidence_coverage"] = self.evidence_coverage
            d["mean_best_chunk_score"] = self.mean_best_chunk_score
        return d

    def _format_metric(self, name: str, value: float, ci: tuple[float, float] | None) -> str:
        if ci is not None:
            return f"  {name} {value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
        return f"  {name} {value:.3f}"

    def __str__(self) -> str:
        # Compute CIs if per-query data is available
        cis: dict[str, tuple[float, float] | None] = {}
        if self.per_query:
            metric_map = {
                "Recall@1:": "recall@1",
                "Recall@5:": "recall@5",
                "Recall@10:": "recall@10",
                "Recall@20:": "recall@20",
                "Recall@50:": "recall@50",
                "MRR:": "mrr",
                "NDCG@10:": "ndcg@10",
            }
            for label, key in metric_map.items():
                scores = [qr.scores.get(key, 0.0) for qr in self.per_query]
                cis[label] = _bootstrap_ci(scores)

            # CIs for corpus-adjusted metrics (only if computed)
            if self.concordance > 0 or self.skill_mrr != 0:
                for label, key in [
                    ("C-index:", "concordance"),
                    ("Skill MRR:", "skill_mrr"),
                ]:
                    scores = [qr.scores.get(key, 0.0) for qr in self.per_query]
                    cis[label] = _bootstrap_ci(scores)
        else:
            for label in [
                "Recall@1:",
                "Recall@5:",
                "Recall@10:",
                "Recall@20:",
                "Recall@50:",
                "MRR:",
                "NDCG@10:",
            ]:
                cis[label] = None

        lines = [
            f"Evaluation Results (n={self.num_queries}):",
            self._format_metric("Recall@1: ", self.recall_at_1, cis["Recall@1:"]),
            self._format_metric("Recall@5: ", self.recall_at_5, cis["Recall@5:"]),
            self._format_metric("Recall@10:", self.recall_at_10, cis["Recall@10:"]),
            self._format_metric("Recall@20:", self.recall_at_20, cis["Recall@20:"]),
            self._format_metric("Recall@50:", self.recall_at_50, cis["Recall@50:"]),
            self._format_metric("MRR:      ", self.mrr, cis["MRR:"]),
            self._format_metric("NDCG@10:  ", self.ndcg_at_10, cis["NDCG@10:"]),
        ]
        if self.concordance > 0 or self.skill_mrr != 0:
            lines.append("")
            lines.append("Corpus-size-adjusted (0=random, 1=perfect):")
            lines.append(self._format_metric("C-index:  ", self.concordance, cis.get("C-index:")))
            lines.append(self._format_metric("Skill MRR:", self.skill_mrr, cis.get("Skill MRR:")))
        if self.evidence_precision > 0 or self.evidence_recall > 0:
            lines.append("")
            lines.append("Evidence Quality (OpenScholar citation accuracy):")
            lines.append(f"  Evidence Precision: {self.evidence_precision:.3f}")
            lines.append(f"  Evidence Recall:    {self.evidence_recall:.3f}")
            lines.append(f"  Evidence F1:        {self.evidence_f1:.3f}")
        if self.evidence_coverage > 0 or self.mean_best_chunk_score > 0:
            lines.append("")
            lines.append("Two-stage retrieval:")
            lines.append(f"  Evidence Coverage:    {self.evidence_coverage:.1%}")
            lines.append(f"  Mean Best Chunk Score: {self.mean_best_chunk_score:.3f}")
        return "\n".join(lines)


def _bootstrap_ci(
    scores: Sequence[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a metric.

    Resamples the per-query scores with replacement to estimate the sampling
    distribution of the mean, then returns the percentile-based CI.
    """
    import numpy as np

    scores_arr = np.array(scores, dtype=np.float64)
    if len(scores_arr) == 0:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    n = len(scores_arr)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = scores_arr[indices].mean(axis=1)

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lower, upper)
