"""Core analysis module for LLM-powered citation recommendation failure diagnosis.

Loads evaluation results, builds LLM prompts to classify citation intent and
failure modes, and persists the diagnosis results as JSONL.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from incite.corpus.loader import load_corpus, load_test_set
from incite.evaluation.experiment_log import ExperimentLogger
from incite.evaluation.intent_taxonomy import (
    CITATION_INTENTS,
    DIFFICULTY_LEVELS,
    FAILURE_MODES,
    VALID_DIFFICULTIES,
    VALID_FAILURE_MODES,
    VALID_INTENTS,
)
from incite.models import QueryResult

logger = logging.getLogger(__name__)

__all__ = [
    "DiagnosisInput",
    "DiagnosisResult",
    "load_diagnosis_inputs",
    "build_diagnosis_prompt",
    "parse_diagnosis_response",
    "save_diagnoses",
    "load_diagnoses",
    "generate_intent_report",
]


@dataclass
class DiagnosisInput:
    """Input data for diagnosing a single evaluation query."""

    context_id: str
    query_text: str  # local_context from CitationContext
    source_paper_id: str
    gt_paper_ids: list[str]
    gt_paper_title: str  # title of the first GT paper
    gt_paper_abstract: str  # abstract of the first GT paper
    is_success: bool  # R@1 == 1.0
    first_relevant_rank: int | None
    scores: dict[str, float]


@dataclass
class DiagnosisResult:
    """LLM-generated diagnosis for a single evaluation query."""

    context_id: str
    intent: str  # from CITATION_INTENTS
    failure_mode: str  # from FAILURE_MODES, "" if success
    difficulty: str  # easy/hard/ambiguous
    reasoning: str  # 1-2 sentence LLM explanation
    top_predictions: list[dict] = field(default_factory=list)
    is_success: bool = False

    def to_dict(self) -> dict:
        return {
            "context_id": self.context_id,
            "intent": self.intent,
            "failure_mode": self.failure_mode,
            "difficulty": self.difficulty,
            "reasoning": self.reasoning,
            "top_predictions": self.top_predictions,
            "is_success": self.is_success,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DiagnosisResult:
        return cls(
            context_id=data["context_id"],
            intent=data["intent"],
            failure_mode=data.get("failure_mode", ""),
            difficulty=data.get("difficulty", ""),
            reasoning=data.get("reasoning", ""),
            top_predictions=data.get("top_predictions", []),
            is_success=data.get("is_success", False),
        )


def load_diagnosis_inputs(
    run_id: str,
    test_set_path: str | Path,
    corpus_path: str | Path,
) -> list[DiagnosisInput]:
    """Load and join evaluation results with test set and corpus metadata.

    Args:
        run_id: Experiment run ID (or prefix) for loading per-query results.
        test_set_path: Path to test_set.jsonl.
        corpus_path: Path to corpus.jsonl.

    Returns:
        List of DiagnosisInput objects ready for prompt construction.
    """
    # Load per-query results
    exp_logger = ExperimentLogger(Path("data/experiments/experiments.jsonl"))
    query_results = exp_logger.load_per_query(run_id)
    if not query_results:
        raise ValueError(f"No per-query results found for run_id={run_id!r}")

    qr_by_id: dict[str, QueryResult] = {qr.context_id: qr for qr in query_results}

    # Load test set and corpus using existing loaders
    contexts = load_test_set(test_set_path)
    ctx_by_id = {ctx.id: ctx for ctx in contexts}

    papers = load_corpus(corpus_path)
    paper_dict = {p.id: p for p in papers}

    # Join into DiagnosisInputs
    inputs: list[DiagnosisInput] = []
    for context_id, qr in qr_by_id.items():
        ctx = ctx_by_id.get(context_id)
        if ctx is None:
            logger.warning("Context %s not found in test set, skipping", context_id)
            continue

        gt_ids = qr.ground_truth_ids
        gt_paper = paper_dict.get(gt_ids[0]) if gt_ids else None

        is_success = qr.scores.get("recall@1", 0.0) == 1.0

        inputs.append(
            DiagnosisInput(
                context_id=context_id,
                query_text=ctx.local_context,
                source_paper_id=qr.source_paper_id or "",
                gt_paper_ids=gt_ids,
                gt_paper_title=gt_paper.title if gt_paper else "",
                gt_paper_abstract=gt_paper.abstract if gt_paper else "",
                is_success=is_success,
                first_relevant_rank=qr.first_relevant_rank,
                scores=qr.scores,
            )
        )

    logger.info(
        "Loaded %d diagnosis inputs (%d successes, %d failures)",
        len(inputs),
        sum(1 for i in inputs if i.is_success),
        sum(1 for i in inputs if not i.is_success),
    )
    return inputs


def build_diagnosis_prompt(
    inp: DiagnosisInput,
    predictions: list[dict] | None = None,
) -> str:
    """Build an LLM prompt for diagnosing a single evaluation query.

    Args:
        inp: Diagnosis input with query text, GT paper info, and success flag.
        predictions: For failures, list of dicts with keys
            {paper_id, title, abstract, score, rank} for the model's top-3.

    Returns:
        Prompt string requesting structured JSON classification.
    """
    intent_values = ", ".join(CITATION_INTENTS)
    difficulty_values = ", ".join(DIFFICULTY_LEVELS)

    if inp.is_success:
        return (
            "You are analyzing a citation recommendation. A researcher wrote text "
            "that cites a specific paper, and the system correctly identified it.\n\n"
            f"**Citation context**: {inp.query_text}\n"
            f"**Correct paper**: Title: {inp.gt_paper_title}  "
            f"Abstract: {inp.gt_paper_abstract}\n\n"
            "Classify the citation intent. Return JSON only:\n"
            '{"intent": "...", "difficulty": "...", "reasoning": "..."}\n\n'
            f"Valid intent values: {intent_values}\n"
            f"Valid difficulty values: {difficulty_values}"
        )

    # Failure prompt
    failure_values = ", ".join(FAILURE_MODES)
    pred_lines = []
    for i, pred in enumerate((predictions or [])[:3], 1):
        title = pred.get("title", "Unknown")
        abstract = pred.get("abstract", "")[:200]
        pred_lines.append(f"{i}. {title} — {abstract}")
    pred_block = "\n".join(pred_lines) if pred_lines else "(no predictions available)"

    return (
        "You are analyzing a citation recommendation failure. A researcher wrote "
        "text that cites a specific paper, but the recommendation system ranked "
        "it incorrectly.\n\n"
        f"**Citation context** (the text that should lead to the cited paper):\n"
        f"{inp.query_text}\n\n"
        f"**Correct paper** (ground truth):\n"
        f"Title: {inp.gt_paper_title}\n"
        f"Abstract: {inp.gt_paper_abstract}\n\n"
        f"**Model's top predictions** (what the system recommended instead):\n"
        f"{pred_block}\n\n"
        "Classify this case. Return JSON only:\n"
        '{"intent": "...", "failure_mode": "...", "difficulty": "...", '
        '"reasoning": "..."}\n\n'
        f"Valid intent values: {intent_values}\n"
        f"Valid failure_mode values: {failure_values}\n"
        f"Valid difficulty values: {difficulty_values}"
    )


def parse_diagnosis_response(
    context_id: str,
    response_text: str,
    is_success: bool,
    predictions: list[dict] | None = None,
) -> DiagnosisResult:
    """Parse and validate an LLM diagnosis response.

    Args:
        context_id: The evaluation query context ID.
        response_text: Raw LLM response text (expected JSON, may have fences).
        is_success: Whether this query was a success (R@1=1).
        predictions: Model's top predictions for inclusion in the result.

    Returns:
        Validated DiagnosisResult.
    """
    # Strip markdown code fences if present
    cleaned = response_text.strip()
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON for context %s: %s", context_id, response_text[:200])
        return DiagnosisResult(
            context_id=context_id,
            intent="unclassified",
            failure_mode="" if is_success else "unclassified",
            difficulty="unclassified",
            reasoning=f"Parse error: {response_text[:200]}",
            top_predictions=predictions or [],
            is_success=is_success,
        )

    intent = data.get("intent", "unclassified")
    if intent not in VALID_INTENTS:
        logger.warning("Invalid intent %r for context %s", intent, context_id)
        intent = "unclassified"

    failure_mode = "" if is_success else data.get("failure_mode", "unclassified")
    if failure_mode not in VALID_FAILURE_MODES:
        logger.warning("Invalid failure_mode %r for context %s", failure_mode, context_id)
        failure_mode = "unclassified"

    difficulty = data.get("difficulty", "unclassified")
    if difficulty not in VALID_DIFFICULTIES:
        logger.warning("Invalid difficulty %r for context %s", difficulty, context_id)
        difficulty = "unclassified"

    return DiagnosisResult(
        context_id=context_id,
        intent=intent,
        failure_mode=failure_mode,
        difficulty=difficulty,
        reasoning=data.get("reasoning", ""),
        top_predictions=predictions or [],
        is_success=is_success,
    )


def save_diagnoses(diagnoses: list[DiagnosisResult], path: Path) -> None:
    """Write diagnosis results to a JSONL file.

    Args:
        diagnoses: List of DiagnosisResult objects.
        path: Output JSONL file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for d in diagnoses:
            f.write(json.dumps(d.to_dict()) + "\n")
    logger.info("Saved %d diagnoses to %s", len(diagnoses), path)


def load_diagnoses(path: Path) -> list[DiagnosisResult]:
    """Read diagnosis results from a JSONL file.

    Args:
        path: Path to JSONL file with diagnosis results.

    Returns:
        List of DiagnosisResult objects.
    """
    results: list[DiagnosisResult] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(DiagnosisResult.from_dict(json.loads(line)))
    logger.info("Loaded %d diagnoses from %s", len(results), path)
    return results


def generate_intent_report(
    diagnoses: list[DiagnosisResult],
    query_results: list[QueryResult],
) -> str:
    """Generate a markdown report analyzing citation intents and failure modes.

    Args:
        diagnoses: LLM diagnosis results with intent labels.
        query_results: Per-query evaluation results.

    Returns:
        Markdown-formatted report string.
    """
    if not diagnoses:
        return "# Citation Recommendation Failure Analysis Report\n\nNo diagnoses to report."

    # Build lookup from context_id -> query scores
    qr_by_id: dict[str, QueryResult] = {qr.context_id: qr for qr in query_results}

    total = len(diagnoses)
    successes = sum(1 for d in diagnoses if d.is_success)
    failures = total - successes
    success_rate = successes / total if total > 0 else 0.0

    lines = [
        "# Citation Recommendation Failure Analysis Report",
        "",
        "## Summary",
        "",
        f"- **Total queries**: {total}",
        f"- **Successes** (R@1=1): {successes} ({successes / total * 100:.1f}%)",
        f"- **Failures** (R@1=0): {failures} ({failures / total * 100:.1f}%)",
        f"- **Success rate**: {success_rate:.1%}",
        "",
    ]

    # Intent distribution with metrics
    intent_groups: dict[str, list[DiagnosisResult]] = defaultdict(list)
    for d in diagnoses:
        intent_groups[d.intent].append(d)

    lines.extend(
        [
            "## Intent Distribution",
            "",
            "| Intent | Count | % | MRR | R@1 | R@10 |",
            "|--------|------:|--:|----:|----:|-----:|",
        ]
    )

    intent_mrrs: dict[str, float] = {}
    for intent in sorted(intent_groups.keys()):
        group = intent_groups[intent]
        count = len(group)
        pct = count / total * 100

        mrr_vals = []
        r1_vals = []
        r10_vals = []
        for d in group:
            qr = qr_by_id.get(d.context_id)
            if qr:
                mrr_vals.append(qr.scores.get("mrr", 0.0))
                r1_vals.append(qr.scores.get("recall@1", 0.0))
                r10_vals.append(qr.scores.get("recall@10", 0.0))

        avg_mrr = sum(mrr_vals) / len(mrr_vals) if mrr_vals else 0.0
        avg_r1 = sum(r1_vals) / len(r1_vals) if r1_vals else 0.0
        avg_r10 = sum(r10_vals) / len(r10_vals) if r10_vals else 0.0
        intent_mrrs[intent] = avg_mrr

        lines.append(
            f"| {intent} | {count} | {pct:.1f}% | {avg_mrr:.3f} | {avg_r1:.3f} | {avg_r10:.3f} |"
        )

    lines.append("")

    # Failure mode distribution (failures only)
    fm_counts: Counter[str] = Counter()
    if failures > 0:
        for d in diagnoses:
            if not d.is_success and d.failure_mode:
                fm_counts[d.failure_mode] += 1

        lines.extend(
            [
                "## Failure Mode Distribution",
                "",
                "| Failure Mode | Count | % of Failures |",
                "|-------------|------:|--------------:|",
            ]
        )

        for mode, count in fm_counts.most_common():
            pct = count / failures * 100
            lines.append(f"| {mode} | {count} | {pct:.1f}% |")

        lines.append("")

    # Difficulty distribution
    diff_counts: Counter[str] = Counter()
    diff_failure_counts: Counter[str] = Counter()
    for d in diagnoses:
        diff_counts[d.difficulty] += 1
        if not d.is_success:
            diff_failure_counts[d.difficulty] += 1

    lines.extend(
        [
            "## Difficulty Distribution",
            "",
            "| Difficulty | Total | Failures | % of Failures |",
            "|-----------|------:|---------:|--------------:|",
        ]
    )

    for diff in ["easy", "hard", "ambiguous"]:
        t = diff_counts.get(diff, 0)
        f = diff_failure_counts.get(diff, 0)
        fpct = f / failures * 100 if failures > 0 else 0.0
        lines.append(f"| {diff} | {t} | {f} | {fpct:.1f}% |")

    lines.append("")

    # Theoretical ceiling
    ambiguous_count = diff_counts.get("ambiguous", 0)
    ambiguous_frac = ambiguous_count / total if total > 0 else 0.0
    ceiling = 1.0 - ambiguous_frac

    lines.extend(
        [
            "## Theoretical Ceiling",
            "",
            f"- **Ambiguous queries**: {ambiguous_count} ({ambiguous_frac:.1%} of total)",
            f"- **Max achievable R@1** (assuming ambiguous queries are unsolvable): ~{ceiling:.1%}",
            "",
        ]
    )

    # Key findings
    if intent_mrrs:
        best = max(intent_mrrs, key=lambda k: intent_mrrs[k])
        worst = min(intent_mrrs, key=lambda k: intent_mrrs[k])

        lines.extend(
            [
                "## Key Findings",
                "",
                f"- **Strongest intent**: {best} (MRR={intent_mrrs[best]:.3f})",
                f"- **Weakest intent**: {worst} (MRR={intent_mrrs[worst]:.3f})",
            ]
        )

        if failures > 0 and fm_counts:
            top_fm = fm_counts.most_common(1)[0]
            lines.append(
                f"- **Most common failure mode**: {top_fm[0]} "
                f"({top_fm[1]} cases, {top_fm[1] / failures * 100:.1f}% of failures)"
            )

    lines.append("")
    return "\n".join(lines)
