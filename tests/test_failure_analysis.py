"""Tests for failure analysis pipeline."""

import json

import pytest

from incite.evaluation.failure_analysis import (
    DiagnosisInput,
    DiagnosisResult,
    build_diagnosis_prompt,
    generate_intent_report,
    parse_diagnosis_response,
)
from incite.evaluation.intent_taxonomy import (
    CITATION_INTENTS,
    VALID_FAILURE_MODES,
    VALID_INTENTS,
)
from incite.models import QueryResult

# --- Taxonomy tests ---


def test_intent_taxonomy_has_required_intents():
    assert "background" in VALID_INTENTS
    assert "method" in VALID_INTENTS
    assert "contrast" in VALID_INTENTS
    assert len(CITATION_INTENTS) >= 5


def test_failure_modes_has_required_modes():
    assert "semantic_mismatch" in VALID_FAILURE_MODES
    assert "ambiguous_context" in VALID_FAILURE_MODES
    assert "" in VALID_FAILURE_MODES  # empty string valid for successes


# --- Prompt construction tests ---


def test_build_failure_prompt_includes_predictions():
    inp = DiagnosisInput(
        context_id="test_1",
        query_text="Smith et al. showed that X causes Y",
        source_paper_id="paper_1",
        gt_paper_ids=["gt_1"],
        gt_paper_title="The Effect of X on Y",
        gt_paper_abstract="We study X and find it causes Y in all conditions.",
        is_success=False,
        first_relevant_rank=5,
        scores={"recall@1": 0.0, "mrr": 0.2},
    )
    preds = [
        {
            "paper_id": "p1",
            "title": "Wrong Paper 1",
            "abstract": "About Z not Y",
            "score": 0.9,
            "rank": 1,
        },
        {
            "paper_id": "p2",
            "title": "Wrong Paper 2",
            "abstract": "About W not Y",
            "score": 0.8,
            "rank": 2,
        },
    ]
    prompt = build_diagnosis_prompt(inp, preds)
    assert "Smith et al." in prompt
    assert "The Effect of X on Y" in prompt
    assert "Wrong Paper 1" in prompt
    assert "intent" in prompt
    assert "failure_mode" in prompt


def test_build_success_prompt_asks_for_intent():
    inp = DiagnosisInput(
        context_id="test_2",
        query_text="Following the method of Jones (2020)",
        source_paper_id="paper_2",
        gt_paper_ids=["gt_2"],
        gt_paper_title="A Novel Method",
        gt_paper_abstract="We present a method for doing things.",
        is_success=True,
        first_relevant_rank=1,
        scores={"recall@1": 1.0, "mrr": 1.0},
    )
    prompt = build_diagnosis_prompt(inp, None)
    assert "Jones" in prompt or "method" in prompt.lower()
    # Success prompt should ask for intent classification
    assert "intent" in prompt


# --- Response parsing tests ---


def test_parse_valid_failure_response():
    response = json.dumps(
        {
            "intent": "method",
            "failure_mode": "lexical_gap",
            "difficulty": "hard",
            "reasoning": "The query uses different terminology than the paper.",
        }
    )
    result = parse_diagnosis_response("ctx_1", response, is_success=False, predictions=[])
    assert result.intent == "method"
    assert result.failure_mode == "lexical_gap"
    assert result.difficulty == "hard"
    assert result.is_success is False


def test_parse_valid_success_response():
    response = json.dumps(
        {
            "intent": "background",
            "difficulty": "easy",
            "reasoning": "Standard background citation.",
        }
    )
    result = parse_diagnosis_response("ctx_2", response, is_success=True, predictions=None)
    assert result.intent == "background"
    assert result.failure_mode == ""
    assert result.is_success is True


def test_parse_malformed_response_returns_unclassified():
    result = parse_diagnosis_response(
        "ctx_3", "not valid json!!!", is_success=False, predictions=[]
    )
    assert result.intent == "unclassified"
    assert result.context_id == "ctx_3"


def test_parse_response_with_markdown_fences():
    response = '```json\n{"intent": "support", "difficulty": "easy", "reasoning": "test"}\n```'
    result = parse_diagnosis_response("ctx_4", response, is_success=True, predictions=None)
    assert result.intent == "support"


def test_parse_response_invalid_intent_becomes_unclassified():
    response = json.dumps(
        {
            "intent": "not_a_real_intent",
            "difficulty": "easy",
            "reasoning": "test",
        }
    )
    result = parse_diagnosis_response("ctx_5", response, is_success=True, predictions=None)
    assert result.intent == "unclassified"


def test_parse_response_invalid_failure_mode_becomes_unclassified():
    response = json.dumps(
        {
            "intent": "method",
            "failure_mode": "not_a_real_mode",
            "difficulty": "hard",
            "reasoning": "test",
        }
    )
    result = parse_diagnosis_response("ctx_6", response, is_success=False, predictions=[])
    assert result.failure_mode == "unclassified"


# --- Serialization round-trip tests ---


def test_diagnosis_result_roundtrip():
    original = DiagnosisResult(
        context_id="ctx_1",
        intent="method",
        failure_mode="lexical_gap",
        difficulty="hard",
        reasoning="Test reasoning",
        top_predictions=[{"paper_id": "p1", "title": "T1", "score": 0.9, "rank": 1}],
        is_success=False,
    )
    d = original.to_dict()
    restored = DiagnosisResult.from_dict(d)
    assert restored.context_id == original.context_id
    assert restored.intent == original.intent
    assert restored.failure_mode == original.failure_mode
    assert restored.top_predictions == original.top_predictions


def test_diagnosis_result_from_dict_defaults():
    """from_dict should handle missing optional fields gracefully."""
    d = {"context_id": "c1", "intent": "background"}
    result = DiagnosisResult.from_dict(d)
    assert result.failure_mode == ""
    assert result.reasoning == ""
    assert result.top_predictions == []
    assert result.is_success is False


# --- Report generation tests ---


def test_generate_intent_report_produces_markdown():
    diagnoses = [
        DiagnosisResult("c1", "background", "", "easy", "test", [], True),
        DiagnosisResult("c2", "method", "lexical_gap", "hard", "test", [], False),
        DiagnosisResult("c3", "background", "semantic_mismatch", "hard", "test", [], False),
    ]
    query_results = [
        QueryResult("c1", "sp1", ["gt1"], {"recall@1": 1.0, "mrr": 1.0, "recall@10": 1.0}, 1),
        QueryResult("c2", "sp1", ["gt2"], {"recall@1": 0.0, "mrr": 0.2, "recall@10": 1.0}, 5),
        QueryResult("c3", "sp2", ["gt3"], {"recall@1": 0.0, "mrr": 0.0, "recall@10": 0.0}, None),
    ]
    report = generate_intent_report(diagnoses, query_results)
    assert "# Citation Recommendation Failure Analysis Report" in report
    assert "background" in report
    assert "method" in report
    assert "Failure Mode" in report


def test_generate_intent_report_empty_diagnoses():
    report = generate_intent_report([], [])
    assert "No diagnoses to report" in report


def test_generate_intent_report_all_successes():
    diagnoses = [
        DiagnosisResult("c1", "background", "", "easy", "test", [], True),
        DiagnosisResult("c2", "method", "", "easy", "test", [], True),
    ]
    query_results = [
        QueryResult("c1", "sp1", ["gt1"], {"recall@1": 1.0, "mrr": 1.0, "recall@10": 1.0}, 1),
        QueryResult("c2", "sp1", ["gt2"], {"recall@1": 1.0, "mrr": 1.0, "recall@10": 1.0}, 1),
    ]
    report = generate_intent_report(diagnoses, query_results)
    assert "100.0%" in report
    # No failure mode section when all succeed
    assert "Failure Mode Distribution" not in report


def test_generate_intent_report_key_findings():
    diagnoses = [
        DiagnosisResult("c1", "background", "", "easy", "test", [], True),
        DiagnosisResult("c2", "method", "lexical_gap", "hard", "test", [], False),
    ]
    query_results = [
        QueryResult("c1", "sp1", ["gt1"], {"recall@1": 1.0, "mrr": 1.0, "recall@10": 1.0}, 1),
        QueryResult("c2", "sp1", ["gt2"], {"recall@1": 0.0, "mrr": 0.2, "recall@10": 1.0}, 5),
    ]
    report = generate_intent_report(diagnoses, query_results)
    assert "Strongest intent" in report
    assert "Weakest intent" in report


# --- Intent-stratified metrics tests ---


def test_evaluate_retrieval_by_intent():
    from incite.evaluation.metrics import evaluate_retrieval_by_intent

    diagnoses = [
        DiagnosisResult("c1", "background", "", "easy", "test", [], True),
        DiagnosisResult("c2", "background", "semantic_mismatch", "hard", "test", [], False),
        DiagnosisResult("c3", "method", "", "easy", "test", [], True),
    ]
    query_results = [
        QueryResult(
            "c1",
            "sp1",
            ["gt1"],
            {
                "recall@1": 1.0,
                "mrr": 1.0,
                "recall@5": 1.0,
                "recall@10": 1.0,
                "recall@20": 1.0,
                "recall@50": 1.0,
                "ndcg@10": 1.0,
            },
            1,
        ),
        QueryResult(
            "c2",
            "sp1",
            ["gt2"],
            {
                "recall@1": 0.0,
                "mrr": 0.0,
                "recall@5": 0.0,
                "recall@10": 0.0,
                "recall@20": 0.0,
                "recall@50": 0.0,
                "ndcg@10": 0.0,
            },
            None,
        ),
        QueryResult(
            "c3",
            "sp2",
            ["gt3"],
            {
                "recall@1": 1.0,
                "mrr": 1.0,
                "recall@5": 1.0,
                "recall@10": 1.0,
                "recall@20": 1.0,
                "recall@50": 1.0,
                "ndcg@10": 1.0,
            },
            1,
        ),
    ]
    by_intent = evaluate_retrieval_by_intent(query_results, diagnoses)
    assert "background" in by_intent
    assert "method" in by_intent
    assert "all" in by_intent
    # background has 1 success + 1 failure -> MRR = 0.5
    assert by_intent["background"].mrr == pytest.approx(0.5)
    assert by_intent["background"].num_queries == 2
    # method has 1 success -> MRR = 1.0
    assert by_intent["method"].mrr == pytest.approx(1.0)
    assert by_intent["method"].num_queries == 1
    # all has 3 queries
    assert by_intent["all"].num_queries == 3


def test_evaluate_retrieval_by_intent_unknown_queries():
    """Queries without a matching diagnosis go to 'unknown'."""
    from incite.evaluation.metrics import evaluate_retrieval_by_intent

    diagnoses = [
        DiagnosisResult("c1", "background", "", "easy", "test", [], True),
    ]
    query_results = [
        QueryResult(
            "c1",
            "sp1",
            ["gt1"],
            {
                "recall@1": 1.0,
                "mrr": 1.0,
                "recall@5": 1.0,
                "recall@10": 1.0,
                "recall@20": 1.0,
                "recall@50": 1.0,
                "ndcg@10": 1.0,
            },
            1,
        ),
        QueryResult(
            "c_no_diag",
            "sp2",
            ["gt2"],
            {
                "recall@1": 0.0,
                "mrr": 0.0,
                "recall@5": 0.0,
                "recall@10": 0.0,
                "recall@20": 0.0,
                "recall@50": 0.0,
                "ndcg@10": 0.0,
            },
            None,
        ),
    ]
    by_intent = evaluate_retrieval_by_intent(query_results, diagnoses)
    assert "unknown" in by_intent
    assert by_intent["unknown"].num_queries == 1
    assert by_intent["unknown"].mrr == pytest.approx(0.0)


def test_evaluate_retrieval_by_intent_empty():
    from incite.evaluation.metrics import evaluate_retrieval_by_intent

    by_intent = evaluate_retrieval_by_intent([], [])
    # Only "all" with empty list
    assert by_intent == {}
