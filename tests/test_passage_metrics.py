"""Tests for passage-level evaluation metrics."""

import json
import tempfile
from pathlib import Path

import pytest

from incite.evaluation.passage_metrics import (
    PassageEvaluationResult,
    PassageTestExample,
    evaluate_passage_retrieval,
    load_passage_test_set,
    passage_token_f1,
    save_passage_test_set,
)
from incite.models import Chunk


def _make_chunk(paper_id: str, idx: int, text: str, section: str = "") -> Chunk:
    return Chunk(
        id=f"{paper_id}::chunk_{idx}",
        paper_id=paper_id,
        text=text,
        section=section,
    )


class TestPassageTokenF1:
    def test_exact_match(self):
        """Identical passages should have F1 = 1.0."""
        text = "The gradient descent method converges quickly."
        assert passage_token_f1(text, text) == 1.0

    def test_partial_overlap(self):
        """Partially overlapping passages should have 0 < F1 < 1."""
        pred = "The gradient descent method converges quickly"
        gold = "The gradient descent method is well known"

        f1 = passage_token_f1(pred, gold)
        assert 0.0 < f1 < 1.0

        # Verify the calculation
        pred_tokens = set(pred.lower().split())
        gold_tokens = set(gold.lower().split())
        overlap = pred_tokens & gold_tokens
        precision = len(overlap) / len(pred_tokens)
        recall = len(overlap) / len(gold_tokens)
        expected = 2 * precision * recall / (precision + recall)
        assert abs(f1 - expected) < 1e-6

    def test_disjoint(self):
        """Non-overlapping passages should have F1 = 0.0."""
        pred = "cats dogs animals"
        gold = "quantum physics entropy"
        assert passage_token_f1(pred, gold) == 0.0

    def test_empty_strings(self):
        """Empty strings should have F1 = 0.0."""
        assert passage_token_f1("", "some text") == 0.0
        assert passage_token_f1("some text", "") == 0.0
        assert passage_token_f1("", "") == 0.0

    def test_case_insensitive(self):
        """F1 should be case-insensitive."""
        assert passage_token_f1("Hello World", "hello world") == 1.0


class TestPassageEvaluation:
    def test_intent_stratification(self):
        """Metrics should be broken out by intent correctly."""
        test_set = [
            PassageTestExample(
                id="1",
                citation_context="we use gradient descent for optimization",
                cited_paper_id="p1",
                gold_passage="we propose gradient descent optimization for training neural networks efficiently",
                gold_passage_section="Methods",
                intent="method",
            ),
            PassageTestExample(
                id="2",
                citation_context="background knowledge about the topic area",
                cited_paper_id="p1",
                gold_passage="this paper provides background knowledge and context for understanding the topic area",
                gold_passage_section="Introduction",
                intent="background",
            ),
        ]

        chunks_by_paper = {
            "p1": [
                _make_chunk("p1", 0, "we propose gradient descent optimization for training neural networks efficiently", "Methods"),
                _make_chunk("p1", 1, "this paper provides background knowledge and context for understanding the topic area", "Introduction"),
                _make_chunk("p1", 2, "results show convergence across many different benchmark datasets and evaluation metrics", "Results"),
            ],
        }

        # Use embedder=None (F1-only mode) for unit tests
        result = evaluate_passage_retrieval(None, test_set, chunks_by_paper)

        assert result.num_queries == 2
        assert "method" in result.by_intent
        assert "background" in result.by_intent
        assert result.by_intent["method"]["num_queries"] == 1
        assert result.by_intent["background"]["num_queries"] == 1

    def test_empty_test_set(self):
        """Empty test set should return zero metrics."""
        result = evaluate_passage_retrieval(None, [], {})

        assert result.num_queries == 0
        assert result.paper_recall_at_10 == 0.0
        assert result.passage_recall_at_1 == 0.0
        assert result.passage_f1 == 0.0

    def test_perfect_retrieval(self):
        """When gold passage is the only chunk, should get perfect scores."""
        gold_text = "This is the exact gold passage text about machine learning methods."
        test_set = [
            PassageTestExample(
                id="1",
                citation_context="machine learning methods are important",
                cited_paper_id="p1",
                gold_passage=gold_text,
                gold_passage_section="Methods",
                intent="method",
            ),
        ]

        chunks_by_paper = {
            "p1": [_make_chunk("p1", 0, gold_text, "Methods")],
        }

        result = evaluate_passage_retrieval(None, test_set, chunks_by_paper)

        assert result.num_queries == 1
        assert result.passage_recall_at_1 == 1.0
        assert result.passage_f1 == 1.0


class TestPassageTestSetIO:
    def test_save_and_load_roundtrip(self):
        """Save and load should preserve all fields."""
        examples = [
            PassageTestExample(
                id="test_1",
                citation_context="context text",
                cited_paper_id="paper_1",
                gold_passage="passage text",
                gold_passage_section="Methods",
                intent="method",
                source_paper_id="source_1",
                reference_set_ids=["paper_1", "paper_2"],
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "test_set.jsonl"
            save_passage_test_set(examples, path)
            loaded = load_passage_test_set(path)

        assert len(loaded) == 1
        ex = loaded[0]
        assert ex.id == "test_1"
        assert ex.citation_context == "context text"
        assert ex.cited_paper_id == "paper_1"
        assert ex.gold_passage == "passage text"
        assert ex.gold_passage_section == "Methods"
        assert ex.intent == "method"
        assert ex.source_paper_id == "source_1"
        assert ex.reference_set_ids == ["paper_1", "paper_2"]


class TestPassageEvaluationResult:
    def test_str_formatting(self):
        """String representation should be readable."""
        result = PassageEvaluationResult(
            paper_recall_at_10=0.9,
            passage_recall_at_1=0.5,
            passage_recall_at_5=0.7,
            passage_recall_at_10=0.8,
            passage_mrr=0.6,
            passage_f1=0.65,
            num_queries=100,
        )
        s = str(result)
        assert "n=100" in s
        assert "0.500" in s

    def test_to_dict(self):
        """to_dict should return all fields."""
        result = PassageEvaluationResult(
            paper_recall_at_10=0.9,
            passage_recall_at_1=0.5,
            num_queries=10,
        )
        d = result.to_dict()
        assert d["paper_recall_at_10"] == 0.9
        assert d["passage_recall_at_1"] == 0.5
        assert d["num_queries"] == 10
