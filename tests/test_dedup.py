"""Tests for deduplication and confidence scoring utilities."""

import math

import pytest

from incite.models import Paper, RetrievalResult
from incite.utils import _normalize_title, compute_confidence, deduplicate_results


class TestNormalizeTitle:
    def test_basic_lowercase(self):
        assert _normalize_title("Hello World") == "hello world"

    def test_strip_leading_the(self):
        assert _normalize_title("The Quick Brown Fox") == "quick brown fox"

    def test_strip_leading_a(self):
        assert _normalize_title("A New Approach") == "new approach"

    def test_strip_leading_an(self):
        assert _normalize_title("An Overview of Methods") == "overview of methods"

    def test_remove_punctuation(self):
        assert _normalize_title("Hello, World!") == "hello world"

    def test_collapse_whitespace(self):
        assert _normalize_title("Hello   World") == "hello world"

    def test_strip_trailing_punctuation(self):
        assert _normalize_title("A title.") == "title"

    def test_accented_characters(self):
        assert _normalize_title("Résumé of François") == "resume of francois"

    def test_empty_string(self):
        assert _normalize_title("") == ""

    def test_identical_after_normalization(self):
        t1 = "The Effect of Ocean Acidification on Coral Reefs"
        t2 = "Effect of Ocean Acidification on Coral Reefs"
        assert _normalize_title(t1) == _normalize_title(t2)

    def test_case_insensitive_match(self):
        t1 = "OCEAN ACIDIFICATION"
        t2 = "ocean acidification"
        assert _normalize_title(t1) == _normalize_title(t2)

    def test_punctuation_differences(self):
        t1 = "Climate Change: A Review"
        t2 = "Climate Change - A Review"
        assert _normalize_title(t1) == _normalize_title(t2)


class TestDeduplicateResults:
    def _make_paper(self, pid, title, abstract=""):
        return Paper(id=pid, title=title, abstract=abstract)

    def _make_result(self, pid, score, rank):
        return RetrievalResult(paper_id=pid, score=score, rank=rank)

    def test_no_duplicates(self):
        papers = {
            "p1": self._make_paper("p1", "Paper One"),
            "p2": self._make_paper("p2", "Paper Two"),
        }
        results = [
            self._make_result("p1", 1.0, 1),
            self._make_result("p2", 0.8, 2),
        ]
        deduped = deduplicate_results(results, papers)
        assert len(deduped) == 2

    def test_removes_duplicate_title(self):
        papers = {
            "p1": self._make_paper("p1", "The Same Title"),
            "p2": self._make_paper("p2", "Same Title"),  # Matches after normalization
        }
        results = [
            self._make_result("p1", 1.0, 1),
            self._make_result("p2", 0.8, 2),
        ]
        deduped = deduplicate_results(results, papers)
        assert len(deduped) == 1
        assert deduped[0].paper_id == "p1"  # Higher score kept

    def test_prefers_abstract_when_scores_close(self):
        papers = {
            "p1": self._make_paper("p1", "Same Title", abstract=""),
            "p2": self._make_paper("p2", "Same Title", abstract="This has an abstract"),
        }
        results = [
            self._make_result("p1", 1.0, 1),
            self._make_result("p2", 0.98, 2),  # Within 5% of p1
        ]
        deduped = deduplicate_results(results, papers)
        assert len(deduped) == 1
        assert deduped[0].paper_id == "p2"  # Has abstract

    def test_keeps_higher_score_when_scores_not_close(self):
        papers = {
            "p1": self._make_paper("p1", "Same Title", abstract=""),
            "p2": self._make_paper("p2", "Same Title", abstract="Has abstract"),
        }
        results = [
            self._make_result("p1", 1.0, 1),
            self._make_result("p2", 0.5, 2),  # Not within 5%
        ]
        deduped = deduplicate_results(results, papers)
        assert len(deduped) == 1
        assert deduped[0].paper_id == "p1"  # Higher score wins

    def test_preserves_order(self):
        papers = {
            "p1": self._make_paper("p1", "First Paper"),
            "p2": self._make_paper("p2", "Second Paper"),
            "p3": self._make_paper("p3", "Third Paper"),
        }
        results = [
            self._make_result("p1", 1.0, 1),
            self._make_result("p2", 0.8, 2),
            self._make_result("p3", 0.6, 3),
        ]
        deduped = deduplicate_results(results, papers)
        assert [r.paper_id for r in deduped] == ["p1", "p2", "p3"]

    def test_handles_missing_paper(self):
        papers = {"p1": self._make_paper("p1", "Known Paper")}
        results = [
            self._make_result("p1", 1.0, 1),
            self._make_result("p_unknown", 0.8, 2),
        ]
        deduped = deduplicate_results(results, papers)
        assert len(deduped) == 2  # Unknown paper kept as-is

    def test_empty_results(self):
        assert deduplicate_results([], {}) == []

    def test_multiple_duplicates(self):
        papers = {
            "p1": self._make_paper("p1", "Same Title"),
            "p2": self._make_paper("p2", "The Same Title"),
            "p3": self._make_paper("p3", "SAME TITLE!"),
            "p4": self._make_paper("p4", "Different Paper"),
        }
        results = [
            self._make_result("p1", 1.0, 1),
            self._make_result("p4", 0.9, 2),
            self._make_result("p2", 0.8, 3),
            self._make_result("p3", 0.7, 4),
        ]
        deduped = deduplicate_results(results, papers)
        assert len(deduped) == 2
        titles = {papers[r.paper_id].title for r in deduped}
        assert "Different Paper" in titles


class TestComputeConfidence:
    def test_hybrid_uses_neural(self):
        breakdown = {"neural": 0.65, "bm25": 15.0}
        conf = compute_confidence(breakdown, mode="hybrid")
        assert conf == 0.65

    def test_paragraph_uses_chunk_score(self):
        breakdown = {"best_chunk_score": 0.72, "num_chunks_matched": 3}
        conf = compute_confidence(breakdown, mode="paragraph")
        assert conf == 0.72

    def test_bm25_only_sigmoid(self):
        breakdown = {"bm25": 10.0}
        conf = compute_confidence(breakdown, mode="bm25")
        assert conf == pytest.approx(0.5, abs=0.01)  # sigmoid(0) = 0.5

    def test_bm25_high_score(self):
        breakdown = {"bm25": 25.0}
        conf = compute_confidence(breakdown, mode="bm25")
        assert conf > 0.9

    def test_bm25_low_score(self):
        breakdown = {"bm25": 0.0}
        conf = compute_confidence(breakdown, mode="bm25")
        assert conf < 0.2

    def test_empty_breakdown(self):
        assert compute_confidence({}, mode="hybrid") == 0.0

    def test_paragraph_hybrid_uses_neural_score(self):
        breakdown = {"neural_score": 0.58, "bm25_score": 12.0}
        conf = compute_confidence(breakdown, mode="hybrid")
        assert conf == 0.58

    def test_confidence_in_range(self):
        for neural in [0.0, 0.3, 0.5, 0.7, 1.0]:
            conf = compute_confidence({"neural": neural}, mode="hybrid")
            assert 0.0 <= conf <= 1.0

        for bm25 in [0.0, 5.0, 10.0, 20.0, 50.0]:
            conf = compute_confidence({"bm25": bm25}, mode="bm25")
            assert 0.0 <= conf <= 1.0
