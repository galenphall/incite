"""Tests for evaluation metrics."""

import pytest

from incite.evaluation.metrics import (
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    evaluate_single,
    bootstrap_ci,
    paired_bootstrap_test,
    clean_test_set,
    _compute_first_relevant_rank,
)
from incite.models import CitationContext, Paper, QueryResult, RetrievalResult


class TestRecallAtK:
    def test_perfect_recall(self):
        retrieved = ["a", "b", "c"]
        relevant = ["a", "b"]
        assert recall_at_k(retrieved, relevant, 3) == 1.0

    def test_zero_recall(self):
        retrieved = ["a", "b", "c"]
        relevant = ["x", "y"]
        assert recall_at_k(retrieved, relevant, 3) == 0.0

    def test_partial_recall(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = ["a", "x"]
        assert recall_at_k(retrieved, relevant, 4) == 0.5

    def test_recall_at_k_cutoff(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = ["c", "d"]
        assert recall_at_k(retrieved, relevant, 2) == 0.0
        assert recall_at_k(retrieved, relevant, 4) == 1.0

    def test_empty_relevant(self):
        assert recall_at_k(["a", "b"], [], 2) == 0.0


class TestMRR:
    def test_first_position(self):
        retrieved = ["a", "b", "c"]
        relevant = ["a"]
        assert mean_reciprocal_rank(retrieved, relevant) == 1.0

    def test_second_position(self):
        retrieved = ["a", "b", "c"]
        relevant = ["b"]
        assert mean_reciprocal_rank(retrieved, relevant) == 0.5

    def test_third_position(self):
        retrieved = ["a", "b", "c"]
        relevant = ["c"]
        assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(1/3)

    def test_not_found(self):
        retrieved = ["a", "b", "c"]
        relevant = ["x"]
        assert mean_reciprocal_rank(retrieved, relevant) == 0.0

    def test_multiple_relevant(self):
        # MRR uses first relevant hit
        retrieved = ["a", "b", "c"]
        relevant = ["b", "c"]
        assert mean_reciprocal_rank(retrieved, relevant) == 0.5


class TestNDCG:
    def test_perfect_ranking(self):
        retrieved = ["a", "b"]
        relevant = ["a", "b"]
        assert ndcg_at_k(retrieved, relevant, 2) == 1.0

    def test_zero_ndcg(self):
        retrieved = ["x", "y"]
        relevant = ["a", "b"]
        assert ndcg_at_k(retrieved, relevant, 2) == 0.0

    def test_partial_ndcg(self):
        # Second position only
        retrieved = ["x", "a"]
        relevant = ["a"]
        # DCG = 1/log2(3) ≈ 0.63
        # IDCG = 1/log2(2) = 1
        assert ndcg_at_k(retrieved, relevant, 2) == pytest.approx(1/1.585, rel=0.01)

    def test_empty_relevant(self):
        assert ndcg_at_k(["a", "b"], [], 2) == 0.0


class TestEvaluateSingle:
    def test_evaluate_single(self):
        results = [
            RetrievalResult(paper_id="a", score=0.9, rank=1),
            RetrievalResult(paper_id="b", score=0.8, rank=2),
            RetrievalResult(paper_id="c", score=0.7, rank=3),
        ]
        ground_truth = ["b"]

        scores = evaluate_single(results, ground_truth)

        assert scores["recall@1"] == 0.0
        assert scores["recall@5"] == 1.0
        assert scores["mrr"] == 0.5


class TestQueryResult:
    def test_to_dict_roundtrip(self):
        qr = QueryResult(
            context_id="ctx1",
            source_paper_id="src1",
            ground_truth_ids=["p1", "p2"],
            scores={"recall@10": 1.0, "mrr": 0.5},
            first_relevant_rank=2,
        )
        d = qr.to_dict()
        qr2 = QueryResult.from_dict(d)
        assert qr2.context_id == "ctx1"
        assert qr2.source_paper_id == "src1"
        assert qr2.ground_truth_ids == ["p1", "p2"]
        assert qr2.scores == {"recall@10": 1.0, "mrr": 0.5}
        assert qr2.first_relevant_rank == 2

    def test_from_dict_missing_optional(self):
        d = {
            "context_id": "ctx1",
            "ground_truth_ids": ["p1"],
            "scores": {"recall@10": 0.0},
        }
        qr = QueryResult.from_dict(d)
        assert qr.source_paper_id is None
        assert qr.first_relevant_rank is None


class TestFirstRelevantRank:
    def test_found_at_rank_1(self):
        results = [
            RetrievalResult(paper_id="p1", score=0.9),
            RetrievalResult(paper_id="p2", score=0.8),
        ]
        assert _compute_first_relevant_rank(results, ["p1"]) == 1

    def test_found_at_rank_3(self):
        results = [
            RetrievalResult(paper_id="x", score=0.9),
            RetrievalResult(paper_id="y", score=0.8),
            RetrievalResult(paper_id="p1", score=0.7),
        ]
        assert _compute_first_relevant_rank(results, ["p1"]) == 3

    def test_not_found(self):
        results = [
            RetrievalResult(paper_id="x", score=0.9),
        ]
        assert _compute_first_relevant_rank(results, ["p1"]) is None

    def test_multiple_relevant(self):
        results = [
            RetrievalResult(paper_id="x", score=0.9),
            RetrievalResult(paper_id="p2", score=0.8),
            RetrievalResult(paper_id="p1", score=0.7),
        ]
        assert _compute_first_relevant_rank(results, ["p1", "p2"]) == 2


class TestBootstrapCI:
    def test_perfect_scores(self):
        scores = [1.0] * 100
        lo, hi = bootstrap_ci(scores)
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(1.0)

    def test_zero_scores(self):
        scores = [0.0] * 100
        lo, hi = bootstrap_ci(scores)
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(0.0)

    def test_mixed_scores_contains_mean(self):
        scores = [0.0, 1.0] * 50
        lo, hi = bootstrap_ci(scores)
        assert lo < 0.5
        assert hi > 0.5

    def test_ci_bounds_ordered(self):
        scores = [0.2, 0.4, 0.6, 0.8, 1.0] * 20
        lo, hi = bootstrap_ci(scores)
        assert lo <= hi

    def test_empty_scores(self):
        lo, hi = bootstrap_ci([])
        assert lo == 0.0
        assert hi == 0.0

    def test_deterministic(self):
        scores = [0.1, 0.5, 0.9, 0.3, 0.7] * 10
        lo1, hi1 = bootstrap_ci(scores, seed=123)
        lo2, hi2 = bootstrap_ci(scores, seed=123)
        assert lo1 == lo2
        assert hi1 == hi2


class TestPairedBootstrapTest:
    def test_identical_scores(self):
        scores = [0.5] * 100
        delta, p, d = paired_bootstrap_test(scores, scores)
        assert delta == pytest.approx(0.0)
        # p-value should be high (not significant)
        assert p > 0.05

    def test_clearly_different(self):
        a = [0.0] * 100
        b = [1.0] * 100
        delta, p, d = paired_bootstrap_test(a, b)
        assert delta == pytest.approx(1.0)
        assert p < 0.01

    def test_delta_sign(self):
        a = [0.3] * 50
        b = [0.7] * 50
        delta, p, d = paired_bootstrap_test(a, b)
        assert delta > 0  # B is better

        delta2, p2, d2 = paired_bootstrap_test(b, a)
        assert delta2 < 0  # A is worse (reversed)

    def test_mismatched_length(self):
        with pytest.raises(ValueError):
            paired_bootstrap_test([0.1, 0.2], [0.3])

    def test_empty_scores(self):
        delta, p, d = paired_bootstrap_test([], [])
        assert delta == 0.0
        assert p == 1.0

    def test_deterministic(self):
        a = [0.1, 0.5, 0.9] * 20
        b = [0.2, 0.6, 0.8] * 20
        r1 = paired_bootstrap_test(a, b, seed=42)
        r2 = paired_bootstrap_test(a, b, seed=42)
        assert r1 == r2


def _make_paper(pid, title="Good Paper", abstract="A real abstract about something meaningful."):
    return Paper(id=pid, title=title, abstract=abstract)


def _make_context(cid, source_paper_id, gt_ids, ref_ids=None, local_context="Some citation context."):
    return CitationContext(
        id=cid,
        local_context=local_context,
        source_paper_id=source_paper_id,
        ground_truth_ids=gt_ids,
        reference_set_ids=ref_ids or gt_ids,
    )


class TestCleanTestSet:
    def test_removes_degenerate_gt(self):
        """Queries whose GT papers have stub abstracts should be removed."""
        papers = {
            "good": _make_paper("good"),
            "stub": _make_paper("stub", abstract="n/a"),
            "empty": _make_paper("empty", abstract=""),
        }
        test_set = [
            _make_context("q1", "src1", ["good"], ["good", "stub"]),
            _make_context("q2", "src1", ["stub"], ["good", "stub"]),  # all GT degenerate
            _make_context("q3", "src1", ["empty"], ["good", "empty"]),  # all GT degenerate
            _make_context("q4", "src1", ["good", "stub"], ["good", "stub"]),  # partial — kept
        ]
        cleaned, stats = clean_test_set(test_set, papers)
        assert len(cleaned) == 2  # q1 and q4 kept
        assert stats.removed_degenerate_gt == 2
        assert "stub" in stats.degenerate_paper_ids
        assert "empty" in stats.degenerate_paper_ids

    def test_removes_duplicate_queries(self):
        """Exact duplicate (local_context + ground_truth_ids) should be removed."""
        papers = {"p1": _make_paper("p1")}
        test_set = [
            _make_context("q1", "src1", ["p1"], local_context="Same context."),
            _make_context("q2", "src1", ["p1"], local_context="Same context."),  # dup
            _make_context("q3", "src1", ["p1"], local_context="Different context."),
        ]
        cleaned, stats = clean_test_set(test_set, papers)
        assert len(cleaned) == 2
        assert stats.removed_duplicate_queries == 1

    def test_removes_domain_mismatch(self):
        """Source papers whose GT papers are isolated from all other ref sets."""
        papers = {
            "shared_ref1": _make_paper("shared_ref1"),
            "shared_ref2": _make_paper("shared_ref2"),
            "isolated_ref": _make_paper("isolated_ref"),
        }
        test_set = [
            # Source A references shared_ref1 — normal
            _make_context("q1", "srcA", ["shared_ref1"], ["shared_ref1", "shared_ref2"],
                          local_context="Context about topic A."),
            _make_context("q2", "srcA", ["shared_ref2"], ["shared_ref1", "shared_ref2"],
                          local_context="Context about topic A continued."),
            # Source B also references shared_ref1 — overlaps with A
            _make_context("q3", "srcB", ["shared_ref1"], ["shared_ref1"],
                          local_context="Context about topic B."),
            # Source C references isolated_ref — no overlap with A or B, small ref set
            _make_context("q4", "srcC", ["isolated_ref"], ["isolated_ref"],
                          local_context="Context about something totally different."),
        ]
        cleaned, stats = clean_test_set(test_set, papers)
        assert len(cleaned) == 3  # q4 removed (srcC isolated)
        assert stats.removed_domain_mismatch == 1
        assert "srcC" in stats.mismatch_source_ids

    def test_no_false_positives_on_large_ref_set(self):
        """Source papers with large ref sets shouldn't be flagged even if GT is unique."""
        papers = {f"p{i}": _make_paper(f"p{i}") for i in range(10)}
        ref_ids = [f"p{i}" for i in range(10)]
        test_set = [
            _make_context("q1", "srcA", ["p0"], ref_ids),
            _make_context("q2", "srcB", ["p5"], ref_ids),
        ]
        cleaned, stats = clean_test_set(test_set, papers)
        assert len(cleaned) == 2  # nothing removed
        assert stats.removed_domain_mismatch == 0

    def test_no_cleaning_flag(self):
        """Disabling both filters should return the same test set."""
        papers = {"stub": _make_paper("stub", abstract="")}
        test_set = [
            _make_context("q1", "src1", ["stub"]),
            _make_context("q2", "src1", ["stub"], local_context="Same."),
            _make_context("q3", "src1", ["stub"], local_context="Same."),
        ]
        cleaned, stats = clean_test_set(
            test_set, papers, remove_degenerate=False, remove_duplicates=False
        )
        assert len(cleaned) == 3
        assert stats.total_removed == 0

    def test_stats_str(self):
        """CleaningStats string representation should be informative."""
        papers = {
            "good": _make_paper("good"),
            "bad": _make_paper("bad", abstract="x"),
        }
        test_set = [
            _make_context("q1", "src1", ["good"]),
            _make_context("q2", "src1", ["bad"]),
        ]
        _, stats = clean_test_set(test_set, papers)
        s = str(stats)
        assert "Degenerate GT" in s
        assert "1 queries" in s

    def test_empty_test_set(self):
        cleaned, stats = clean_test_set([], {})
        assert cleaned == []
        assert stats.total_removed == 0

    def test_all_queries_valid(self):
        """No queries removed when all GT papers are healthy."""
        papers = {
            "p1": _make_paper("p1"),
            "p2": _make_paper("p2"),
        }
        test_set = [
            _make_context("q1", "src1", ["p1"], ["p1", "p2"]),
            _make_context("q2", "src1", ["p2"], ["p1", "p2"]),
        ]
        cleaned, stats = clean_test_set(test_set, papers)
        assert len(cleaned) == 2
        assert stats.total_removed == 0
