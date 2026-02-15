"""Integration tests for evaluate_retrieval."""

import json
import pytest
from pathlib import Path

from incite.evaluation.metrics import evaluate_retrieval
from incite.evaluation.experiment_log import ExperimentLogger
from incite.interfaces import Retriever
from incite.models import CitationContext, EvaluationResult, QueryResult, RetrievalResult


class MockRetriever(Retriever):
    """Mock retriever that returns predetermined results."""

    def __init__(self, results_by_query: dict[str, list[RetrievalResult]] = None):
        self._results = results_by_query or {}
        self._default_results = []

    def set_default_results(self, results: list[RetrievalResult]):
        self._default_results = results

    def retrieve(self, query: str, k: int = 10, **kwargs) -> list[RetrievalResult]:
        if query in self._results:
            return self._results[query][:k]
        return self._default_results[:k]


class TestEvaluateRetrieval:
    @pytest.fixture
    def test_set(self):
        return [
            CitationContext(
                id="ctx1",
                local_context="Sea level rise is accelerating.",
                source_paper_id="src1",
                ground_truth_ids=["p1"],
                reference_set_ids=["p1", "p2", "p3"],
            ),
            CitationContext(
                id="ctx2",
                local_context="Deep learning advances in NLP.",
                source_paper_id="src2",
                ground_truth_ids=["p4"],
                reference_set_ids=["p4", "p5", "p6"],
            ),
        ]

    def test_perfect_retrieval(self, test_set):
        retriever = MockRetriever()
        # p1 is ground truth for ctx1 (rank 1), p4 for ctx2 (rank 2)
        retriever.set_default_results([
            RetrievalResult(paper_id="p1", score=0.9, rank=1),
            RetrievalResult(paper_id="p4", score=0.8, rank=2),
            RetrievalResult(paper_id="p2", score=0.7, rank=3),
        ])

        result = evaluate_retrieval(
            retriever, test_set, k=10, use_reference_sets=False
        )

        assert isinstance(result, EvaluationResult)
        assert result.num_queries == 2
        # ctx1: p1 at rank 1 → R@1=1, MRR=1; ctx2: p4 at rank 2 → R@1=0, MRR=0.5
        assert result.recall_at_1 == pytest.approx(0.5)
        assert result.mrr == pytest.approx(0.75)
        assert result.recall_at_5 == 1.0

    def test_zero_retrieval(self, test_set):
        retriever = MockRetriever()
        # Return non-matching papers
        retriever.set_default_results([
            RetrievalResult(paper_id="x1", score=0.9, rank=1),
            RetrievalResult(paper_id="x2", score=0.8, rank=2),
        ])

        result = evaluate_retrieval(
            retriever, test_set, k=10, use_reference_sets=False
        )

        assert result.num_queries == 2
        assert result.recall_at_1 == 0.0
        assert result.mrr == 0.0

    def test_reference_set_filtering(self, test_set):
        retriever = MockRetriever()
        # Return p1 (ground truth for ctx1) and some non-reference papers
        retriever.set_default_results([
            RetrievalResult(paper_id="outside_ref", score=0.95, rank=1),
            RetrievalResult(paper_id="p1", score=0.9, rank=2),
            RetrievalResult(paper_id="p4", score=0.85, rank=3),
        ])

        # With reference set filtering, "outside_ref" should be removed
        result = evaluate_retrieval(
            retriever, test_set, k=10, use_reference_sets=True
        )

        assert result.num_queries == 2
        # p1 should be found for ctx1 (it's in reference set)
        # p4 should be found for ctx2 (it's in reference set)
        assert result.recall_at_1 > 0  # At least one should hit

    def test_skips_empty_ground_truth(self):
        test_set = [
            CitationContext(
                id="ctx1",
                local_context="Some text.",
                ground_truth_ids=[],  # No ground truth
            ),
            CitationContext(
                id="ctx2",
                local_context="Other text.",
                ground_truth_ids=["p1"],
            ),
        ]

        retriever = MockRetriever()
        retriever.set_default_results([
            RetrievalResult(paper_id="p1", score=0.9, rank=1),
        ])

        result = evaluate_retrieval(retriever, test_set, k=10, use_reference_sets=False)
        assert result.num_queries == 1  # Only ctx2 counted

    def test_empty_test_set(self):
        retriever = MockRetriever()
        result = evaluate_retrieval(retriever, [], k=10)
        assert result.num_queries == 0

    def test_scale_parameter(self):
        test_set = [
            CitationContext(
                id="ctx1",
                local_context="Local text.",
                narrow_context="Narrow context text.",
                ground_truth_ids=["p1"],
            ),
        ]

        retriever = MockRetriever()
        retriever.set_default_results([
            RetrievalResult(paper_id="p1", score=0.9, rank=1),
        ])

        # Should work with different scales
        for scale in ["local", "narrow", "broad"]:
            result = evaluate_retrieval(
                retriever, test_set, k=10, scale=scale, use_reference_sets=False
            )
            assert result.num_queries == 1

    def test_metric_aggregation(self):
        test_set = [
            CitationContext(
                id="ctx1",
                local_context="Text 1.",
                ground_truth_ids=["p1"],
            ),
            CitationContext(
                id="ctx2",
                local_context="Text 2.",
                ground_truth_ids=["p2"],
            ),
        ]

        retriever = MockRetriever()
        # p1 at rank 1, p2 at rank 2 for all queries
        retriever.set_default_results([
            RetrievalResult(paper_id="p1", score=0.9, rank=1),
            RetrievalResult(paper_id="p2", score=0.8, rank=2),
        ])

        result = evaluate_retrieval(
            retriever, test_set, k=10, use_reference_sets=False
        )

        # ctx1: p1 at rank 1 → R@1=1, MRR=1
        # ctx2: p2 at rank 2 → R@1=0, MRR=0.5
        assert result.num_queries == 2
        assert result.recall_at_1 == pytest.approx(0.5)
        assert result.mrr == pytest.approx(0.75)
        assert result.recall_at_5 == pytest.approx(1.0)


class TestPerQueryResults:
    """Tests for per-query result collection in evaluate_retrieval."""

    def test_per_query_populated(self):
        test_set = [
            CitationContext(
                id="ctx1",
                local_context="Text 1.",
                source_paper_id="src1",
                ground_truth_ids=["p1"],
            ),
            CitationContext(
                id="ctx2",
                local_context="Text 2.",
                source_paper_id="src2",
                ground_truth_ids=["p2"],
            ),
        ]

        retriever = MockRetriever()
        retriever.set_default_results([
            RetrievalResult(paper_id="p1", score=0.9, rank=1),
            RetrievalResult(paper_id="p2", score=0.8, rank=2),
        ])

        result = evaluate_retrieval(
            retriever, test_set, k=10, use_reference_sets=False
        )

        assert len(result.per_query) == 2
        assert result.per_query[0].context_id == "ctx1"
        assert result.per_query[0].source_paper_id == "src1"
        assert result.per_query[0].first_relevant_rank == 1
        assert result.per_query[1].context_id == "ctx2"
        assert result.per_query[1].first_relevant_rank == 2

    def test_per_query_not_in_to_dict(self):
        test_set = [
            CitationContext(
                id="ctx1",
                local_context="Text.",
                ground_truth_ids=["p1"],
            ),
        ]
        retriever = MockRetriever()
        retriever.set_default_results([
            RetrievalResult(paper_id="p1", score=0.9),
        ])
        result = evaluate_retrieval(
            retriever, test_set, k=10, use_reference_sets=False
        )
        d = result.to_dict()
        assert "per_query" not in d

    def test_per_query_first_rank_none(self):
        test_set = [
            CitationContext(
                id="ctx1",
                local_context="Text.",
                ground_truth_ids=["p1"],
            ),
        ]
        retriever = MockRetriever()
        retriever.set_default_results([
            RetrievalResult(paper_id="x", score=0.9),
        ])
        result = evaluate_retrieval(
            retriever, test_set, k=10, use_reference_sets=False
        )
        assert result.per_query[0].first_relevant_rank is None

    def test_per_query_scores_match_aggregates(self):
        test_set = [
            CitationContext(
                id=f"ctx{i}",
                local_context=f"Text {i}.",
                ground_truth_ids=["p1"],
            )
            for i in range(10)
        ]
        retriever = MockRetriever()
        retriever.set_default_results([
            RetrievalResult(paper_id="p1", score=0.9),
        ])
        result = evaluate_retrieval(
            retriever, test_set, k=10, use_reference_sets=False
        )
        # Manually average per-query scores and compare
        avg_r10 = sum(qr.scores["recall@10"] for qr in result.per_query) / len(result.per_query)
        assert avg_r10 == pytest.approx(result.recall_at_10)


class TestMacroAverage:
    """Tests for macro-averaging across source papers."""

    def test_macro_average_corrects_skew(self):
        # Source A has 3 queries (all hit), source B has 1 query (miss)
        # Micro: 3/4 = 0.75, Macro: (1.0 + 0.0) / 2 = 0.5
        test_set = [
            CitationContext(id="a1", local_context="T.", source_paper_id="A", ground_truth_ids=["p1"]),
            CitationContext(id="a2", local_context="T.", source_paper_id="A", ground_truth_ids=["p1"]),
            CitationContext(id="a3", local_context="T.", source_paper_id="A", ground_truth_ids=["p1"]),
            CitationContext(id="b1", local_context="T.", source_paper_id="B", ground_truth_ids=["p2"]),
        ]
        retriever = MockRetriever()
        retriever.set_default_results([
            RetrievalResult(paper_id="p1", score=0.9),
        ])

        micro = evaluate_retrieval(
            retriever, test_set, k=10, use_reference_sets=False, macro_average=False
        )
        macro = evaluate_retrieval(
            retriever, test_set, k=10, use_reference_sets=False, macro_average=True
        )

        # Micro: (1+1+1+0)/4 = 0.75
        assert micro.recall_at_1 == pytest.approx(0.75)
        # Macro: avg(A=1.0, B=0.0) = 0.5
        assert macro.recall_at_1 == pytest.approx(0.5)

    def test_macro_same_as_micro_uniform(self):
        # When each source has same number of queries, macro == micro
        test_set = [
            CitationContext(id="a1", local_context="T.", source_paper_id="A", ground_truth_ids=["p1"]),
            CitationContext(id="b1", local_context="T.", source_paper_id="B", ground_truth_ids=["p1"]),
        ]
        retriever = MockRetriever()
        retriever.set_default_results([
            RetrievalResult(paper_id="p1", score=0.9),
        ])

        micro = evaluate_retrieval(
            retriever, test_set, k=10, use_reference_sets=False, macro_average=False
        )
        macro = evaluate_retrieval(
            retriever, test_set, k=10, use_reference_sets=False, macro_average=True
        )

        assert micro.recall_at_1 == pytest.approx(macro.recall_at_1)


class TestEvaluationResultStr:
    """Tests for EvaluationResult.__str__ with CIs."""

    def test_str_without_per_query(self):
        r = EvaluationResult(recall_at_10=0.8, num_queries=100)
        s = str(r)
        assert "0.800" in s
        assert "[" not in s  # No CIs

    def test_str_with_per_query(self):
        qrs = [
            QueryResult(
                context_id=f"c{i}", source_paper_id="s",
                ground_truth_ids=["p"], first_relevant_rank=1,
                scores={"recall@1": 1.0, "recall@5": 1.0, "recall@10": 1.0,
                        "recall@20": 1.0, "recall@50": 1.0, "mrr": 1.0, "ndcg@10": 1.0},
            )
            for i in range(50)
        ]
        r = EvaluationResult(
            recall_at_10=1.0, num_queries=50, per_query=qrs,
        )
        s = str(r)
        assert "[" in s  # CIs present


class TestPerQuerySaveLoad:
    """Tests for save/load per-query data in ExperimentLogger."""

    def test_save_and_load(self, tmp_path):
        logger = ExperimentLogger(log_path=str(tmp_path / "experiments.jsonl"))

        qrs = [
            QueryResult(
                context_id="ctx1", source_paper_id="s1",
                ground_truth_ids=["p1"], first_relevant_rank=1,
                scores={"recall@10": 1.0},
            ),
            QueryResult(
                context_id="ctx2", source_paper_id="s2",
                ground_truth_ids=["p2"], first_relevant_rank=None,
                scores={"recall@10": 0.0},
            ),
        ]

        path = logger.save_per_query("abc123", qrs)
        assert path.exists()

        loaded = logger.load_per_query("abc123")
        assert len(loaded) == 2
        assert loaded[0].context_id == "ctx1"
        assert loaded[0].first_relevant_rank == 1
        assert loaded[1].first_relevant_rank is None

    def test_load_prefix_match(self, tmp_path):
        logger = ExperimentLogger(log_path=str(tmp_path / "experiments.jsonl"))
        qrs = [QueryResult(
            context_id="c1", source_paper_id="s", ground_truth_ids=["p"],
            scores={"recall@10": 0.5},
        )]
        logger.save_per_query("abc12345", qrs)

        loaded = logger.load_per_query("abc1")
        assert len(loaded) == 1

    def test_load_missing(self, tmp_path):
        logger = ExperimentLogger(log_path=str(tmp_path / "experiments.jsonl"))
        loaded = logger.load_per_query("nonexistent")
        assert loaded == []


class TestDiffRuns:
    """Tests for diff_runs in ExperimentLogger."""

    def test_diff_output_format(self, tmp_path):
        logger = ExperimentLogger(log_path=str(tmp_path / "experiments.jsonl"))

        qrs_a = [
            QueryResult(context_id="c1", source_paper_id="s", ground_truth_ids=["p"],
                        scores={"recall@10": 0.0}, first_relevant_rank=None),
            QueryResult(context_id="c2", source_paper_id="s", ground_truth_ids=["p"],
                        scores={"recall@10": 1.0}, first_relevant_rank=1),
        ]
        qrs_b = [
            QueryResult(context_id="c1", source_paper_id="s", ground_truth_ids=["p"],
                        scores={"recall@10": 1.0}, first_relevant_rank=1),
            QueryResult(context_id="c2", source_paper_id="s", ground_truth_ids=["p"],
                        scores={"recall@10": 1.0}, first_relevant_rank=1),
        ]

        logger.save_per_query("run_a", qrs_a)
        logger.save_per_query("run_b", qrs_b)

        diff = logger.diff_runs("run_a", "run_b", metric="recall@10")
        assert "Improved:" in diff
        assert "Regressed:" in diff
        assert "Unchanged:" in diff
        assert "Delta:" in diff
        assert "p=" in diff

    def test_diff_missing_run(self, tmp_path):
        logger = ExperimentLogger(log_path=str(tmp_path / "experiments.jsonl"))
        diff = logger.diff_runs("missing_a", "missing_b")
        assert "No per-query data" in diff
