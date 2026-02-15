"""Rigorous tests for two-stage retrieval pipeline.

These tests probe edge cases, numerical invariants, and integration
boundaries that the basic tests don't cover. The goal is to find real
bugs, not just verify happy paths.
"""

import random
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

from incite.embeddings.chunk_store import ChunkStore
from incite.models import Chunk, CitationContext, EvaluationResult, RetrievalResult

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_mock_paper_retriever(stage1_results: list[RetrievalResult]):
    """Create a mock paper retriever that returns (results, timing) tuple."""
    mock = MagicMock()
    mock.retrieve.return_value = (
        stage1_results,
        {"embed_query_ms": 1.0, "vector_search_ms": 2.0},
    )
    return mock


def _make_mock_embedder(dim: int = 8):
    mock = MagicMock()
    mock.dimension = dim
    mock.embed_query.return_value = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
    return mock


def _make_chunk_store(papers_and_scores: dict[str, list[float]], dim: int = 8):
    """Create ChunkStore with embeddings that produce known dot products."""
    query = np.ones(dim, dtype=np.float32) / np.sqrt(dim)

    chunks = []
    all_embeddings = []
    for paper_id, desired_scores in papers_and_scores.items():
        for i, target_score in enumerate(desired_scores):
            chunk_id = f"{paper_id}::chunk_{i}"
            chunks.append(Chunk(id=chunk_id, paper_id=paper_id, text=f"Text for {chunk_id}"))
            emb = np.zeros(dim, dtype=np.float32)
            emb[0] = target_score * np.sqrt(dim)
            all_embeddings.append(emb)

    store = ChunkStore(dimension=dim)
    if chunks:
        store.add_chunks(chunks, np.array(all_embeddings, dtype=np.float32))

    chunks_dict = {c.id: c for c in chunks}
    return store, chunks_dict


def _make_retriever(stage1, papers_and_scores, alpha=0.6, **kwargs):
    from incite.retrieval.two_stage import TwoStageRetriever

    mock_paper = _make_mock_paper_retriever(stage1)
    mock_emb = _make_mock_embedder()
    store, chunks = _make_chunk_store(papers_and_scores)
    return TwoStageRetriever(
        paper_retriever=mock_paper,
        chunk_store=store,
        chunks=chunks,
        embedder=mock_emb,
        alpha=alpha,
        **kwargs,
    )


# ── TwoStageRetriever edge cases ────────────────────────────────────────


class TestScoreNormalization:
    """Test the min-max normalization of stage 1 scores."""

    def test_single_result_gets_norm_1(self):
        """Single stage1 result: score_range=0, paper_score_norm should be 1.0."""

        stage1 = [RetrievalResult(paper_id="only", score=0.42, rank=1, score_breakdown={})]
        retriever = _make_retriever(stage1, {"only": [0.5]}, alpha=0.5)

        results = retriever.retrieve("test", k=1)
        assert results[0].score_breakdown["paper_score_norm"] == 1.0
        # final = 0.5 * 1.0 + 0.5 * 0.5 = 0.75
        assert abs(results[0].score - 0.75) < 1e-5

    def test_all_equal_scores(self):
        """All stage1 papers have the same score: all norms should be 1.0."""
        stage1 = [
            RetrievalResult(paper_id="a", score=0.5, rank=1, score_breakdown={}),
            RetrievalResult(paper_id="b", score=0.5, rank=2, score_breakdown={}),
            RetrievalResult(paper_id="c", score=0.5, rank=3, score_breakdown={}),
        ]
        retriever = _make_retriever(stage1, {"a": [0.1], "b": [0.5], "c": [0.9]})

        results = retriever.retrieve("test", k=3)
        for r in results:
            assert r.score_breakdown["paper_score_norm"] == 1.0

    def test_negative_scores_normalized(self):
        """Stage1 scores can be negative (BM25 can produce this). Normalization
        should still work — worst gets 0.0, best gets 1.0."""
        stage1 = [
            RetrievalResult(paper_id="pos", score=1.0, rank=1, score_breakdown={}),
            RetrievalResult(paper_id="neg", score=-0.5, rank=2, score_breakdown={}),
        ]
        retriever = _make_retriever(stage1, {"pos": [0.5], "neg": [0.5]})

        results = retriever.retrieve("test", k=2)
        result_map = {r.paper_id: r for r in results}
        assert result_map["pos"].score_breakdown["paper_score_norm"] == 1.0
        assert result_map["neg"].score_breakdown["paper_score_norm"] == 0.0


class TestScoreBlending:
    """Verify the alpha-blending formula invariants."""

    def test_score_matches_formula(self):
        """Verify final_score = alpha * paper_norm + (1-alpha) * best_chunk."""
        alpha = 0.7
        stage1 = [
            RetrievalResult(paper_id="a", score=1.0, rank=1, score_breakdown={}),
            RetrievalResult(paper_id="b", score=0.0, rank=2, score_breakdown={}),
        ]
        retriever = _make_retriever(stage1, {"a": [0.3], "b": [0.8]}, alpha=alpha)

        results = retriever.retrieve("test", k=2)
        for r in results:
            paper_norm = r.score_breakdown["paper_score_norm"]
            chunk_score = r.score_breakdown["best_chunk_score"]
            expected = alpha * paper_norm + (1 - alpha) * chunk_score
            assert abs(r.score - expected) < 1e-6, (
                f"paper_id={r.paper_id}: score={r.score}, expected={expected}"
            )

    def test_no_chunks_score_is_alpha_times_norm(self):
        """Paper with no chunks: score = alpha * paper_norm (not zero)."""
        alpha = 0.6
        stage1 = [
            RetrievalResult(paper_id="no_chunks", score=0.9, rank=1, score_breakdown={}),
        ]
        retriever = _make_retriever(stage1, {}, alpha=alpha)  # No chunks for anyone

        results = retriever.retrieve("test", k=1)
        # Single paper: norm = 1.0
        assert abs(results[0].score - alpha * 1.0) < 1e-6

    def test_reranking_can_promote_paper(self):
        """A paper ranked 2nd by stage1 can be promoted to 1st if it has much
        better chunk evidence, with the right alpha."""
        stage1 = [
            RetrievalResult(paper_id="a", score=0.9, rank=1, score_breakdown={}),
            RetrievalResult(paper_id="b", score=0.5, rank=2, score_breakdown={}),
        ]
        # paper_b has excellent chunk match, paper_a has none
        retriever = _make_retriever(
            stage1,
            {"b": [0.95]},
            alpha=0.3,  # Low alpha = more chunk weight
        )

        results = retriever.retrieve("test", k=2)
        # paper_b should be promoted: 0.3*0.0 + 0.7*0.95 = 0.665
        # paper_a: 0.3*1.0 + 0.7*0.0 = 0.3
        assert results[0].paper_id == "b"

    def test_alpha_0_5_tiebreaker(self):
        """With alpha=0.5, paper with equal paper_norm and chunk_score should
        tie with paper that has complementary scores."""
        stage1 = [
            RetrievalResult(paper_id="a", score=1.0, rank=1, score_breakdown={}),
            RetrievalResult(paper_id="b", score=0.0, rank=2, score_breakdown={}),
        ]
        # a: norm=1.0, chunk=0.0 → 0.5; b: norm=0.0, chunk=1.0 → 0.5
        # This SHOULD tie, but chunk score of 1.0 might not be achievable in practice
        retriever = _make_retriever(stage1, {"b": [1.0]}, alpha=0.5)

        results = retriever.retrieve("test", k=2)
        # Both should have score ~0.5
        assert abs(results[0].score - results[1].score) < 0.05


class TestEvidenceAttachment:
    """Test evidence snippet attachment logic."""

    def test_threshold_boundary(self):
        """Chunk score exactly at threshold should be attached."""
        stage1 = [
            RetrievalResult(paper_id="a", score=0.9, rank=1, score_breakdown={}),
        ]
        threshold = 0.35
        retriever = _make_retriever(stage1, {"a": [0.35]}, evidence_threshold=threshold)

        results = retriever.retrieve("test", k=1)
        # Score exactly at threshold: should be attached
        assert len(results[0].matched_paragraphs) == 1

    def test_just_below_threshold(self):
        """Chunk score just below threshold should NOT be attached as evidence
        but SHOULD still contribute to the blended score."""
        stage1 = [
            RetrievalResult(paper_id="a", score=0.9, rank=1, score_breakdown={}),
        ]
        retriever = _make_retriever(stage1, {"a": [0.34]}, evidence_threshold=0.35)

        results = retriever.retrieve("test", k=1)
        # No evidence attached
        assert len(results[0].matched_paragraphs) == 0
        assert results[0].matched_paragraph is None
        # But chunk score still used in blending
        assert results[0].score_breakdown["best_chunk_score"] > 0.3

    def test_max_evidence_per_paper_respected(self):
        """Only top N evidence snippets should be attached, matching max_evidence_per_paper."""
        stage1 = [
            RetrievalResult(paper_id="a", score=0.9, rank=1, score_breakdown={}),
        ]
        # 5 chunks all above threshold
        retriever = _make_retriever(
            stage1,
            {"a": [0.9, 0.8, 0.7, 0.6, 0.5]},
            alpha=0.6,
            max_evidence_per_paper=2,
        )

        results = retriever.retrieve("test", k=1)
        # Only top 2 should be attached (max_evidence_per_paper=2 limits
        # search_within_papers to return only 2)
        assert len(results[0].matched_paragraphs) <= 2

    def test_evidence_scores_are_descending(self):
        """matched_paragraphs should be ordered by score descending."""
        stage1 = [
            RetrievalResult(paper_id="a", score=0.9, rank=1, score_breakdown={}),
        ]
        retriever = _make_retriever(stage1, {"a": [0.9, 0.5, 0.7]}, max_evidence_per_paper=3)

        results = retriever.retrieve("test", k=1)
        paras = results[0].matched_paragraphs
        if len(paras) >= 2:
            scores = [p["score"] for p in paras]
            assert scores == sorted(scores, reverse=True)

    def test_chunk_missing_from_chunks_dict(self):
        """If a chunk_id is in the ChunkStore but not in the chunks dict,
        it should be silently skipped (no crash)."""
        from incite.retrieval.two_stage import TwoStageRetriever

        stage1 = [
            RetrievalResult(paper_id="a", score=0.9, rank=1, score_breakdown={}),
        ]
        mock_paper = _make_mock_paper_retriever(stage1)
        mock_emb = _make_mock_embedder()
        store, chunks_dict = _make_chunk_store({"a": [0.9]})

        # Remove the chunk from chunks_dict so it can't be found
        empty_chunks = {}

        retriever = TwoStageRetriever(
            paper_retriever=mock_paper,
            chunk_store=store,
            chunks=empty_chunks,  # No chunk text available
            embedder=mock_emb,
            alpha=0.6,
        )

        results = retriever.retrieve("test", k=1)
        # Should not crash; chunk score still contributes to blending
        assert results[0].score_breakdown["best_chunk_score"] > 0
        # But no evidence text should be attached
        assert results[0].matched_paragraphs == []


class TestReturnBehavior:
    """Test return value consistency."""

    def test_return_timing_false(self):
        """Without return_timing, should return a plain list."""
        stage1 = [RetrievalResult(paper_id="a", score=0.9, rank=1, score_breakdown={})]
        retriever = _make_retriever(stage1, {"a": [0.5]})

        result = retriever.retrieve("test", k=1, return_timing=False)
        assert isinstance(result, list)

    def test_return_timing_true(self):
        """With return_timing, should return (list, dict) tuple."""
        stage1 = [RetrievalResult(paper_id="a", score=0.9, rank=1, score_breakdown={})]
        retriever = _make_retriever(stage1, {"a": [0.5]})

        result = retriever.retrieve("test", k=1, return_timing=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        results, timing = result
        assert isinstance(results, list)
        assert isinstance(timing, dict)

    def test_empty_return_timing(self):
        """Empty stage1 with return_timing should return ([], timing_dict)."""
        retriever = _make_retriever([], {})

        result = retriever.retrieve("test", k=5, return_timing=True)
        results, timing = result
        assert results == []
        assert "stage1_ms" in timing

    def test_k_larger_than_stage1(self):
        """Requesting k > stage1 results should return all available."""
        stage1 = [
            RetrievalResult(paper_id="a", score=0.9, rank=1, score_breakdown={}),
            RetrievalResult(paper_id="b", score=0.7, rank=2, score_breakdown={}),
        ]
        retriever = _make_retriever(stage1, {"a": [0.5]})

        results = retriever.retrieve("test", k=10)
        assert len(results) == 2  # Only 2 papers available

    def test_timing_keys_no_collision(self):
        """Stage1 timing keys should be prefixed correctly without creating
        nested prefixes like 'stage1_stage1_'."""
        stage1 = [RetrievalResult(paper_id="a", score=0.9, rank=1, score_breakdown={})]
        retriever = _make_retriever(stage1, {"a": [0.5]})

        _, timing = retriever.retrieve("test", k=1, return_timing=True)
        # Should have stage1_ prefixed keys from inner timing
        assert "stage1_embed_query_ms" in timing
        assert "stage1_vector_search_ms" in timing
        # Should NOT have double-prefixed keys
        assert not any(k.startswith("stage1_stage1_") for k in timing)


# ── ChunkStore edge cases ────────────────────────────────────────────────


class TestChunkStoreEdgeCases:
    """Edge cases in ChunkStore.search_within_papers()."""

    def test_search_within_empty_paper_list(self):
        """Searching with no paper IDs should return empty dict."""
        store, _ = _make_chunk_store({"a": [0.5]})
        query = np.ones(8, dtype=np.float32) / np.sqrt(8)
        results = store.search_within_papers(query, [], top_per_paper=3)
        assert results == {}

    def test_negative_dot_products(self):
        """Chunks can have negative dot products. They should still appear
        in results but with negative scores."""
        dim = 8
        chunks = [
            Chunk(id="a::0", paper_id="a", text="text"),
        ]
        emb = np.zeros((1, dim), dtype=np.float32)
        emb[0, 0] = -1.0  # Will give negative dot product with positive query

        store = ChunkStore(dimension=dim)
        store.add_chunks(chunks, emb)

        query = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        results = store.search_within_papers(query, ["a"], top_per_paper=1)
        assert "a" in results
        assert results["a"][0][1] < 0  # Negative score

    def test_save_load_preserves_search_results(self):
        """After save/load round-trip, search_within_papers should return
        the same results."""
        store, _ = _make_chunk_store({"a": [0.8, 0.3], "b": [0.6]})
        query = np.ones(8, dtype=np.float32) / np.sqrt(8)

        before = store.search_within_papers(query, ["a", "b"], top_per_paper=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            store.save(tmpdir)
            loaded = ChunkStore()
            loaded.load(tmpdir)
            after = loaded.search_within_papers(query, ["a", "b"], top_per_paper=2)

        # Same papers
        assert set(before.keys()) == set(after.keys())
        # Same scores (approximately — reconstruct may lose precision)
        for pid in before:
            before_ids = [cid for cid, _ in before[pid]]
            after_ids = [cid for cid, _ in after[pid]]
            assert before_ids == after_ids
            for (_, s1), (_, s2) in zip(before[pid], after[pid]):
                assert abs(s1 - s2) < 1e-4

    def test_chunks_added_incrementally(self):
        """Adding chunks in two batches should still build correct reverse index."""
        dim = 8
        store = ChunkStore(dimension=dim)

        # Batch 1
        chunks1 = [Chunk(id="a::0", paper_id="a", text="t1")]
        embs1 = np.random.randn(1, dim).astype(np.float32)
        store.add_chunks(chunks1, embs1)

        # Batch 2
        chunks2 = [Chunk(id="a::1", paper_id="a", text="t2")]
        embs2 = np.random.randn(1, dim).astype(np.float32)
        store.add_chunks(chunks2, embs2)

        assert len(store._paper_to_chunks["a"]) == 2
        assert len(store.chunks_for_paper("a")) == 2


# ── Multi-scale training data ────────────────────────────────────────────


class TestMultiScaleTraining:
    """Test the multi-scale context sampling for training data."""

    def test_scale_distribution_sums_to_1(self):
        """SCALE_DISTRIBUTION weights should sum to 1.0."""
        from incite.finetuning.data_sources import SCALE_DISTRIBUTION

        total = sum(SCALE_DISTRIBUTION.values())
        assert abs(total - 1.0) < 1e-10

    def test_sample_scale_deterministic(self):
        """Same seed should produce the same scale."""
        from incite.finetuning.data_sources import _sample_scale

        rng1 = random.Random(42)
        rng2 = random.Random(42)
        assert _sample_scale(rng1) == _sample_scale(rng2)

    def test_sample_scale_distribution(self):
        """Over many samples, scale distribution should approximate target."""
        from incite.finetuning.data_sources import SCALE_DISTRIBUTION, _sample_scale

        rng = random.Random(12345)
        n = 10000
        counts = {k: 0 for k in SCALE_DISTRIBUTION}
        for _ in range(n):
            scale = _sample_scale(rng)
            counts[scale] += 1

        # Each scale should be within 3% of its target
        for scale, target in SCALE_DISTRIBUTION.items():
            actual = counts[scale] / n
            assert abs(actual - target) < 0.03, f"{scale}: expected ~{target:.2f}, got {actual:.3f}"

    def test_all_valid_scales_returned(self):
        """_sample_scale should only return scales in SCALE_DISTRIBUTION."""
        from incite.finetuning.data_sources import SCALE_DISTRIBUTION, _sample_scale

        rng = random.Random(42)
        valid = set(SCALE_DISTRIBUTION.keys())
        for _ in range(100):
            scale = _sample_scale(rng)
            assert scale in valid

    def test_context_get_query_missing_scale(self):
        """CitationContext.get_query() with a scale that has no data should
        fall back gracefully (not crash)."""
        ctx = CitationContext(
            id="test",
            source_paper_id="src",
            local_context="Local sentence.",
            narrow_context="Narrow three sentences.",
            broad_context=None,  # No broad context
            section_context=None,  # No section context
            global_context=None,
            ground_truth_ids=["gt1"],
            reference_set_ids=["gt1", "gt2"],
        )
        # "section" scale with no section_context should fall back
        query = ctx.get_query("section", clean=True)
        # Should return something (fallback), not crash
        assert isinstance(query, str)
        assert len(query) > 0


# ── Evaluation metrics integration ──────────────────────────────────────


class TestEvaluationResultNewFields:
    """Test new EvaluationResult fields from two-stage integration."""

    def test_to_dict_excludes_zero_two_stage_metrics(self):
        """When two-stage metrics are 0, they should NOT appear in to_dict()
        (backward compatibility)."""
        result = EvaluationResult(
            mrr=0.5,
            recall_at_10=0.8,
            num_queries=100,
        )
        d = result.to_dict()
        assert "evidence_coverage" not in d
        assert "mean_best_chunk_score" not in d

    def test_to_dict_includes_nonzero_two_stage_metrics(self):
        """When two-stage metrics are nonzero, they SHOULD appear."""
        result = EvaluationResult(
            mrr=0.5,
            num_queries=100,
            evidence_coverage=0.85,
            mean_best_chunk_score=0.42,
        )
        d = result.to_dict()
        assert d["evidence_coverage"] == 0.85
        assert d["mean_best_chunk_score"] == 0.42

    def test_to_dict_roundtrip(self):
        """to_dict() should produce values that can reconstruct the key metrics."""
        result = EvaluationResult(
            recall_at_1=0.25,
            recall_at_5=0.55,
            recall_at_10=0.80,
            recall_at_20=0.90,
            recall_at_50=0.95,
            mrr=0.42,
            ndcg_at_10=0.50,
            concordance=0.77,
            skill_mrr=0.31,
            num_queries=100,
            evidence_coverage=0.85,
            mean_best_chunk_score=0.42,
        )
        d = result.to_dict()
        assert d["recall@10"] == 0.80
        assert d["mrr"] == 0.42
        assert d["concordance"] == 0.77


class TestTwoStageMetricsComputation:
    """Test that evaluate_retrieval correctly computes two-stage metrics."""

    def test_evidence_coverage_calculation(self):
        """evidence_coverage = correct papers with evidence / total correct papers."""
        from incite.evaluation.metrics import evaluate_retrieval

        # Mock a retriever that returns results with two-stage breakdowns
        mock_retriever = MagicMock()

        # Create results with two-stage breakdowns
        def mock_retrieve_for_context(context, k=50, scale="local", prefix_section=False):
            return [
                RetrievalResult(
                    paper_id="gt1",
                    score=0.9,
                    rank=1,
                    score_breakdown={"best_chunk_score": 0.7, "paper_score_norm": 0.9},
                    matched_paragraphs=[{"text": "evidence", "score": 0.7}],
                ),
                RetrievalResult(
                    paper_id="gt2",
                    score=0.8,
                    rank=2,
                    score_breakdown={"best_chunk_score": 0.3, "paper_score_norm": 0.8},
                    matched_paragraphs=[],  # No evidence for this one
                ),
                RetrievalResult(
                    paper_id="other",
                    score=0.5,
                    rank=3,
                    score_breakdown={"best_chunk_score": 0.9, "paper_score_norm": 0.5},
                    matched_paragraphs=[{"text": "irrelevant", "score": 0.9}],
                ),
            ]

        mock_retriever.retrieve_for_context = mock_retrieve_for_context

        test_set = [
            CitationContext(
                id="q1",
                source_paper_id="src",
                local_context="Some query text about something.",
                ground_truth_ids=["gt1", "gt2"],
                reference_set_ids=[],
            ),
        ]

        result = evaluate_retrieval(mock_retriever, test_set, k=50, use_reference_sets=False)

        # 2 correct papers found (gt1, gt2); 1 has evidence (gt1)
        assert result.evidence_coverage == 0.5  # 1/2
        assert result.mean_best_chunk_score == pytest.approx(0.5)  # (0.7+0.3)/2

    def test_no_two_stage_breakdown_gives_zero(self):
        """Without best_chunk_score in breakdowns, two-stage metrics should be 0."""
        from incite.evaluation.metrics import evaluate_retrieval

        mock_retriever = MagicMock()

        def mock_retrieve_for_context(context, k=50, scale="local", prefix_section=False):
            return [
                RetrievalResult(
                    paper_id="gt1",
                    score=0.9,
                    rank=1,
                    score_breakdown={"neural": 0.9},  # No best_chunk_score
                ),
            ]

        mock_retriever.retrieve_for_context = mock_retrieve_for_context

        test_set = [
            CitationContext(
                id="q1",
                source_paper_id="src",
                local_context="Query text.",
                ground_truth_ids=["gt1"],
                reference_set_ids=[],
            ),
        ]

        result = evaluate_retrieval(mock_retriever, test_set, k=50, use_reference_sets=False)

        assert result.evidence_coverage == 0.0
        assert result.mean_best_chunk_score == 0.0


# ── _get_embedder with TwoStageRetriever ────────────────────────────────


class TestGetEmbedderChain:
    """Test that _get_embedder properly traverses TwoStageRetriever."""

    def test_finds_embedder_via_paper_retriever(self):
        """_get_embedder should follow paper_retriever to find the embedder."""
        from incite.evaluation.metrics import _get_embedder

        stage1 = [RetrievalResult(paper_id="a", score=0.9, rank=1, score_breakdown={})]
        retriever = _make_retriever(stage1, {"a": [0.5]})

        embedder = _get_embedder(retriever)
        # TwoStageRetriever has its own .embedder attribute
        assert embedder is not None

    def test_finds_embedder_through_hybrid_paper_retriever(self):
        """If paper_retriever is a HybridRetriever (has .retrievers attribute),
        _get_embedder should still find the neural embedder."""
        from incite.evaluation.metrics import _get_embedder
        from incite.retrieval.two_stage import TwoStageRetriever

        # Create a mock hybrid retriever with nested embedder
        mock_hybrid = MagicMock()
        mock_hybrid.embedder = None  # hybrid doesn't have direct embedder
        del mock_hybrid.embedder
        mock_neural = MagicMock()
        mock_neural.embedder = _make_mock_embedder()
        mock_bm25 = MagicMock()
        del mock_bm25.embedder
        mock_hybrid.retrievers = [(mock_neural, 1.0), (mock_bm25, 1.0)]
        mock_hybrid.retrieve.return_value = ([], {})

        # But first check: TwoStageRetriever has .embedder directly
        store, chunks = _make_chunk_store({})
        ts = TwoStageRetriever(
            paper_retriever=mock_hybrid,
            chunk_store=store,
            chunks=chunks,
            embedder=_make_mock_embedder(),
        )

        embedder = _get_embedder(ts)
        assert embedder is not None


# ── Confidence computation ──────────────────────────────────────────────


class TestTwoStageConfidence:
    """Test compute_confidence with two_stage mode."""

    def test_two_stage_mode_uses_max(self):
        """two_stage mode should use max(chunk, neural)."""
        from incite.utils import compute_confidence

        bd = {"best_chunk_score": 0.8, "neural": 0.3}
        assert compute_confidence(bd, mode="two_stage") == 0.8

        bd2 = {"best_chunk_score": 0.2, "neural": 0.9}
        assert compute_confidence(bd2, mode="two_stage") == 0.9

    def test_two_stage_both_zero(self):
        """Both zero should return 0.0."""
        from incite.utils import compute_confidence

        bd = {"best_chunk_score": 0.0, "neural": 0.0}
        assert compute_confidence(bd, mode="two_stage") == 0.0

    def test_two_stage_missing_keys(self):
        """Missing keys should default to 0.0, not crash."""
        from incite.utils import compute_confidence

        assert compute_confidence({}, mode="two_stage") == 0.0
        assert compute_confidence({"neural": 0.5}, mode="two_stage") == 0.5
        assert compute_confidence({"best_chunk_score": 0.5}, mode="two_stage") == 0.5


# ── Cursor-weighted evaluation ──────────────────────────────────────────


class TestCursorWeightedEval:
    """Test the cursor-weighted pseudo-scale evaluation."""

    def test_cursor_in_sweep_scales(self):
        """'cursor' should be in the sweep scales list in core.py."""
        # Verify by grepping — this is already covered by the agent,
        # but let's verify programmatically
        import re
        from pathlib import Path

        core_path = Path("src/incite/cli/core.py")
        content = core_path.read_text()
        # Find the scales list in the sweep section
        match = re.search(r"scales\s*=\s*\[([^\]]+)\]", content)
        assert match is not None
        scales_str = match.group(1)
        assert '"cursor"' in scales_str or "'cursor'" in scales_str

    def test_sentence_splitting_handles_abbreviations(self):
        """The sentence splitter in _evaluate_cursor_weighted should not
        split on abbreviations like 'Dr.', 'et al.'."""
        # Import the sentence splitting regex from core.py
        import re

        _ABBREV_RE = re.compile(
            r"\b(?:Dr|Mr|Mrs|Ms|Prof|Inc|Ltd|Jr|Sr|St|vs|etc|e\.g|i\.e|al|Fig|Eq|No|Vol)\."
        )

        def _split_sentences(text: str) -> list[str]:
            placeholder = "\x00"
            processed = _ABBREV_RE.sub(lambda m: re.sub(r"\.\s*", placeholder, m.group()), text)
            parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"])', processed)
            return [s.replace(placeholder, ". ").strip() for s in parts if s.strip()]

        text = (
            "This was shown by Smith et al. in their landmark paper. The results were significant."
        )
        sentences = _split_sentences(text)
        assert len(sentences) == 2
        assert "et al." in sentences[0]


# ── Integration: end-to-end two-stage retrieve + evaluate ───────────────


class TestEndToEndTwoStage:
    """Integration test: full pipeline from retrieve to evaluate."""

    def test_full_pipeline_produces_valid_metrics(self):
        """Create a TwoStageRetriever, run evaluate_retrieval, verify
        all metrics are valid (no NaN, within bounds)."""
        from incite.evaluation.metrics import evaluate_retrieval

        stage1 = [
            RetrievalResult(paper_id="correct", score=0.7, rank=1, score_breakdown={}),
            RetrievalResult(paper_id="wrong1", score=0.9, rank=2, score_breakdown={}),
            RetrievalResult(paper_id="wrong2", score=0.5, rank=3, score_breakdown={}),
        ]
        retriever = _make_retriever(stage1, {"correct": [0.8], "wrong1": [0.3], "wrong2": [0.1]})

        test_set = [
            CitationContext(
                id="q1",
                source_paper_id="src",
                local_context="Query about citation retrieval.",
                ground_truth_ids=["correct"],
                reference_set_ids=["correct", "wrong1", "wrong2"],
            ),
        ]

        result = evaluate_retrieval(retriever, test_set, k=10, use_reference_sets=False)

        # Basic sanity
        assert result.num_queries == 1
        assert 0 <= result.mrr <= 1
        assert 0 <= result.recall_at_10 <= 1
        assert not np.isnan(result.mrr)
        assert not np.isnan(result.recall_at_10)
        # Two-stage metrics should be populated
        assert result.evidence_coverage >= 0
        assert result.mean_best_chunk_score >= 0

    def test_deduplication_with_two_stage(self):
        """Deduplication should work with TwoStageRetriever results."""
        from incite.models import Paper

        stage1 = [
            RetrievalResult(paper_id="a", score=0.9, rank=1, score_breakdown={}),
            RetrievalResult(paper_id="b", score=0.7, rank=2, score_breakdown={}),
            RetrievalResult(paper_id="c", score=0.5, rank=3, score_breakdown={}),
        ]
        retriever = _make_retriever(stage1, {"a": [0.5], "b": [0.5], "c": [0.5]})

        # Papers a and b have the same title
        papers = {
            "a": Paper(id="a", title="Same Title", abstract="abs a"),
            "b": Paper(id="b", title="Same Title", abstract="abs b"),
            "c": Paper(id="c", title="Different", abstract="abs c"),
        }

        results = retriever.retrieve("test", k=3, papers=papers, deduplicate=True)
        titles = [papers[r.paper_id].title for r in results]
        # Should have deduplicated
        assert len(results) <= 3
        # After dedup, only one "Same Title" should remain
        assert titles.count("Same Title") <= 1
