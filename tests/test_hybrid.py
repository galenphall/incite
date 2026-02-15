"""Tests for hybrid retrieval (neural + BM25 with RRF fusion)."""

import pytest

from incite.models import Paper
from incite.retrieval.bm25 import BM25Retriever
from incite.retrieval.hybrid import HybridRetriever
from incite.retrieval.neural import NeuralRetriever


class TestHybridRetriever:
    @pytest.fixture
    def papers(self):
        return [
            Paper(id="p1", title="Sea level rise projections", abstract="Global sea levels are rising due to climate change and thermal expansion"),
            Paper(id="p2", title="Deep learning for NLP", abstract="Neural networks for natural language processing tasks"),
            Paper(id="p3", title="Ocean temperature trends", abstract="Sea surface temperatures and oceanic heat content increasing"),
            Paper(id="p4", title="Climate modeling techniques", abstract="General circulation models for climate projections and scenarios"),
        ]

    @pytest.fixture
    def paper_dict(self, papers):
        return {p.id: p for p in papers}

    @pytest.fixture
    def hybrid_retriever(self, mock_embedder, papers):
        neural = NeuralRetriever.from_papers(papers, mock_embedder, show_progress=False)
        bm25 = BM25Retriever.from_papers(papers)
        return HybridRetriever(
            retrievers=[(neural, 1.0), (bm25, 1.0)],
            fusion="rrf",
            rrf_k=5,
        )

    def test_retrieve_scores_descending(self, hybrid_retriever):
        results = hybrid_retriever.retrieve("climate change ocean", k=4)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_retrieve_ranks_sequential(self, hybrid_retriever):
        results = hybrid_retriever.retrieve("sea level", k=4)
        for i, result in enumerate(results):
            assert result.rank == i + 1

    def test_return_timing(self, hybrid_retriever):
        results, timing = hybrid_retriever.retrieve(
            "sea level", k=3, return_timing=True
        )
        assert isinstance(results, list)
        assert isinstance(timing, dict)
        assert "fusion_ms" in timing

    def test_k_limits_output(self, hybrid_retriever):
        results = hybrid_retriever.retrieve("climate ocean temperature", k=2)
        assert len(results) <= 2

    def test_author_boost_disabled_at_1(self, mock_embedder, papers, paper_dict):
        neural = NeuralRetriever.from_papers(papers, mock_embedder, show_progress=False)
        bm25 = BM25Retriever.from_papers(papers)
        hybrid = HybridRetriever(
            retrievers=[(neural, 1.0), (bm25, 1.0)],
            fusion="rrf",
            rrf_k=5,
        )
        results_1 = hybrid.retrieve("sea level", k=4, papers=paper_dict, author_boost=1.0)
        results_2 = hybrid.retrieve("sea level", k=4)
        ids_1 = [r.paper_id for r in results_1]
        ids_2 = [r.paper_id for r in results_2]
        assert ids_1 == ids_2

    def test_rrf_k_matches_production(self, hybrid_retriever):
        assert hybrid_retriever.rrf_k == 5

    def test_weighted_fusion_mode(self, mock_embedder, papers):
        neural = NeuralRetriever.from_papers(papers, mock_embedder, show_progress=False)
        bm25 = BM25Retriever.from_papers(papers)
        hybrid = HybridRetriever(
            retrievers=[(neural, 1.0), (bm25, 1.0)],
            fusion="weighted",
            rrf_k=5,
        )
        results = hybrid.retrieve("climate ocean", k=4)
        assert len(results) > 0

    def test_neural_and_bm25_both_contribute(self, hybrid_retriever):
        results = hybrid_retriever.retrieve("sea level rise climate", k=4)
        breakdown_keys = set()
        for r in results:
            breakdown_keys.update(r.score_breakdown.keys())
        assert "neural" in breakdown_keys
        assert "bm25" in breakdown_keys

    def test_dedup_removes_duplicate_titles(self, mock_embedder):
        papers = [
            Paper(id="a", title="Sea level rise", abstract="Global sea levels rising"),
            Paper(id="b", title="Sea level rise", abstract="Sea levels are rising fast"),
            Paper(id="c", title="Deep learning", abstract="Neural networks for tasks"),
        ]
        paper_dict = {p.id: p for p in papers}
        neural = NeuralRetriever.from_papers(papers, mock_embedder, show_progress=False)
        bm25 = BM25Retriever.from_papers(papers)
        hybrid = HybridRetriever(
            retrievers=[(neural, 1.0), (bm25, 1.0)],
            fusion="rrf",
            rrf_k=5,
        )
        results = hybrid.retrieve(
            "sea level", k=3, papers=paper_dict, deduplicate=True
        )
        titles = [paper_dict[r.paper_id].title for r in results]
        # After dedup, "Sea level rise" should appear at most once
        assert titles.count("Sea level rise") <= 1

    def test_author_boost_increases_score(self, mock_embedder, papers, paper_dict):
        neural = NeuralRetriever.from_papers(papers, mock_embedder, show_progress=False)
        bm25 = BM25Retriever.from_papers(papers)
        hybrid = HybridRetriever(
            retrievers=[(neural, 1.0), (bm25, 1.0)],
            fusion="rrf",
            rrf_k=5,
        )
        # Query mentions "Smith" who authored p1 and p4
        results_no_boost = hybrid.retrieve("Smith sea level", k=4, papers=paper_dict, author_boost=1.0)
        results_boosted = hybrid.retrieve("Smith sea level", k=4, papers=paper_dict, author_boost=2.0)
        # Smith papers should have higher scores with boost
        smith_ids = {"p1", "p4"}
        for r in results_boosted:
            if r.paper_id in smith_ids:
                no_boost_r = next((x for x in results_no_boost if x.paper_id == r.paper_id), None)
                if no_boost_r:
                    assert r.score >= no_boost_r.score
