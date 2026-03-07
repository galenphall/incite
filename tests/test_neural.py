"""Tests for NeuralRetriever."""

import pytest

from incite.retrieval.neural import NeuralRetriever


class TestNeuralRetriever:
    @pytest.fixture(scope="class")
    def neural_retriever(self, mock_embedder, sample_papers):
        return NeuralRetriever.from_papers(
            sample_papers, mock_embedder, show_progress=False
        )

    def test_retrieve_returns_results(self, neural_retriever):
        results = neural_retriever.retrieve("sea level rise", k=5)
        assert len(results) > 0

    def test_scores_descending(self, neural_retriever):
        results = neural_retriever.retrieve("climate change ocean", k=6)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_sequential(self, neural_retriever):
        results = neural_retriever.retrieve("deep learning NLP", k=6)
        for i, r in enumerate(results):
            assert r.rank == i + 1

    def test_k_limits_output(self, neural_retriever):
        results = neural_retriever.retrieve("climate", k=3)
        assert len(results) <= 3

    def test_k_larger_than_corpus(self, neural_retriever):
        results = neural_retriever.retrieve("test", k=100)
        assert len(results) == 6

    def test_score_breakdown_has_neural(self, neural_retriever):
        results = neural_retriever.retrieve("ocean temperature", k=3)
        for r in results:
            assert "neural" in r.score_breakdown

    def test_return_timing(self, neural_retriever):
        results, timing = neural_retriever.retrieve(
            "sea level", k=3, return_timing=True
        )
        assert isinstance(results, list)
        assert isinstance(timing, dict)
        assert "embed_query_ms" in timing
        assert "vector_search_ms" in timing

    def test_precomputed_embedding(self, neural_retriever, mock_embedder):
        query = "coastal flooding"
        qvec = mock_embedder.embed_query(query)
        results_normal = neural_retriever.retrieve(query, k=6)
        results_precomp = neural_retriever.retrieve(
            query, k=6, query_embedding=qvec
        )
        ids_normal = [r.paper_id for r in results_normal]
        ids_precomp = [r.paper_id for r in results_precomp]
        assert ids_normal == ids_precomp

    def test_from_papers_builds_index(self, neural_retriever, sample_papers):
        assert neural_retriever.store.size == len(sample_papers)

    def test_empty_query(self, neural_retriever):
        results = neural_retriever.retrieve("", k=3)
        assert isinstance(results, list)
