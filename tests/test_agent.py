"""Tests for agent-friendly testing interface."""

import json
import pytest

from incite.agent import (
    AgentRecommendation,
    AgentResponse,
    InCiteAgent,
    TimingInfo,
)
from incite.models import Paper


class TestAgentResponse:
    def test_serialization(self):
        """Test full serialization chain: AgentResponse -> dict -> JSON and back."""
        timing = TimingInfo(total_ms=50.0, embed_query_ms=30.0, vector_search_ms=15.0)
        rec = AgentRecommendation(
            paper_id="abc123",
            rank=1,
            score=0.95,
            title="Test Paper",
        )
        response = AgentResponse(
            query="test query",
            recommendations=[rec],
            timing=timing,
            corpus_size=1000,
            method="hybrid",
            embedder="minilm",
            timestamp="2024-01-01T00:00:00Z",
        )

        # to_dict: nested objects serialize correctly
        d = response.to_dict()
        assert d["query"] == "test query"
        assert d["corpus_size"] == 1000
        assert len(d["recommendations"]) == 1
        assert d["timing"]["total_ms"] == 50.0
        # None values excluded from nested dicts
        assert "bm25_search_ms" not in d["timing"]
        assert "year" not in d["recommendations"][0]

        # to_json: valid JSON round-trip
        parsed = json.loads(response.to_json())
        assert parsed["query"] == "test query"
        assert parsed["recommendations"][0]["paper_id"] == "abc123"


def _write_corpus(tmp_path, papers, filename="test_corpus.jsonl"):
    """Helper to write a corpus JSONL file."""
    corpus_path = tmp_path / filename
    with open(corpus_path, "w") as f:
        for paper in papers:
            f.write(json.dumps(paper) + "\n")
    return corpus_path


class TestInCiteAgentFromCorpus:
    """Tests for inCiteAgent.from_corpus() with a minimal test corpus."""

    @pytest.fixture
    def test_corpus(self, tmp_path):
        """Create a minimal corpus for testing."""
        return _write_corpus(tmp_path, [
            {
                "id": "paper1",
                "title": "Deep Learning for NLP",
                "abstract": "This paper presents deep learning methods for natural language processing.",
                "authors": ["Smith, John", "Doe, Jane"],
                "year": 2023,
            },
            {
                "id": "paper2",
                "title": "Machine Learning Basics",
                "abstract": "An introduction to machine learning concepts and algorithms.",
                "authors": ["Wilson, Bob"],
                "year": 2022,
            },
            {
                "id": "paper3",
                "title": "Neural Network Architectures",
                "abstract": "A survey of neural network architectures for various tasks.",
                "authors": ["Jones, Alice", "Brown, Charlie"],
                "year": 2024,
            },
        ])

    def test_from_corpus_creates_agent(self, test_corpus):
        """Test that from_corpus creates a working agent."""
        agent = InCiteAgent.from_corpus(
            corpus_path=str(test_corpus),
            method="bm25",  # BM25 is fastest for testing
        )
        assert agent is not None
        assert len(agent._papers) == 3

    def test_recommend_returns_response(self, test_corpus):
        """Test that recommend returns an AgentResponse with proper structure."""
        agent = InCiteAgent.from_corpus(
            corpus_path=str(test_corpus),
            method="bm25",
        )
        response = agent.recommend("deep learning natural language", k=2)

        assert isinstance(response, AgentResponse)
        assert len(response.recommendations) <= 2
        assert response.corpus_size == 3
        assert response.method == "bm25"

    def test_recommend_timing_populated(self, test_corpus):
        """Test that timing information is populated."""
        agent = InCiteAgent.from_corpus(
            corpus_path=str(test_corpus),
            method="bm25",
        )
        response = agent.recommend("machine learning", k=2)

        # Timing should be populated
        assert response.timing.total_ms > 0
        assert response.timing.bm25_search_ms is not None
        assert response.timing.bm25_search_ms >= 0

    def test_get_stats_returns_dict(self, test_corpus):
        """Test that get_stats returns expected statistics."""
        agent = InCiteAgent.from_corpus(
            corpus_path=str(test_corpus),
            method="bm25",
        )
        stats = agent.get_stats()

        assert stats["corpus_size"] == 3
        assert stats["method"] == "bm25"
        assert stats["mode"] == "paper"
        assert stats["papers_with_abstract"] == 3
        assert stats["papers_with_year"] == 3

    def test_batch_recommend_sequential(self, test_corpus):
        """Test batch recommend in sequential mode."""
        agent = InCiteAgent.from_corpus(
            corpus_path=str(test_corpus),
            method="bm25",
        )
        queries = ["deep learning", "machine learning"]
        responses = agent.batch_recommend(queries, k=2, parallel=False)

        assert len(responses) == 2
        assert all(isinstance(r, AgentResponse) for r in responses)

    def test_batch_recommend_parallel(self, test_corpus):
        """Test batch recommend in parallel mode."""
        agent = InCiteAgent.from_corpus(
            corpus_path=str(test_corpus),
            method="bm25",
        )
        queries = ["deep learning", "neural networks", "machine learning"]
        responses = agent.batch_recommend(queries, k=2, parallel=True)

        assert len(responses) == 3
        # Check order is preserved
        assert "deep learning" in responses[0].query
        assert "neural" in responses[1].query

    def test_response_json_serializable(self, test_corpus):
        """Test that the full response chain is JSON serializable."""
        agent = InCiteAgent.from_corpus(
            corpus_path=str(test_corpus),
            method="bm25",
        )
        response = agent.recommend("deep learning", k=2)

        # Should not raise
        json_str = response.to_json()
        parsed = json.loads(json_str)

        # Check structure
        assert "query" in parsed
        assert "recommendations" in parsed
        assert "timing" in parsed
        assert "corpus_size" in parsed


class TestInCiteAgentHybrid:
    """Tests specifically for hybrid retrieval mode."""

    @pytest.fixture
    def test_corpus(self, tmp_path):
        """Create a corpus for testing hybrid retrieval."""
        return _write_corpus(tmp_path, [
            {
                "id": "paper1",
                "title": "Climate Change Impacts",
                "abstract": "Study of climate change impacts on global ecosystems.",
                "authors": ["Smith, John"],
                "year": 2023,
            },
            {
                "id": "paper2",
                "title": "Environmental Policy",
                "abstract": "Analysis of environmental policy effectiveness.",
                "authors": ["Doe, Jane"],
                "year": 2022,
            },
        ])

    @pytest.mark.slow
    def test_hybrid_recommend(self, test_corpus):
        """Test hybrid retrieval (may be slow due to model loading)."""
        agent = InCiteAgent.from_corpus(
            corpus_path=str(test_corpus),
            method="hybrid",
            embedder_type="minilm",
        )
        response = agent.recommend("climate change effects", k=2)

        assert response.method == "hybrid"
        assert response.embedder == "minilm"
        # Hybrid should have both embed and fusion timing
        assert response.timing.embed_query_ms > 0
        assert response.timing.fusion_ms is not None

    @pytest.mark.slow
    def test_neural_recommend(self, test_corpus):
        """Test neural-only retrieval."""
        agent = InCiteAgent.from_corpus(
            corpus_path=str(test_corpus),
            method="neural",
            embedder_type="minilm",
        )
        response = agent.recommend("environmental impact", k=2)

        assert response.method == "neural"
        assert response.timing.embed_query_ms > 0
        assert response.timing.vector_search_ms > 0
        # Neural-only shouldn't have BM25 or fusion timing
        assert response.timing.bm25_search_ms is None
