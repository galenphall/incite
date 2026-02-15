"""Tests for the FastAPI server."""

from unittest.mock import MagicMock

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed (pip install incite[api])")
from fastapi.testclient import TestClient

from incite import api as api_module
from incite.agent import AgentRecommendation, AgentResponse, TimingInfo
from incite.api import app


def _make_response(query="test query", n=2):
    """Build a fake AgentResponse."""
    recs = [
        AgentRecommendation(
            paper_id=f"paper_{i}",
            rank=i,
            score=1.0 - i * 0.1,
            title=f"Paper {i}",
            authors=["Author A"],
            year=2024,
        )
        for i in range(1, n + 1)
    ]
    return AgentResponse(
        query=query,
        recommendations=recs,
        timing=TimingInfo(total_ms=42.0, embed_query_ms=10.0, vector_search_ms=30.0),
        corpus_size=100,
        method="hybrid",
        embedder="minilm",
        timestamp="2026-01-01T00:00:00Z",
    )


@pytest.fixture()
def mock_agent():
    """Inject a mock InCiteAgent into the api module."""
    agent = MagicMock()
    agent._papers = {f"paper_{i}": MagicMock() for i in range(1, 101)}
    agent._method = "hybrid"
    agent._embedder_type = "minilm"
    agent._mode = "paper"
    # Public properties used by api.py
    agent.corpus_size = 100
    agent.method = "hybrid"
    agent.mode = "paper"
    agent.chunking_strategy = "paragraph"

    agent.get_stats.return_value = {
        "corpus_size": 100,
        "method": "hybrid",
        "embedder": "minilm",
        "mode": "paper",
    }
    agent.recommend.return_value = _make_response()
    agent.batch_recommend.return_value = [
        _make_response("q1"),
        _make_response("q2"),
    ]

    original = api_module._agent
    api_module._agent = agent
    yield agent
    api_module._agent = original


@pytest.fixture()
def client(mock_agent):
    return TestClient(app, raise_server_exceptions=False)


class TestHealth:
    def test_health_ready(self, client, mock_agent):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is True
        assert data["status"] == "ready"
        assert data["corpus_size"] == 100
        assert data["mode"] == "paper"

    def test_health_not_ready(self):
        original = api_module._agent
        api_module._agent = None
        try:
            c = TestClient(app, raise_server_exceptions=False)
            resp = c.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["ready"] is False
            assert data["status"] == "loading"
            assert "corpus_size" not in data
        finally:
            api_module._agent = original


class TestStats:
    def test_stats(self, client, mock_agent):
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["corpus_size"] == 100
        mock_agent.get_stats.assert_called_once()

    def test_stats_not_ready(self):
        original = api_module._agent
        api_module._agent = None
        try:
            c = TestClient(app, raise_server_exceptions=False)
            resp = c.get("/stats")
            assert resp.status_code == 503
        finally:
            api_module._agent = original


class TestConfig:
    def test_config(self, client, mock_agent):
        resp = client.get("/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["method"] == "hybrid"
        assert data["embedder"] == "minilm"
        assert data["mode"] == "paper"
        assert "minilm" in data["available_embedders"]
        assert "hybrid" in data["available_methods"]


class TestRecommend:
    def test_recommend_success(self, client, mock_agent):
        resp = client.post("/recommend", json={"query": "climate change"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "test query"  # from mock
        assert len(data["recommendations"]) == 2
        assert data["timing"]["total_ms"] == 42.0
        mock_agent.recommend.assert_called_once_with(
            query="climate change",
            k=10,
            author_boost=1.0,
            cursor_sentence_index=None,
            focus_decay=0.5,
        )

    def test_recommend_custom_params(self, client, mock_agent):
        resp = client.post(
            "/recommend",
            json={"query": "neural nets", "k": 5, "author_boost": 1.3},
        )
        assert resp.status_code == 200
        mock_agent.recommend.assert_called_once_with(
            query="neural nets",
            k=5,
            author_boost=1.3,
            cursor_sentence_index=None,
            focus_decay=0.5,
        )

    def test_recommend_empty_query(self, client, mock_agent):
        resp = client.post("/recommend", json={"query": "   "})
        assert resp.status_code == 422

    def test_recommend_missing_query(self, client, mock_agent):
        resp = client.post("/recommend", json={})
        assert resp.status_code == 422


class TestBatch:
    def test_batch_success(self, client, mock_agent):
        resp = client.post(
            "/batch",
            json={"queries": ["q1", "q2"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        mock_agent.batch_recommend.assert_called_once_with(
            queries=["q1", "q2"], k=10, author_boost=1.0
        )

    def test_batch_empty_list(self, client, mock_agent):
        resp = client.post("/batch", json={"queries": []})
        assert resp.status_code == 422
