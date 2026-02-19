"""Shared pytest configuration and fixtures."""

import hashlib

import numpy as np
import pytest

from incite.embeddings.base import BaseEmbedder
from incite.models import Chunk, Paper

# ---------------------------------------------------------------------------
# E2E cloud test configuration
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    """Add custom CLI options for e2e cloud tests."""
    parser.addoption(
        "--api-url",
        default="http://localhost:9100",
        help="Base URL for the cloud processing API (default: http://localhost:9100)",
    )
    parser.addoption(
        "--api-key",
        default="",
        help="API key for authenticated endpoints (default: empty = no auth)",
    )
    parser.addoption(
        "--invite-code",
        default="",
        help="Invite code for signup during e2e tests (default: empty)",
    )


@pytest.fixture(scope="session")
def api_url(request):
    """Base URL of the cloud processing server."""
    return request.config.getoption("--api-url").rstrip("/")


@pytest.fixture(scope="session")
def api_key(request):
    """API key for Bearer auth (empty string means no auth)."""
    return request.config.getoption("--api-key")


@pytest.fixture(scope="session")
def api_headers(api_key):
    """HTTP headers dict with Bearer token if api_key is set."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


@pytest.fixture(scope="session")
def invite_code(request):
    """Invite code for signup during e2e tests."""
    return request.config.getoption("--invite-code")


# ---------------------------------------------------------------------------
# MockEmbedder — deterministic dim-8 embedder for fast unit tests
# ---------------------------------------------------------------------------

_MOCK_DIM = 8


class MockEmbedder(BaseEmbedder):
    """Deterministic embedder that produces L2-normalized dim-8 vectors.

    Same text always yields the same vector. Different texts yield different
    vectors (with overwhelming probability). Suitable for FAISS inner-product
    search because all vectors are unit-normalized.
    """

    @property
    def dimension(self) -> int:
        return _MOCK_DIM

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        vecs = np.zeros((len(texts), _MOCK_DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(_MOCK_DIM).astype(np.float32)
            vec /= np.linalg.norm(vec)
            vecs[i] = vec
        return vecs


@pytest.fixture(scope="session")
def mock_embedder():
    return MockEmbedder()


# ---------------------------------------------------------------------------
# Sample papers — two topic clusters (climate + NLP)
# ---------------------------------------------------------------------------

_PAPERS = [
    Paper(
        id="climate_1",
        title="Sea level rise projections",
        abstract="Global sea levels are rising due to climate change and thermal expansion of oceans.",
        authors=["Smith, John", "Jones, Alice"],
        year=2023,
    ),
    Paper(
        id="climate_2",
        title="Ocean temperature trends",
        abstract="Sea surface temperatures and oceanic heat content have been increasing over decades.",
        authors=["Chen, Wei"],
        year=2022,
    ),
    Paper(
        id="climate_3",
        title="Climate modeling techniques",
        abstract="General circulation models for climate projections and emission scenarios.",
        authors=["Park, Soo-Jin", "Kim, Daeho"],
        year=2021,
    ),
    Paper(
        id="climate_4",
        title="Coastal flood risk assessment",
        abstract="Evaluating coastal flood risk under different sea level rise and storm surge scenarios.",
        authors=["Smith, John"],
        year=2024,
    ),
    Paper(
        id="nlp_1",
        title="Deep learning for NLP",
        abstract="Neural networks have revolutionized natural language processing tasks.",
        authors=["Li, Ming"],
        year=2023,
    ),
    Paper(
        id="nlp_2",
        title="Transformer architectures for text",
        abstract="Self-attention mechanisms in transformers enable parallel text processing.",
        authors=["Brown, Tom", "Wu, Jeff"],
        year=2022,
    ),
]


@pytest.fixture(scope="session")
def sample_papers():
    return list(_PAPERS)


@pytest.fixture(scope="session")
def sample_paper_dict():
    return {p.id: p for p in _PAPERS}


# ---------------------------------------------------------------------------
# Sample chunks — 2-3 per paper
# ---------------------------------------------------------------------------

_CHUNKS = [
    Chunk(
        id="climate_1::chunk_0",
        paper_id="climate_1",
        text="Sea levels are rising globally.",
        parent_text="Sea levels are rising globally. This is caused by thermal expansion and ice sheet melt.",
    ),
    Chunk(
        id="climate_1::chunk_1",
        paper_id="climate_1",
        text="Thermal expansion causes ocean volume increase.",
    ),
    Chunk(
        id="climate_2::chunk_0",
        paper_id="climate_2",
        text="Sea surface temperatures have risen by 0.8C.",
        parent_text="Sea surface temperatures have risen by 0.8C. Oceanic heat content continues to grow.",
    ),
    Chunk(
        id="climate_2::chunk_1",
        paper_id="climate_2",
        text="Ocean heat content measurements from Argo floats.",
    ),
    Chunk(
        id="climate_3::chunk_0",
        paper_id="climate_3",
        text="General circulation models simulate atmospheric dynamics.",
    ),
    Chunk(
        id="climate_3::chunk_1",
        paper_id="climate_3",
        text="Emission scenarios drive future climate projections.",
    ),
    Chunk(
        id="climate_4::chunk_0",
        paper_id="climate_4",
        text="Coastal communities face increasing flood risk.",
    ),
    Chunk(
        id="climate_4::chunk_1",
        paper_id="climate_4",
        text="Storm surge combined with sea level rise amplifies flooding.",
    ),
    Chunk(
        id="nlp_1::chunk_0",
        paper_id="nlp_1",
        text="Deep learning models achieve state-of-the-art NLP performance.",
        parent_text="Deep learning models achieve state-of-the-art NLP performance. Recurrent and transformer architectures dominate.",
    ),
    Chunk(
        id="nlp_1::chunk_1",
        paper_id="nlp_1",
        text="Word embeddings capture semantic relationships between words.",
    ),
    Chunk(
        id="nlp_2::chunk_0",
        paper_id="nlp_2",
        text="Transformers use self-attention for parallel sequence processing.",
    ),
    Chunk(
        id="nlp_2::chunk_1",
        paper_id="nlp_2",
        text="Multi-head attention enables learning diverse representations.",
    ),
]


@pytest.fixture(scope="session")
def sample_chunks():
    return list(_CHUNKS)


@pytest.fixture(scope="session")
def sample_chunk_dict():
    return {c.id: c for c in _CHUNKS}
