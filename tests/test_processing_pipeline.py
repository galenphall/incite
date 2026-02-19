"""Tests for the processing pipeline abstraction (corpus/pipeline.py)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from incite.corpus.pipeline import (
    CloudPipeline,
    LocalPipeline,
    ProcessingPipeline,
    ProcessingStatus,
    get_pipeline,
)
from incite.models import Chunk, Paper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_paper(tmp_path):
    """Create a sample Paper with a minimal PDF stub."""
    return Paper(
        id="test_paper_001",
        title="Test Paper on Climate Change",
        abstract="This paper examines the effects of CO2 on global temperature.",
        authors=["Smith, J.", "Jones, A."],
        year=2023,
        source_file=None,
    )


@pytest.fixture
def paper_with_pdf(tmp_path):
    """Create a sample Paper with a fake source file path."""
    # We don't create a real PDF; tests mock extract_pdf_text
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake content")
    return Paper(
        id="test_paper_002",
        title="Paper With PDF",
        abstract="Abstract of the paper with PDF.",
        authors=["Doe, J."],
        year=2022,
        source_file=str(pdf_path),
    )


# ---------------------------------------------------------------------------
# ProcessingStatus
# ---------------------------------------------------------------------------


class TestProcessingStatus:
    def test_defaults(self):
        status = ProcessingStatus(paper_id="p1", state="pending")
        assert status.progress == 0.0
        assert status.message == ""
        assert status.num_chunks == 0
        assert status.error is None

    def test_completed(self):
        status = ProcessingStatus(
            paper_id="p1", state="completed", progress=1.0, num_chunks=15
        )
        assert status.state == "completed"
        assert status.num_chunks == 15


# ---------------------------------------------------------------------------
# LocalPipeline
# ---------------------------------------------------------------------------


class TestLocalPipeline:
    def test_abstract_only_when_no_pdf(self, sample_paper):
        """Papers without PDFs should produce a single abstract chunk."""
        pipeline = LocalPipeline()
        chunks = pipeline.process_paper(sample_paper)
        assert len(chunks) == 1
        assert chunks[0].text == sample_paper.abstract
        assert chunks[0].section == "Abstract"
        assert chunks[0].paper_id == sample_paper.id

    def test_empty_when_no_pdf_no_abstract(self):
        """Papers with no PDF and no abstract return empty."""
        paper = Paper(id="empty", title="No Abstract", abstract="")
        pipeline = LocalPipeline()
        chunks = pipeline.process_paper(paper)
        assert chunks == []

    def test_process_paper_with_pdf(self, paper_with_pdf):
        """When PDF extraction works, chunker is called."""
        mock_result = MagicMock()
        mock_result.full_text = "Some extracted text from the PDF."
        mock_result.paragraphs = ["Para 1.", "Para 2."]

        expected_chunks = [
            Chunk(id="test_paper_002::chunk_0", paper_id="test_paper_002", text="Para 1."),
            Chunk(id="test_paper_002::chunk_1", paper_id="test_paper_002", text="Para 2."),
        ]

        mock_chunker = MagicMock(return_value=expected_chunks)

        with (
            patch("incite.corpus.pdf_extractor.extract_pdf_text", return_value=mock_result),
            patch("incite.retrieval.factory.get_chunker", return_value=mock_chunker),
        ):
            pipeline = LocalPipeline()
            chunks = pipeline.process_paper(paper_with_pdf)

        assert chunks == expected_chunks
        mock_chunker.assert_called_once()

    def test_fallback_to_abstract_on_empty_extraction(self, paper_with_pdf):
        """If PDF extraction returns empty text, fall back to abstract."""
        mock_result = MagicMock()
        mock_result.full_text = ""
        mock_result.paragraphs = []

        with patch("incite.corpus.pdf_extractor.extract_pdf_text", return_value=mock_result):
            pipeline = LocalPipeline()
            chunks = pipeline.process_paper(paper_with_pdf)

        assert len(chunks) == 1
        assert chunks[0].text == paper_with_pdf.abstract

    def test_process_batch(self, sample_paper):
        """Batch processing should process each paper."""
        pipeline = LocalPipeline()
        papers = [sample_paper]

        callback_calls = []

        def callback(current, total, msg):
            callback_calls.append((current, total, msg))

        chunks = pipeline.process_batch(papers, progress_callback=callback)
        assert len(chunks) == 1  # One abstract chunk
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == 1
        assert callback_calls[0][1] == 1

    def test_get_status_returns_none(self):
        """LocalPipeline is synchronous, so get_status returns None."""
        pipeline = LocalPipeline()
        assert pipeline.get_status("any_id") is None


# ---------------------------------------------------------------------------
# CloudPipeline
# ---------------------------------------------------------------------------


class TestCloudPipeline:
    def test_init(self):
        pipeline = CloudPipeline(api_url="https://api.example.com/", api_key="key123")
        assert pipeline.api_url == "https://api.example.com"  # trailing slash stripped
        assert pipeline.api_key == "key123"

    def test_headers_with_key(self):
        pipeline = CloudPipeline(api_url="https://api.example.com", api_key="test-key")
        headers = pipeline._headers()
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Accept"] == "application/json"

    def test_headers_without_key(self):
        pipeline = CloudPipeline(api_url="https://api.example.com")
        headers = pipeline._headers()
        assert "Authorization" not in headers
        assert headers["Accept"] == "application/json"

    def test_process_paper_no_pdf(self, sample_paper):
        """Papers without PDFs return empty from cloud pipeline."""
        pipeline = CloudPipeline(api_url="https://api.example.com")
        chunks = pipeline.process_paper(sample_paper)
        assert chunks == []

    def test_check_health_success(self):
        """Health check succeeds when API returns 200."""
        pipeline = CloudPipeline(api_url="https://api.example.com")
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            assert pipeline.check_health() is True

    def test_check_health_failure(self):
        """Health check returns False on network error."""
        pipeline = CloudPipeline(api_url="https://api.example.com")

        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            assert pipeline.check_health() is False

    def test_upload_failure_returns_empty(self, paper_with_pdf):
        """If upload fails, process_paper returns empty list."""
        pipeline = CloudPipeline(api_url="https://api.example.com", timeout=1)

        with patch.object(pipeline, "_upload_pdf", return_value=None):
            chunks = pipeline.process_paper(paper_with_pdf)
            assert chunks == []

    def test_download_chunks_parses_response(self):
        """_download_chunks correctly parses the API response into Chunk objects."""
        pipeline = CloudPipeline(api_url="https://api.example.com")

        response_data = {
            "chunks": [
                {
                    "id": "p1::chunk_0",
                    "paper_id": "p1",
                    "text": "First paragraph",
                    "section": "Introduction",
                    "page_number": 1,
                },
                {
                    "id": "p1::chunk_1",
                    "paper_id": "p1",
                    "text": "Second paragraph",
                    "section": "Methods",
                    "page_number": 3,
                },
            ]
        }

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            chunks = pipeline._download_chunks("job123")

        assert len(chunks) == 2
        assert chunks[0].id == "p1::chunk_0"
        assert chunks[0].page_number == 1
        assert chunks[1].section == "Methods"
        assert chunks[1].page_number == 3


# ---------------------------------------------------------------------------
# get_pipeline factory
# ---------------------------------------------------------------------------


class TestGetPipeline:
    def test_default_is_local(self):
        """Without config, default pipeline is local."""
        pipeline = get_pipeline(mode="local")
        assert isinstance(pipeline, LocalPipeline)

    def test_explicit_local(self):
        pipeline = get_pipeline(mode="local", chunking_strategy="paragraph")
        assert isinstance(pipeline, LocalPipeline)
        assert pipeline.chunking_strategy == "paragraph"

    def test_cloud_fallback_to_local_without_config(self):
        """Cloud mode without config falls back to local."""
        with patch("incite.webapp.state.get_config", return_value={}):
            pipeline = get_pipeline(mode="cloud")
            assert isinstance(pipeline, LocalPipeline)

    def test_cloud_with_config(self):
        """Cloud mode with proper config returns CloudPipeline."""
        config = {
            "cloud": {"api_url": "https://api.example.com", "api_key": "key123"},
        }
        with patch("incite.webapp.state.get_config", return_value=config):
            pipeline = get_pipeline(mode="cloud")
            assert isinstance(pipeline, CloudPipeline)
            assert pipeline.api_url == "https://api.example.com"
            assert pipeline.api_key == "key123"

    def test_auto_reads_config(self):
        """When mode is None, reads from config."""
        config = {"processing": {"mode": "local"}}
        with patch("incite.webapp.state.get_config", return_value=config):
            pipeline = get_pipeline(mode=None)
            assert isinstance(pipeline, LocalPipeline)
