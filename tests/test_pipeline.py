"""Tests for the PDF acquisition pipeline."""

import json
from unittest.mock import MagicMock, patch

import pytest

from incite.acquire.config import ProxyConfig
from incite.acquire.pipeline import AcquisitionPipeline, AcquisitionResult, AcquisitionSummary
from incite.acquire.unpaywall import UnpaywallResult
from incite.models import Paper


@pytest.fixture
def tmp_dest(tmp_path):
    dest = tmp_path / "acquired"
    dest.mkdir()
    return dest


@pytest.fixture
def sample_papers():
    return [
        Paper(id="p1", title="OA Paper", doi="10.1038/nature12373"),
        Paper(id="p2", title="Closed Paper", doi="10.1016/j.cell.2023.12.004"),
        Paper(id="p3", title="No DOI Paper"),
        Paper(
            id="p4", title="Already Has PDF",
            doi="10.1234/existing", source_file="/fake/path.pdf",
        ),
    ]


class TestAcquisitionSummary:
    def test_counts(self):
        summary = AcquisitionSummary(results=[
            AcquisitionResult(doi="a", status="acquired", source="unpaywall_oa"),
            AcquisitionResult(doi="b", status="acquired", source="proxy"),
            AcquisitionResult(doi="c", status="skipped", source="existing"),
            AcquisitionResult(doi="d", status="failed", error="Not found"),
        ])
        assert summary.acquired == 2
        assert summary.skipped == 1
        assert summary.failed == 1
        assert summary.by_source == {"unpaywall_oa": 1, "proxy": 1}

    def test_empty_summary(self):
        summary = AcquisitionSummary()
        assert summary.acquired == 0
        assert summary.skipped == 0
        assert summary.failed == 0
        assert summary.by_source == {}


class TestAcquisitionPipeline:
    def test_paper_without_doi_fails(self, tmp_dest):
        pipeline = AcquisitionPipeline(
            dest_dir=tmp_dest,
            email="test@example.com",
            free_only=True,
        )
        paper = Paper(id="no-doi", title="No DOI")
        result = pipeline.acquire_paper(paper)
        assert result.status == "failed"
        assert "No DOI" in result.error

    def test_existing_pdf_skipped(self, tmp_dest):
        # Create a fake existing PDF
        doi = "10.1038/nature12373"
        pdf_path = tmp_dest / "10.1038_nature12373.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")

        pipeline = AcquisitionPipeline(
            dest_dir=tmp_dest,
            email="test@example.com",
            free_only=True,
        )
        paper = Paper(id="p1", title="Test", doi=doi)
        result = pipeline.acquire_paper(paper)
        assert result.status == "skipped"
        assert result.source == "existing"

    def test_existing_source_file_skipped(self, tmp_dest, tmp_path):
        # Create a fake source file
        source_pdf = tmp_path / "paper.pdf"
        source_pdf.write_bytes(b"%PDF-1.4 content")

        pipeline = AcquisitionPipeline(
            dest_dir=tmp_dest,
            email="test@example.com",
            free_only=True,
        )
        paper = Paper(id="p1", title="Test", doi="10.1038/test", source_file=str(source_pdf))
        result = pipeline.acquire_paper(paper)
        assert result.status == "skipped"
        assert result.source == "existing"

    def test_oa_download(self, tmp_dest):
        """Test OA paper is downloaded directly from Unpaywall URL."""
        uw_result = UnpaywallResult(
            doi="10.1038/nature12373",
            is_oa=True,
            best_oa_url="https://example.com/oa.pdf",
            publisher_pdf_url=None,
            oa_status="green",
        )

        pipeline = AcquisitionPipeline(
            dest_dir=tmp_dest,
            email="test@example.com",
            free_only=True,
        )

        with patch.object(pipeline._unpaywall, "lookup", return_value=uw_result):
            with patch(
                "incite.acquire.pipeline._download_url",
                return_value=tmp_dest / "10.1038_nature12373.pdf",
            ):
                paper = Paper(id="p1", title="OA Paper", doi="10.1038/nature12373")
                result = pipeline.acquire_paper(paper)

        assert result.status == "acquired"
        assert result.source == "unpaywall_oa"

    def test_free_only_skips_proxy(self, tmp_dest):
        """Test that --free-only mode never tries proxy."""
        uw_result = UnpaywallResult(
            doi="10.1016/j.cell.2023.12.004",
            is_oa=False,
            best_oa_url=None,
            publisher_pdf_url="https://publisher.com/pdf",
            oa_status="closed",
        )

        pipeline = AcquisitionPipeline(
            dest_dir=tmp_dest,
            email="test@example.com",
            proxy_config=ProxyConfig(proxy_type="ezproxy_prefix", proxy_url="https://proxy.example.com/login?url="),
            free_only=True,  # Key: free_only=True
        )

        with patch.object(pipeline._unpaywall, "lookup", return_value=uw_result):
            with patch("incite.acquire.pipeline._download_url", return_value=None):
                paper = Paper(id="p2", title="Closed", doi="10.1016/j.cell.2023.12.004")
                result = pipeline.acquire_paper(paper)

        # Should fail because free_only skips proxy
        assert result.status == "failed"

    def test_dry_run(self, tmp_dest):
        pipeline = AcquisitionPipeline(
            dest_dir=tmp_dest,
            email="test@example.com",
            dry_run=True,
        )
        paper = Paper(id="p1", title="Test", doi="10.1038/nature12373")
        result = pipeline.acquire_paper(paper)
        assert result.status == "acquired"
        assert result.source == "dry_run"

    def test_proxy_with_publisher_url(self, tmp_dest):
        """Test proxy download using Unpaywall's publisher URL."""
        uw_result = UnpaywallResult(
            doi="10.1016/j.cell.2023.12.004",
            is_oa=False,
            best_oa_url=None,
            publisher_pdf_url="https://publisher.com/article.pdf",
            oa_status="closed",
        )

        mock_proxy = MagicMock()
        mock_proxy.ensure_authenticated.return_value = True
        mock_proxy.download_pdf.return_value = tmp_dest / "10.1016_j.cell.2023.12.004.pdf"

        pipeline = AcquisitionPipeline(
            dest_dir=tmp_dest,
            email="test@example.com",
            proxy_config=ProxyConfig(proxy_type="ezproxy_prefix", proxy_url="https://proxy.example.com/login?url="),
        )
        pipeline._proxy = mock_proxy
        pipeline._proxy_authenticated = True

        with patch.object(pipeline._unpaywall, "lookup", return_value=uw_result):
            with patch("incite.acquire.pipeline._download_url", return_value=None):
                paper = Paper(id="p2", title="Closed", doi="10.1016/j.cell.2023.12.004")
                result = pipeline.acquire_paper(paper)

        assert result.status == "acquired"
        assert result.source == "proxy"
        mock_proxy.download_pdf.assert_called_once_with(
            "https://publisher.com/article.pdf",
            tmp_dest / "10.1016_j.cell.2023.12.004.pdf",
        )

    def test_proxy_with_doi_resolution(self, tmp_dest):
        """Test proxy download via DOI resolution to landing page."""
        uw_result = UnpaywallResult(
            doi="10.1038/nature12373",
            is_oa=False,
            best_oa_url=None,
            publisher_pdf_url=None,  # No publisher URL from Unpaywall
            oa_status="closed",
        )

        mock_proxy = MagicMock()
        mock_proxy.ensure_authenticated.return_value = True
        mock_proxy.download_pdf.return_value = tmp_dest / "10.1038_nature12373.pdf"

        pipeline = AcquisitionPipeline(
            dest_dir=tmp_dest,
            email="test@example.com",
            proxy_config=ProxyConfig(proxy_type="ezproxy_prefix", proxy_url="https://proxy.example.com/login?url="),
        )
        pipeline._proxy = mock_proxy
        pipeline._proxy_authenticated = True

        with patch.object(pipeline._unpaywall, "lookup", return_value=uw_result):
            with patch("incite.acquire.pipeline._download_url", return_value=None):
                with patch(
                    "incite.acquire.pipeline.resolve_doi",
                    return_value="https://www.nature.com/articles/nature12373",
                ):
                    paper = Paper(id="p1", title="Nature paper", doi="10.1038/nature12373")
                    result = pipeline.acquire_paper(paper)

        assert result.status == "acquired"
        assert result.source == "proxy_doi"

    def test_priority_order_oa_first(self, tmp_dest):
        """OA should be tried before proxy."""
        uw_result = UnpaywallResult(
            doi="10.1038/nature12373",
            is_oa=True,
            best_oa_url="https://example.com/oa.pdf",
            publisher_pdf_url="https://publisher.com/article.pdf",
            oa_status="green",
        )

        mock_proxy = MagicMock()

        pipeline = AcquisitionPipeline(
            dest_dir=tmp_dest,
            email="test@example.com",
            proxy_config=ProxyConfig(proxy_type="ezproxy_prefix", proxy_url="https://proxy.example.com/"),
        )
        pipeline._proxy = mock_proxy
        pipeline._proxy_authenticated = True

        with patch.object(pipeline._unpaywall, "lookup", return_value=uw_result):
            with patch(
                "incite.acquire.pipeline._download_url",
                return_value=tmp_dest / "10.1038_nature12373.pdf",
            ):
                paper = Paper(id="p1", title="OA Paper", doi="10.1038/nature12373")
                result = pipeline.acquire_paper(paper)

        assert result.status == "acquired"
        assert result.source == "unpaywall_oa"
        # Proxy should NOT have been called
        mock_proxy.download_pdf.assert_not_called()

    def test_batch_acquire(self, tmp_dest):
        pipeline = AcquisitionPipeline(
            dest_dir=tmp_dest,
            email="test@example.com",
            free_only=True,
            dry_run=True,
        )

        papers = [
            Paper(id="p1", title="Paper 1", doi="10.1038/a"),
            Paper(id="p2", title="Paper 2", doi="10.1038/b"),
            Paper(id="p3", title="No DOI"),
        ]

        summary = pipeline.acquire_batch(papers)

        assert len(summary.results) == 3
        # dry_run acquires papers with DOIs
        assert summary.acquired == 2
        assert summary.failed == 1  # No DOI

    def test_batch_with_limit(self, tmp_dest):
        pipeline = AcquisitionPipeline(
            dest_dir=tmp_dest,
            email="test@example.com",
            dry_run=True,
        )

        papers = [
            Paper(id=f"p{i}", title=f"Paper {i}", doi=f"10.1038/{i}")
            for i in range(10)
        ]

        summary = pipeline.acquire_batch(papers, limit=3)
        assert len(summary.results) == 3

    def test_manifest_saved(self, tmp_dest):
        """Test that manifest.json is written after batch."""
        pipeline = AcquisitionPipeline(
            dest_dir=tmp_dest,
            email="test@example.com",
            free_only=True,
        )

        uw_result = UnpaywallResult(
            doi="10.1038/nature12373",
            is_oa=True,
            best_oa_url="https://example.com/oa.pdf",
            publisher_pdf_url=None,
            oa_status="green",
        )

        with patch.object(pipeline._unpaywall, "lookup", return_value=uw_result):
            with patch(
                "incite.acquire.pipeline._download_url",
                return_value=tmp_dest / "10.1038_nature12373.pdf",
            ):
                papers = [Paper(id="p1", title="OA Paper", doi="10.1038/nature12373")]
                pipeline.acquire_batch(papers)

        manifest_path = tmp_dest / "manifest.json"
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text())
        assert "acquired" in manifest
        assert len(manifest["acquired"]) == 1
        assert manifest["acquired"][0]["doi"] == "10.1038/nature12373"
        assert manifest["acquired"][0]["source"] == "unpaywall_oa"

    def test_manifest_merges_with_existing(self, tmp_dest):
        """Test that manifest merges with existing entries."""
        # Pre-existing manifest
        existing = {
            "acquired": [{"doi": "10.old/existing", "source": "unpaywall_oa"}],
            "failed": [],
        }
        (tmp_dest / "manifest.json").write_text(json.dumps(existing))

        uw_result = UnpaywallResult(
            doi="10.new/paper",
            is_oa=True,
            best_oa_url="https://example.com/oa.pdf",
            publisher_pdf_url=None,
            oa_status="green",
        )

        pipeline = AcquisitionPipeline(
            dest_dir=tmp_dest,
            email="test@example.com",
            free_only=True,
        )

        with patch.object(pipeline._unpaywall, "lookup", return_value=uw_result):
            with patch(
                "incite.acquire.pipeline._download_url",
                return_value=tmp_dest / "10.new_paper.pdf",
            ):
                papers = [Paper(id="p1", title="New Paper", doi="10.new/paper")]
                pipeline.acquire_batch(papers)

        manifest = json.loads((tmp_dest / "manifest.json").read_text())
        assert len(manifest["acquired"]) == 2
        dois = {e["doi"] for e in manifest["acquired"]}
        assert "10.old/existing" in dois
        assert "10.new/paper" in dois


class TestPdfValidation:
    def test_is_pdf(self):
        from incite.acquire.proxy import _is_pdf

        assert _is_pdf(b"%PDF-1.4 some content")
        assert not _is_pdf(b"<html>Not a PDF</html>")
        assert not _is_pdf(b"")
        assert not _is_pdf(b"%PD")  # Too short
