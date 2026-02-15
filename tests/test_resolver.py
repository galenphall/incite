"""Tests for DOI resolution and citation_pdf_url extraction."""

from unittest.mock import MagicMock, patch

from incite.acquire.resolver import doi_slug, resolve_doi
from incite.acquire.session import _extract_citation_pdf_url


class TestDoiSlug:
    def test_basic(self):
        assert doi_slug("10.1038/nature12373") == "10.1038_nature12373"

    def test_nested_path(self):
        assert doi_slug("10.1016/j.cell.2023.12.004") == "10.1016_j.cell.2023.12.004"

    def test_acs(self):
        assert doi_slug("10.1021/acs.est.1c02147") == "10.1021_acs.est.1c02147"


class TestResolveDoi:
    def test_successful_resolution(self):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.url = "https://www.nature.com/articles/nature12373"

        with patch("incite.acquire.resolver.requests.head", return_value=mock_resp):
            result = resolve_doi("10.1038/nature12373")

        assert result == "https://www.nature.com/articles/nature12373"

    def test_failed_resolution(self):
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.url = "https://doi.org/10.xxxx/missing"  # No redirect happened

        with patch("incite.acquire.resolver.requests.head", return_value=mock_resp):
            result = resolve_doi("10.xxxx/missing")

        assert result is None

    def test_failed_resolution_with_redirect(self):
        """Publishers that return 403 to HEAD but still redirect should return the URL."""
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.url = "https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023GL106553"

        with patch("incite.acquire.resolver.requests.head", return_value=mock_resp):
            result = resolve_doi("10.1029/2023GL106553")

        assert result == "https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023GL106553"

    def test_network_error(self):
        import requests

        with patch(
            "incite.acquire.resolver.requests.head",
            side_effect=requests.RequestException("timeout"),
        ):
            result = resolve_doi("10.1038/nature12373")

        assert result is None


class TestExtractCitationPdfUrl:
    """Test citation_pdf_url meta tag extraction."""

    def test_standard_meta_tag(self):
        html = '<meta name="citation_pdf_url" content="https://example.com/paper.pdf">'
        assert _extract_citation_pdf_url(html) == "https://example.com/paper.pdf"

    def test_reversed_attributes(self):
        html = '<meta content="https://example.com/paper.pdf" name="citation_pdf_url">'
        assert _extract_citation_pdf_url(html) == "https://example.com/paper.pdf"

    def test_no_meta_tag(self):
        html = "<html><head><title>No PDF</title></head></html>"
        assert _extract_citation_pdf_url(html) is None

    def test_single_quotes(self):
        html = "<meta name='citation_pdf_url' content='https://example.com/paper.pdf'>"
        assert _extract_citation_pdf_url(html) == "https://example.com/paper.pdf"

    def test_extra_attributes(self):
        html = '<meta name="citation_pdf_url" content="https://example.com/p.pdf" scheme="URL">'
        assert _extract_citation_pdf_url(html) == "https://example.com/p.pdf"
