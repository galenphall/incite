"""Tests for Unpaywall API client."""

from unittest.mock import MagicMock, patch

import pytest

from incite.acquire.unpaywall import UnpaywallClient, UnpaywallResult


@pytest.fixture
def client():
    return UnpaywallClient(email="test@example.com", delay=0.0)


class TestUnpaywallResult:
    def test_oa_result(self):
        result = UnpaywallResult(
            doi="10.1038/nature12373",
            is_oa=True,
            best_oa_url="https://europepmc.org/articles/pmc3814764?pdf=render",
            publisher_pdf_url="https://www.nature.com/articles/nature12373.pdf",
            oa_status="green",
        )
        assert result.is_oa
        assert result.best_oa_url is not None
        assert result.oa_status == "green"

    def test_closed_result(self):
        result = UnpaywallResult(
            doi="10.1016/j.cell.2023.12.004",
            is_oa=False,
            best_oa_url=None,
            publisher_pdf_url="https://www.cell.com/cell/pdf/S0092-8674(23)01507-6.pdf",
            oa_status="closed",
        )
        assert not result.is_oa
        assert result.best_oa_url is None
        assert result.publisher_pdf_url is not None


class TestUnpaywallClient:
    def test_lookup_oa_paper(self, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "doi": "10.1038/nature12373",
            "is_oa": True,
            "oa_status": "green",
            "best_oa_location": {
                "url_for_pdf": "https://europepmc.org/articles/pmc3814764?pdf=render",
                "host_type": "repository",
            },
            "oa_locations": [
                {
                    "host_type": "publisher",
                    "url_for_pdf": "https://www.nature.com/articles/nature12373.pdf",
                    "url": "https://www.nature.com/articles/nature12373",
                },
                {
                    "host_type": "repository",
                    "url_for_pdf": "https://europepmc.org/articles/pmc3814764?pdf=render",
                },
            ],
        }

        with patch("incite.acquire.unpaywall.requests.get", return_value=mock_response):
            result = client.lookup("10.1038/nature12373")

        assert result is not None
        assert result.doi == "10.1038/nature12373"
        assert result.is_oa is True
        assert result.oa_status == "green"
        assert "europepmc" in result.best_oa_url
        assert "nature.com" in result.publisher_pdf_url

    def test_lookup_closed_paper(self, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "doi": "10.1016/j.cell.2023.12.004",
            "is_oa": False,
            "oa_status": "closed",
            "best_oa_location": None,
            "oa_locations": [],
        }

        with patch("incite.acquire.unpaywall.requests.get", return_value=mock_response):
            result = client.lookup("10.1016/j.cell.2023.12.004")

        assert result is not None
        assert result.is_oa is False
        assert result.best_oa_url is None
        assert result.oa_status == "closed"

    def test_lookup_not_found(self, client):
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("incite.acquire.unpaywall.requests.get", return_value=mock_response):
            result = client.lookup("10.xxxx/nonexistent")

        assert result is None

    def test_lookup_network_error(self, client):
        import requests

        with patch(
            "incite.acquire.unpaywall.requests.get",
            side_effect=requests.RequestException("Connection error"),
        ):
            result = client.lookup("10.1038/nature12373")

        assert result is None

    def test_batch_lookup(self, client):
        dois = ["10.1038/nature12373", "10.xxxx/missing"]
        mock_responses = {
            "10.1038/nature12373": {
                "doi": "10.1038/nature12373",
                "is_oa": True,
                "oa_status": "green",
                "best_oa_location": {"url_for_pdf": "https://example.com/pdf"},
                "oa_locations": [],
            },
        }

        def mock_get(url, **kwargs):
            # Extract DOI from URL
            doi = url.split("/v2/")[-1]
            if doi in mock_responses:
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = mock_responses[doi]
                return resp
            else:
                resp = MagicMock()
                resp.status_code = 404
                return resp

        with patch("incite.acquire.unpaywall.requests.get", side_effect=mock_get):
            results = client.batch_lookup(dois)

        assert len(results) == 1
        assert "10.1038/nature12373" in results
        assert "10.xxxx/missing" not in results

    def test_rate_limiting(self, client):
        """Test that rate limiting doesn't crash (timing not tested)."""
        client.delay = 0.0  # No actual delay for test

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("incite.acquire.unpaywall.requests.get", return_value=mock_response):
            client.lookup("10.1038/a")
            client.lookup("10.1038/b")

    def test_email_param_sent(self, client):
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch(
            "incite.acquire.unpaywall.requests.get", return_value=mock_response
        ) as mock_get:
            client.lookup("10.1038/nature12373")

        # Verify email was passed as query parameter
        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["params"]["email"] == "test@example.com"
