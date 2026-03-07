"""Tests for UnpaywallClient — all use unittest.mock, no real API calls."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from incite.acquire.unpaywall import UnpaywallClient, UnpaywallResult


@pytest.fixture
def client() -> UnpaywallClient:
    return UnpaywallClient(email="test@example.com")


class TestUnpaywallResult:
    def test_frozen(self) -> None:
        result = UnpaywallResult(is_oa=True, best_oa_url="https://example.com/paper.pdf")
        with pytest.raises(AttributeError):
            result.is_oa = False  # type: ignore[misc]


class TestUnpaywallClientLookup:
    @patch("incite.acquire.unpaywall.requests.get")
    def test_lookup_found(self, mock_get: MagicMock, client: UnpaywallClient) -> None:
        """200 with OA PDF URL returns UnpaywallResult with correct fields."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "is_oa": True,
                "best_oa_location": {
                    "url_for_pdf": "https://arxiv.org/pdf/2301.00001.pdf",
                    "url": "https://arxiv.org/abs/2301.00001",
                },
            },
        )

        result = client.lookup("10.1234/test")

        assert result is not None
        assert result.is_oa is True
        assert result.best_oa_url == "https://arxiv.org/pdf/2301.00001.pdf"
        mock_get.assert_called_once_with(
            "https://api.unpaywall.org/v2/10.1234/test",
            params={"email": "test@example.com"},
            timeout=10,
        )

    @patch("incite.acquire.unpaywall.requests.get")
    def test_lookup_no_oa(self, mock_get: MagicMock, client: UnpaywallClient) -> None:
        """200 but best_oa_location is None returns result with no URL."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "is_oa": False,
                "best_oa_location": None,
            },
        )

        result = client.lookup("10.1234/closed")

        assert result is not None
        assert result.is_oa is False
        assert result.best_oa_url is None

    @patch("incite.acquire.unpaywall.requests.get")
    def test_lookup_not_found(self, mock_get: MagicMock, client: UnpaywallClient) -> None:
        """404 returns None."""
        mock_get.return_value = MagicMock(status_code=404)

        result = client.lookup("10.1234/missing")

        assert result is None

    @patch("incite.acquire.unpaywall.requests.get")
    def test_lookup_timeout(self, mock_get: MagicMock, client: UnpaywallClient) -> None:
        """requests.Timeout returns None."""
        mock_get.side_effect = requests.Timeout("Connection timed out")

        result = client.lookup("10.1234/slow")

        assert result is None

    @patch("incite.acquire.unpaywall.requests.get")
    def test_lookup_prefers_pdf_url(self, mock_get: MagicMock, client: UnpaywallClient) -> None:
        """Falls back to url when url_for_pdf is None."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "is_oa": True,
                "best_oa_location": {
                    "url_for_pdf": None,
                    "url": "https://doi.org/10.1234/fallback",
                },
            },
        )

        result = client.lookup("10.1234/fallback")

        assert result is not None
        assert result.is_oa is True
        assert result.best_oa_url == "https://doi.org/10.1234/fallback"
