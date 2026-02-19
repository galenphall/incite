"""Tests for cloud/error_reporter.py."""

import time
from unittest.mock import MagicMock, patch

import pytest

from cloud.error_reporter import (
    _MAX_REPORTS_PER_HOUR,
    _report_timestamps,
    report_error,
)


@pytest.fixture(autouse=True)
def _clear_rate_limit():
    """Reset rate limit state between tests."""
    _report_timestamps.clear()
    yield
    _report_timestamps.clear()


@patch("cloud.error_reporter.GITHUB_TOKEN", "ghp_test123")
class TestReportError:
    """Tests for the report_error function."""

    @patch("cloud.error_reporter.requests.post")
    @patch("cloud.error_reporter.requests.get")
    def test_creates_new_issue_when_none_exists(self, mock_get, mock_post):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"items": []}),
        )
        mock_post.return_value = MagicMock(
            status_code=201,
            json=MagicMock(return_value={"number": 42}),
        )

        report_error("Test error: ValueError", "```\ntraceback here\n```")

        mock_post.assert_called_once()
        call_json = mock_post.call_args.kwargs["json"]
        assert call_json["title"] == "Test error: ValueError"
        assert "traceback here" in call_json["body"]
        assert call_json["labels"] == ["bug", "production"]

    @patch("cloud.error_reporter.requests.post")
    @patch("cloud.error_reporter.requests.get")
    def test_adds_comment_when_issue_exists(self, mock_get, mock_post):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={"items": [{"title": "Test error: ValueError", "number": 7}]}
            ),
        )
        mock_post.return_value = MagicMock(status_code=201, json=MagicMock(return_value={}))

        report_error("Test error: ValueError", "new occurrence")

        # Should post a comment, not create an issue
        mock_post.assert_called_once()
        assert "/issues/7/comments" in mock_post.call_args.args[0]

    @patch("cloud.error_reporter.requests.get")
    def test_rate_limiting(self, mock_get):
        # Fill up rate limit
        now = time.monotonic()
        _report_timestamps.extend([now] * _MAX_REPORTS_PER_HOUR)

        report_error("Should be skipped", "body")

        # Should not even search for existing issues
        mock_get.assert_not_called()

    @patch("cloud.error_reporter.GITHUB_TOKEN", "")
    def test_skips_when_no_token(self):
        # Should not raise, just return silently
        report_error("No token", "body")

    @patch("cloud.error_reporter.requests.get")
    def test_handles_api_failure_gracefully(self, mock_get):
        mock_get.side_effect = Exception("Network error")

        # Should not raise
        report_error("API failure test", "body")

    @patch("cloud.error_reporter.requests.post")
    @patch("cloud.error_reporter.requests.get")
    def test_custom_labels(self, mock_get, mock_post):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"items": []}),
        )
        mock_post.return_value = MagicMock(
            status_code=201,
            json=MagicMock(return_value={"number": 1}),
        )

        report_error("Test", "body", labels=["critical"])

        call_json = mock_post.call_args.kwargs["json"]
        assert call_json["labels"] == ["critical"]

    @patch("cloud.error_reporter.requests.post")
    @patch("cloud.error_reporter.requests.get")
    def test_body_includes_hostname_and_timestamp(self, mock_get, mock_post):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"items": []}),
        )
        mock_post.return_value = MagicMock(
            status_code=201,
            json=MagicMock(return_value={"number": 1}),
        )

        report_error("Test", "error details")

        call_json = mock_post.call_args.kwargs["json"]
        assert "**Server**:" in call_json["body"]
        assert "**Time**:" in call_json["body"]
        assert "error details" in call_json["body"]

    @patch("cloud.error_reporter.requests.get")
    def test_ignores_issues_with_different_titles(self, mock_get):
        """Search returns issues but none with exact matching title."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={"items": [{"title": "Different error: TypeError", "number": 5}]}
            ),
        )

        with patch("cloud.error_reporter.requests.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=201,
                json=MagicMock(return_value={"number": 10}),
            )
            report_error("Test error: ValueError", "body")

            # Should create a new issue, not comment
            call_json = mock_post.call_args.kwargs["json"]
            assert call_json["title"] == "Test error: ValueError"
            assert "/comments" not in mock_post.call_args.args[0]
