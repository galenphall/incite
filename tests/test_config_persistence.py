"""Tests for config persistence in serve and setup commands."""

from unittest.mock import patch

from incite.cli.serve import _config_default
from incite.cli.setup import _user_friendly_error

# ---------------------------------------------------------------------------
# _config_default
# ---------------------------------------------------------------------------


class TestConfigDefault:
    @patch("incite.webapp.state.get_config")
    def test_reads_from_config(self, mock_config):
        mock_config.return_value = {"embedder": "minilm-ft", "method": "hybrid", "port": 8230}
        assert _config_default("embedder", "minilm") == "minilm-ft"
        assert _config_default("method", "hybrid") == "hybrid"
        assert _config_default("port", 8230) == 8230

    @patch("incite.webapp.state.get_config")
    def test_falls_back_on_missing_key(self, mock_config):
        mock_config.return_value = {}
        assert _config_default("embedder", "minilm") == "minilm"
        assert _config_default("port", 8230) == 8230

    @patch("incite.webapp.state.get_config")
    def test_validates_embedder_choice(self, mock_config):
        mock_config.return_value = {"embedder": "invalid-model"}
        assert _config_default("embedder", "minilm") == "minilm"

    @patch("incite.webapp.state.get_config")
    def test_handles_exception(self, mock_config):
        mock_config.side_effect = Exception("config broken")
        assert _config_default("embedder", "minilm") == "minilm"


# ---------------------------------------------------------------------------
# _user_friendly_error
# ---------------------------------------------------------------------------


class TestUserFriendlyError:
    def test_sqlite_locked(self):
        import sqlite3

        exc = sqlite3.OperationalError("database is locked")
        msg = _user_friendly_error("load papers", exc)
        assert "locked" in msg.lower()
        assert "close zotero" in msg.lower()

    def test_permission_error(self):
        exc = PermissionError("not allowed")
        msg = _user_friendly_error("load papers", exc)
        assert "permission" in msg.lower()

    def test_file_not_found(self):
        exc = FileNotFoundError("zotero.sqlite")
        msg = _user_friendly_error("load papers", exc)
        assert "not found" in msg.lower()

    def test_model_download_error(self):
        exc = OSError("Error downloading model from huggingface")
        msg = _user_friendly_error("build index", exc)
        assert "internet" in msg.lower()

    def test_connection_error(self):
        exc = ConnectionError("refused")
        msg = _user_friendly_error("connect", exc)
        assert "network" in msg.lower()

    def test_auto_detect_error(self):
        exc = ValueError("Could not auto-detect Zotero directory")
        msg = _user_friendly_error("setup", exc)
        assert "setup" in msg.lower()

    def test_generic_error(self):
        exc = RuntimeError("something unknown")
        msg = _user_friendly_error("do thing", exc)
        assert msg == "something unknown"
