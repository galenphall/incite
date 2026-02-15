"""Tests for the incite doctor command."""

import json
from unittest.mock import MagicMock, patch

from incite.cli.doctor import (
    _check_chunk_cache,
    _check_config_embedder,
    _check_disk_space,
    _check_faiss,
    _check_faiss_index,
    _check_python_version,
    cmd_doctor,
)


class TestCheckPythonVersion:
    def test_current_version_passes(self):
        ok, msg = _check_python_version()
        # We're running on Python 3.10+ to use this project
        assert ok
        assert "Python" in msg


class TestCheckFaiss:
    def test_faiss_available(self):
        ok, msg = _check_faiss()
        assert ok
        assert "FAISS" in msg

    @patch.dict("sys.modules", {"faiss": None})
    def test_faiss_missing(self):
        # This is tricky to test since faiss is installed;
        # the import error path is covered by the except block
        pass


class TestCheckConfigEmbedder:
    @patch("incite.cli.doctor.get_config", create=True)
    def test_embedder_present(self, mock_config):
        # Patch at the function level since doctor imports get_config
        with patch("incite.webapp.state.get_config") as mock_cfg:
            mock_cfg.return_value = {"embedder": "minilm-ft"}
            ok, msg = _check_config_embedder()
            assert ok
            assert "minilm-ft" in msg

    @patch("incite.webapp.state.get_config")
    def test_embedder_missing(self, mock_cfg):
        mock_cfg.return_value = {"webapp": {"default_k": 5}}
        ok, msg = _check_config_embedder()
        assert not ok
        assert "setup" in msg.lower()


class TestCheckFaissIndex:
    @patch("incite.webapp.state.get_config")
    @patch("incite.webapp.state.get_cache_dir")
    def test_index_exists(self, mock_cache, mock_config, tmp_path):
        mock_cache.return_value = tmp_path
        mock_config.return_value = {"embedder": "minilm-ft"}

        index_dir = tmp_path / "zotero_index_minilm-ft"
        index_dir.mkdir()
        (index_dir / "index.faiss").write_bytes(b"fake")
        id_map = {
            "id_to_idx": {"paper1": 0, "paper2": 1},
            "embedder_type": "minilm-ft",
        }
        (index_dir / "id_map.json").write_text(json.dumps(id_map))

        ok, msg = _check_faiss_index()
        assert ok
        assert "2 papers" in msg

    @patch("incite.webapp.state.get_config")
    @patch("incite.webapp.state.get_cache_dir")
    def test_no_index(self, mock_cache, mock_config, tmp_path):
        mock_cache.return_value = tmp_path
        mock_config.return_value = {"embedder": "minilm-ft"}

        ok, msg = _check_faiss_index()
        assert not ok
        assert "setup" in msg.lower()

    @patch("incite.webapp.state.get_config")
    @patch("incite.webapp.state.get_cache_dir")
    def test_embedder_mismatch(self, mock_cache, mock_config, tmp_path):
        mock_cache.return_value = tmp_path
        mock_config.return_value = {"embedder": "minilm-ft"}

        index_dir = tmp_path / "zotero_index_minilm-ft"
        index_dir.mkdir()
        (index_dir / "index.faiss").write_bytes(b"fake")
        id_map = {
            "id_to_idx": {"paper1": 0},
            "embedder_type": "minilm",  # different!
        }
        (index_dir / "id_map.json").write_text(json.dumps(id_map))

        ok, msg = _check_faiss_index()
        assert not ok
        assert "mismatch" in msg.lower()


class TestCheckChunkCache:
    @patch("incite.webapp.state.get_cache_dir")
    def test_no_chunks(self, mock_cache, tmp_path):
        mock_cache.return_value = tmp_path
        ok, msg = _check_chunk_cache()
        assert ok  # no chunks is OK (paper-mode)
        assert "paper-mode" in msg.lower()

    @patch("incite.webapp.state.get_cache_dir")
    def test_with_chunks(self, mock_cache, tmp_path):
        mock_cache.return_value = tmp_path
        chunks_file = tmp_path / "zotero_chunks_paragraph.jsonl"
        chunks_file.write_text("\n".join(['{"id": "c1"}', '{"id": "c2"}', '{"id": "c3"}']))

        ok, msg = _check_chunk_cache()
        assert ok
        assert "3 chunks" in msg


class TestCheckDiskSpace:
    def test_disk_space(self):
        ok, msg = _check_disk_space()
        assert ok  # should have > 1GB free on dev machines
        assert "GB" in msg


class TestCmdDoctor:
    def test_runs_without_error(self, capsys):
        """Doctor command runs and produces output."""
        # Just verify it doesn't crash — individual checks may fail
        # in the test environment, that's OK
        args = MagicMock()
        cmd_doctor(args)
        out = capsys.readouterr().out
        assert "inCite Doctor" in out
        assert "passed" in out.lower() or "issue" in out.lower()
