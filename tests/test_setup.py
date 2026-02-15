"""Tests for the incite setup wizard."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from incite.cli.setup import (
    _build_index,
    _load_papers,
    _save_setup_config,
    _setup_folder,
    _setup_zotero_noninteractive,
    cmd_setup,
)
from incite.models import Paper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_papers(n=5):
    """Create a list of minimal Paper objects for testing."""
    return [
        Paper(
            id=f"paper_{i}",
            title=f"Test Paper {i}",
            abstract=f"Abstract for paper {i}." if i % 2 == 0 else "",
            authors=[f"Author {i}"],
            year=2020 + i,
            source_file=f"/fake/path/paper_{i}.pdf" if i % 3 == 0 else None,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# _setup_folder
# ---------------------------------------------------------------------------


class TestSetupFolder:
    def test_valid_folder_with_pdfs(self, tmp_path):
        """Folder with PDFs returns correct config."""
        (tmp_path / "paper1.pdf").write_bytes(b"%PDF-1.4 fake")
        (tmp_path / "paper2.pdf").write_bytes(b"%PDF-1.4 fake")

        result = _setup_folder(str(tmp_path))
        assert result is not None
        assert result["type"] == "folder"
        assert result["path"] == str(tmp_path)

    def test_nonexistent_folder(self, capsys):
        result = _setup_folder("/nonexistent/path/to/nowhere")
        assert result is None
        assert "not found" in capsys.readouterr().out.lower()

    def test_not_a_directory(self, tmp_path, capsys):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        result = _setup_folder(str(f))
        assert result is None
        assert "not a directory" in capsys.readouterr().out.lower()

    def test_empty_folder(self, tmp_path, capsys):
        result = _setup_folder(str(tmp_path))
        assert result is None
        assert "no pdf" in capsys.readouterr().out.lower()

    def test_tilde_expansion(self, tmp_path, monkeypatch):
        """Folder path with ~ is expanded."""
        (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setenv("HOME", str(tmp_path.parent))
        result = _setup_folder(str(tmp_path))
        assert result is not None
        # Path should be resolved (absolute, no ~)
        assert "~" not in result["path"]


# ---------------------------------------------------------------------------
# _setup_zotero_noninteractive
# ---------------------------------------------------------------------------


class TestSetupZoteroNoninteractive:
    @patch("incite.corpus.zotero_reader.find_zotero_data_dir")
    def test_zotero_found(self, mock_find):
        mock_find.return_value = Path("/Users/test/Zotero")
        result = _setup_zotero_noninteractive()
        assert result == {"type": "zotero", "path": "/Users/test/Zotero"}

    @patch("incite.corpus.zotero_reader.find_zotero_data_dir")
    def test_zotero_not_found(self, mock_find, capsys):
        mock_find.return_value = None
        result = _setup_zotero_noninteractive()
        assert result is None
        assert "could not auto-detect" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# _load_papers
# ---------------------------------------------------------------------------


class TestLoadPapers:
    @patch("incite.webapp.state.load_zotero_direct")
    def test_load_zotero(self, mock_load):
        papers = _make_papers(3)
        mock_load.return_value = papers
        result = _load_papers({"type": "zotero", "path": "/Users/test/Zotero"})
        assert result == papers
        mock_load.assert_called_once()

    @patch("incite.corpus.folder_source.FolderCorpusSource")
    def test_load_folder(self, mock_cls):
        papers = _make_papers(2)
        mock_instance = MagicMock()
        mock_instance.load_papers.return_value = papers
        mock_cls.return_value = mock_instance
        result = _load_papers({"type": "folder", "path": "/tmp/pdfs"})
        assert result == papers

    @patch("incite.webapp.state.load_zotero_direct")
    def test_load_error(self, mock_load, capsys):
        mock_load.side_effect = FileNotFoundError("db not found")
        result = _load_papers({"type": "zotero", "path": "/bad/path"})
        assert result is None
        assert "db not found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _build_index
# ---------------------------------------------------------------------------


class TestBuildIndex:
    @patch("incite.webapp.state.get_retriever")
    @patch("incite.webapp.state.get_cache_dir")
    def test_skips_when_fresh(self, mock_cache, mock_get, tmp_path):
        """Skips rebuild when index exists with matching paper count."""
        index_dir = tmp_path / "zotero_index_minilm"
        index_dir.mkdir()
        (index_dir / "index.faiss").write_bytes(b"fake")

        papers = _make_papers(3)
        id_map = {"id_to_idx": {p.id: i for i, p in enumerate(papers)}}
        (index_dir / "id_map.json").write_text(json.dumps(id_map))

        mock_cache.return_value = tmp_path
        mock_retriever = MagicMock()
        mock_get.return_value = mock_retriever

        result = _build_index(papers, "minilm")
        assert result is mock_retriever

    @patch("incite.webapp.state.get_retriever")
    @patch("incite.webapp.state.get_cache_dir")
    def test_rebuilds_when_count_mismatch(self, mock_cache, mock_get, tmp_path):
        """Rebuilds when paper count doesn't match cached index."""
        index_dir = tmp_path / "zotero_index_minilm"
        index_dir.mkdir()
        (index_dir / "index.faiss").write_bytes(b"fake")
        # Cached with 2 papers but we have 5
        id_map = {"id_to_idx": {"a": 0, "b": 1}}
        (index_dir / "id_map.json").write_text(json.dumps(id_map))

        mock_cache.return_value = tmp_path
        mock_retriever = MagicMock()
        mock_get.return_value = mock_retriever

        result = _build_index(_make_papers(5), "minilm")
        assert result is mock_retriever
        # Should call get_retriever with progress callback (forces rebuild internally)
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs.get("progress_callback") is not None

    @patch("incite.webapp.state.get_retriever")
    @patch("incite.webapp.state.get_cache_dir")
    def test_builds_when_no_index(self, mock_cache, mock_get, tmp_path):
        """Builds when no index exists."""
        mock_cache.return_value = tmp_path
        mock_retriever = MagicMock()
        mock_get.return_value = mock_retriever

        result = _build_index(_make_papers(3), "minilm")
        assert result is mock_retriever

    @patch("incite.webapp.state.get_retriever")
    @patch("incite.webapp.state.get_cache_dir")
    def test_handles_build_error(self, mock_cache, mock_get, tmp_path, capsys):
        mock_cache.return_value = tmp_path
        mock_get.side_effect = RuntimeError("model download failed")

        result = _build_index(_make_papers(3), "minilm")
        assert result is None
        assert "error" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# _save_setup_config
# ---------------------------------------------------------------------------


class TestSaveConfig:
    @patch("incite.webapp.state.save_config")
    @patch("incite.webapp.state.get_config")
    def test_merges_with_existing(self, mock_get, mock_save):
        mock_get.return_value = {"webapp": {"default_k": 5}, "zotero": {"data_dir": ""}}

        config = {
            "source": {"type": "zotero", "path": "/Users/test/Zotero"},
            "embedder": "minilm",
            "method": "hybrid",
            "processing": {"mode": "local"},
        }
        _save_setup_config(config)

        saved = mock_save.call_args[0][0]
        # New keys present
        assert saved["source"] == config["source"]
        assert saved["embedder"] == "minilm"
        # Old keys preserved
        assert saved["webapp"] == {"default_k": 5}


# ---------------------------------------------------------------------------
# Full cmd_setup (integration-style with mocks)
# ---------------------------------------------------------------------------


class TestCmdSetup:
    @patch("incite.cli.setup._save_setup_config")
    @patch("incite.cli.setup._test_query")
    @patch("incite.cli.setup._build_index")
    @patch("incite.cli.setup._load_papers")
    @patch("incite.corpus.zotero_reader.find_zotero_data_dir")
    def test_noninteractive_zotero(
        self, mock_find, mock_load, mock_build, mock_test, mock_save
    ):
        mock_find.return_value = Path("/Users/test/Zotero")
        papers = _make_papers(10)
        mock_load.return_value = papers
        mock_build.return_value = MagicMock()  # retriever

        args = MagicMock()
        args.zotero = True
        args.folder = None
        args.embedder = "minilm"

        cmd_setup(args)

        mock_load.assert_called_once()
        mock_build.assert_called_once_with(papers, "minilm")
        mock_save.assert_called_once()
        saved_config = mock_save.call_args[0][0]
        assert saved_config["source"]["type"] == "zotero"

    @patch("incite.cli.setup._save_setup_config")
    @patch("incite.cli.setup._test_query")
    @patch("incite.cli.setup._build_index")
    @patch("incite.cli.setup._load_papers")
    def test_noninteractive_folder(
        self, mock_load, mock_build, mock_test, mock_save, tmp_path
    ):
        (tmp_path / "paper.pdf").write_bytes(b"%PDF fake")
        papers = _make_papers(3)
        mock_load.return_value = papers
        mock_build.return_value = MagicMock()

        args = MagicMock()
        args.zotero = False
        args.folder = str(tmp_path)
        args.embedder = "minilm"

        cmd_setup(args)

        mock_load.assert_called_once()
        saved_config = mock_save.call_args[0][0]
        assert saved_config["source"]["type"] == "folder"

    @patch("incite.cli.setup._load_papers")
    @patch("incite.corpus.zotero_reader.find_zotero_data_dir")
    def test_exits_on_no_zotero(self, mock_find, mock_load):
        """--zotero flag with no Zotero installed exits with error."""
        mock_find.return_value = None

        args = MagicMock()
        args.zotero = True
        args.folder = None
        args.embedder = "minilm"

        with pytest.raises(SystemExit):
            cmd_setup(args)

        mock_load.assert_not_called()

    @patch("incite.cli.setup._build_index")
    @patch("incite.cli.setup._load_papers")
    def test_exits_on_load_failure(self, mock_load, mock_build, tmp_path):
        """Exits when paper loading fails."""
        (tmp_path / "paper.pdf").write_bytes(b"%PDF fake")
        mock_load.return_value = None

        args = MagicMock()
        args.zotero = False
        args.folder = str(tmp_path)
        args.embedder = "minilm"

        with pytest.raises(SystemExit):
            cmd_setup(args)

        mock_build.assert_not_called()

    @patch("incite.cli.setup._load_papers")
    @patch("incite.cli.setup._build_index")
    def test_exits_on_build_failure(self, mock_build, mock_load, tmp_path):
        """Exits when index building fails."""
        (tmp_path / "paper.pdf").write_bytes(b"%PDF fake")
        mock_load.return_value = _make_papers(3)
        mock_build.return_value = None

        args = MagicMock()
        args.zotero = False
        args.folder = str(tmp_path)
        args.embedder = "minilm"

        with pytest.raises(SystemExit):
            cmd_setup(args)
