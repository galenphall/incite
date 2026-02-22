"""Tests for FolderCorpusSource (folder-of-PDFs support)."""

import time
from pathlib import Path
from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz", reason="pymupdf not installed (pip install incite-app[pdf])")

from incite.corpus.folder_source import (
    FolderCorpusSource,
    _clean_filename_title,
    _extract_year_from_creation_date,
    _extract_year_from_filename,
    _paper_id_from_path,
    _scan_folder_pdfs,
    extract_pdf_metadata,
)
from incite.models import Paper

# ---------------------------------------------------------------------------
# Helpers for creating minimal test PDFs via PyMuPDF
# ---------------------------------------------------------------------------


def _create_test_pdf(path: Path, title: str = "Test Paper", text: str = ""):
    """Create a minimal PDF with metadata using PyMuPDF."""
    doc = fitz.open()
    page = doc.new_page()
    # Write some text on the page
    default = f"Content of {title}. Important findings."
    content = text or default
    page.insert_text((72, 72), title, fontsize=18)
    page.insert_text((72, 120), content, fontsize=11)
    # Set metadata
    doc.set_metadata(
        {
            "title": title,
            "author": "Smith, John; Doe, Jane",
            "creationDate": "D:20230615120000",
        }
    )
    doc.save(str(path))
    doc.close()


def _create_pdf_no_metadata(path: Path, body_text: str = ""):
    """Create a PDF with no metadata fields set."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Some Title Text", fontsize=18)
    page.insert_text((72, 120), body_text or "Body paragraph text here.", fontsize=11)
    doc.save(str(path))
    doc.close()


def _create_corrupt_pdf(path: Path):
    """Create a file that looks like a PDF but is actually corrupt."""
    path.write_bytes(b"%PDF-1.4 this is not a real pdf\x00\x00garbage")


# ---------------------------------------------------------------------------
# Tests for _scan_folder_pdfs
# ---------------------------------------------------------------------------


class TestScanFolderPdfs:
    def test_finds_pdfs_in_flat_dir(self, tmp_path):
        (tmp_path / "a.pdf").touch()
        (tmp_path / "b.pdf").touch()
        (tmp_path / "readme.txt").touch()

        pdfs = _scan_folder_pdfs(tmp_path)
        assert len(pdfs) == 2
        names = {p.name for p in pdfs}
        assert names == {"a.pdf", "b.pdf"}

    def test_finds_pdfs_recursively(self, tmp_path):
        sub = tmp_path / "subdir" / "nested"
        sub.mkdir(parents=True)
        (tmp_path / "top.pdf").touch()
        (sub / "deep.pdf").touch()

        pdfs = _scan_folder_pdfs(tmp_path)
        assert len(pdfs) == 2
        names = {p.name for p in pdfs}
        assert names == {"top.pdf", "deep.pdf"}

    def test_empty_folder_returns_empty_list(self, tmp_path):
        pdfs = _scan_folder_pdfs(tmp_path)
        assert pdfs == []

    def test_case_sensitivity(self, tmp_path):
        """Only .pdf (lowercase) should match rglob('*.pdf') on case-sensitive FS."""
        (tmp_path / "lower.pdf").touch()
        (tmp_path / "upper.PDF").touch()

        pdfs = _scan_folder_pdfs(tmp_path)
        # On macOS (case-insensitive HFS+), both match; on Linux, only lower.pdf
        assert len(pdfs) >= 1
        assert any(p.name == "lower.pdf" for p in pdfs)

    def test_results_are_sorted(self, tmp_path):
        (tmp_path / "c.pdf").touch()
        (tmp_path / "a.pdf").touch()
        (tmp_path / "b.pdf").touch()

        pdfs = _scan_folder_pdfs(tmp_path)
        names = [p.name for p in pdfs]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# Tests for helper functions
# ---------------------------------------------------------------------------


class TestCleanFilenameTitle:
    def test_removes_extension(self):
        assert _clean_filename_title("my_paper.pdf") == "my paper"

    def test_replaces_underscores_and_hyphens(self):
        assert _clean_filename_title("some-paper_name.pdf") == "some paper name"

    def test_collapses_whitespace(self):
        assert _clean_filename_title("too   many   spaces.pdf") == "too many spaces"


class TestExtractYearFromCreationDate:
    def test_standard_format(self):
        assert _extract_year_from_creation_date("D:20230615120000") == 2023

    def test_no_date(self):
        assert _extract_year_from_creation_date("") is None
        assert _extract_year_from_creation_date(None) is None

    def test_fallback_year(self):
        assert _extract_year_from_creation_date("2024") == 2024

    def test_invalid_year_rejected(self):
        # Year outside 1900-2100 in D: format won't match
        assert _extract_year_from_creation_date("D:18000101") is None


class TestExtractYearFromFilename:
    def test_year_in_filename(self):
        assert _extract_year_from_filename("smith_2023_climate.pdf") == 2023

    def test_no_year(self):
        assert _extract_year_from_filename("no_year_here.pdf") is None


class TestPaperIdFromPath:
    def test_deterministic(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.touch()
        id1 = _paper_id_from_path(pdf)
        id2 = _paper_id_from_path(pdf)
        assert id1 == id2

    def test_different_paths_differ(self, tmp_path):
        pdf1 = tmp_path / "a.pdf"
        pdf2 = tmp_path / "b.pdf"
        pdf1.touch()
        pdf2.touch()
        assert _paper_id_from_path(pdf1) != _paper_id_from_path(pdf2)

    def test_is_hex_string(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.touch()
        pid = _paper_id_from_path(pdf)
        assert len(pid) == 16
        int(pid, 16)  # Should not raise


# ---------------------------------------------------------------------------
# Tests for extract_pdf_metadata
# ---------------------------------------------------------------------------


class TestExtractPdfMetadata:
    def test_extracts_metadata_fields(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        _create_test_pdf(pdf, title="My Great Paper")

        meta = extract_pdf_metadata(pdf)
        assert meta["title"] == "My Great Paper"
        assert "Smith" in meta["authors"][0] or "John" in meta["authors"][0]
        assert meta["year"] == 2023

    def test_fallback_to_filename_when_no_title(self, tmp_path):
        pdf = tmp_path / "fallback_title_2021.pdf"
        _create_pdf_no_metadata(pdf)

        meta = extract_pdf_metadata(pdf)
        # Should use the largest-font text from page 1 or the filename
        assert meta["title"]  # Not empty
        assert len(meta["title"]) > 0

    def test_year_from_filename_fallback(self, tmp_path):
        pdf = tmp_path / "smith_2019_review.pdf"
        _create_pdf_no_metadata(pdf)

        meta = extract_pdf_metadata(pdf)
        assert meta["year"] == 2019

    def test_author_splitting(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        _create_test_pdf(pdf)

        meta = extract_pdf_metadata(pdf)
        # Authors were set as "Smith, John; Doe, Jane"
        assert len(meta["authors"]) == 2

    def test_handles_missing_fitz_gracefully(self, tmp_path):
        pdf = tmp_path / "test_2022.pdf"
        pdf.touch()

        with patch.dict("sys.modules", {"fitz": None}):
            # When fitz import fails, should fall back to filename
            meta = extract_pdf_metadata(pdf)
            assert meta["title"] == "test 2022"

    def test_corrupt_pdf_returns_filename_fallback(self, tmp_path):
        pdf = tmp_path / "corrupt_2020.pdf"
        _create_corrupt_pdf(pdf)

        meta = extract_pdf_metadata(pdf)
        # Should not crash; falls back to filename
        assert "corrupt" in meta["title"].lower() or meta["title"]
        assert meta["year"] == 2020


# ---------------------------------------------------------------------------
# Tests for FolderCorpusSource
# ---------------------------------------------------------------------------


class TestFolderCorpusSourceInit:
    def test_raises_on_nonexistent_folder(self):
        with pytest.raises(FileNotFoundError, match="Folder not found"):
            FolderCorpusSource("/nonexistent/path/12345")

    def test_raises_on_file_not_dir(self, tmp_path):
        f = tmp_path / "not_a_dir.txt"
        f.touch()
        with pytest.raises(NotADirectoryError, match="Not a directory"):
            FolderCorpusSource(f)

    def test_accepts_valid_directory(self, tmp_path):
        source = FolderCorpusSource(tmp_path)
        assert source.name == "folder"


class TestFolderCorpusSourceCacheKey:
    def test_deterministic(self, tmp_path):
        source = FolderCorpusSource(tmp_path)
        key1 = source.cache_key()
        key2 = source.cache_key()
        assert key1 == key2

    def test_starts_with_folder_prefix(self, tmp_path):
        source = FolderCorpusSource(tmp_path)
        assert source.cache_key().startswith("folder_")

    def test_different_folders_different_keys(self, tmp_path):
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        s1 = FolderCorpusSource(dir1)
        s2 = FolderCorpusSource(dir2)
        assert s1.cache_key() != s2.cache_key()

    def test_key_format(self, tmp_path):
        source = FolderCorpusSource(tmp_path)
        key = source.cache_key()
        # Should be "folder_" + 12 hex chars
        assert len(key) == len("folder_") + 12


class TestFolderCorpusSourceLoadPapers:
    def test_loads_papers_from_pdfs(self, tmp_path, monkeypatch):
        # Create test PDFs
        _create_test_pdf(tmp_path / "paper1.pdf", title="Paper One")
        _create_test_pdf(tmp_path / "paper2.pdf", title="Paper Two")

        # Redirect cache dir to tmp to avoid polluting ~/.incite
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(Path, "home", lambda: cache_dir)

        source = FolderCorpusSource(tmp_path)
        papers = source.load_papers()

        assert len(papers) == 2
        titles = {p.title for p in papers}
        assert "Paper One" in titles
        assert "Paper Two" in titles

    def test_papers_have_required_fields(self, tmp_path, monkeypatch):
        _create_test_pdf(tmp_path / "test.pdf", title="Test Paper")

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(Path, "home", lambda: cache_dir)

        source = FolderCorpusSource(tmp_path)
        papers = source.load_papers()
        paper = papers[0]

        assert isinstance(paper, Paper)
        assert paper.id  # Non-empty
        assert paper.title == "Test Paper"
        assert paper.source_file  # Path to PDF

    def test_papers_have_full_text(self, tmp_path, monkeypatch):
        _create_test_pdf(
            tmp_path / "test.pdf",
            title="Full Text Paper",
            text="This is a long paragraph of text that should be extracted as full text content.",
        )

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(Path, "home", lambda: cache_dir)

        source = FolderCorpusSource(tmp_path)
        papers = source.load_papers()
        paper = papers[0]

        assert paper.full_text is not None
        assert len(paper.full_text) > 0

    def test_empty_folder_raises_error(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(Path, "home", lambda: cache_dir)

        source = FolderCorpusSource(tmp_path)
        with pytest.raises(ValueError, match="No PDF files found"):
            source.load_papers()

    def test_corrupt_pdf_skipped_gracefully(self, tmp_path, monkeypatch):
        _create_test_pdf(tmp_path / "good.pdf", title="Good Paper")
        _create_corrupt_pdf(tmp_path / "bad.pdf")

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(Path, "home", lambda: cache_dir)

        source = FolderCorpusSource(tmp_path)
        papers = source.load_papers()

        # Should have at least the good paper; corrupt one may be skipped
        assert len(papers) >= 1
        assert any(p.title == "Good Paper" for p in papers)

    def test_caching_second_call_loads_from_cache(self, tmp_path, monkeypatch):
        _create_test_pdf(tmp_path / "paper.pdf", title="Cached Paper")

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(Path, "home", lambda: cache_dir)

        source = FolderCorpusSource(tmp_path)

        # First call processes the PDF
        papers1 = source.load_papers()
        assert len(papers1) == 1

        # Verify cache files exist
        cache_path = source._corpus_cache_path()
        meta_path = source._meta_cache_path()
        assert cache_path.exists()
        assert meta_path.exists()

        # Second call should use cache (papers unchanged)
        papers2 = source.load_papers()
        assert len(papers2) == 1
        assert papers2[0].title == papers1[0].title
        assert papers2[0].id == papers1[0].id

    def test_incremental_update_new_pdf(self, tmp_path, monkeypatch):
        _create_test_pdf(tmp_path / "first.pdf", title="First Paper")

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(Path, "home", lambda: cache_dir)

        source = FolderCorpusSource(tmp_path)
        papers1 = source.load_papers()
        assert len(papers1) == 1

        # Add a new PDF
        time.sleep(0.05)  # Ensure mtime differs
        _create_test_pdf(tmp_path / "second.pdf", title="Second Paper")

        papers2 = source.load_papers()
        assert len(papers2) == 2
        titles = {p.title for p in papers2}
        assert "First Paper" in titles
        assert "Second Paper" in titles

    def test_incremental_update_deleted_pdf(self, tmp_path, monkeypatch):
        _create_test_pdf(tmp_path / "keep.pdf", title="Keep Paper")
        _create_test_pdf(tmp_path / "remove.pdf", title="Remove Paper")

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(Path, "home", lambda: cache_dir)

        source = FolderCorpusSource(tmp_path)
        papers1 = source.load_papers()
        assert len(papers1) == 2

        # Delete one PDF
        (tmp_path / "remove.pdf").unlink()

        papers2 = source.load_papers()
        assert len(papers2) == 1
        assert papers2[0].title == "Keep Paper"

    def test_nested_pdfs_included(self, tmp_path, monkeypatch):
        sub = tmp_path / "subdir"
        sub.mkdir()
        _create_test_pdf(tmp_path / "top.pdf", title="Top Paper")
        _create_test_pdf(sub / "nested.pdf", title="Nested Paper")

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(Path, "home", lambda: cache_dir)

        source = FolderCorpusSource(tmp_path)
        papers = source.load_papers()
        assert len(papers) == 2
        titles = {p.title for p in papers}
        assert "Top Paper" in titles
        assert "Nested Paper" in titles


class TestFolderCorpusSourceNeedsRefresh:
    def test_returns_true_when_no_cache(self, tmp_path, monkeypatch):
        (tmp_path / "test.pdf").touch()

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(Path, "home", lambda: cache_dir)

        source = FolderCorpusSource(tmp_path)
        assert source.needs_refresh() is True

    def test_returns_false_after_load(self, tmp_path, monkeypatch):
        _create_test_pdf(tmp_path / "test.pdf", title="Test")

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(Path, "home", lambda: cache_dir)

        source = FolderCorpusSource(tmp_path)
        source.load_papers()

        assert source.needs_refresh() is False

    def test_returns_true_when_new_pdf_added(self, tmp_path, monkeypatch):
        _create_test_pdf(tmp_path / "first.pdf", title="First")

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(Path, "home", lambda: cache_dir)

        source = FolderCorpusSource(tmp_path)
        source.load_papers()

        # Add new PDF with future mtime
        time.sleep(0.05)
        _create_test_pdf(tmp_path / "second.pdf", title="Second")

        assert source.needs_refresh() is True
