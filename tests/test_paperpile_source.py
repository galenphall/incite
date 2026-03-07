"""Tests for PaperpileSource (Paperpile BibTeX + Google Drive PDFs)."""

from pathlib import Path

import pytest

from incite.corpus.paperpile_source import (
    PaperpileSource,
    _extract_author_year_title_from_filename,
    _normalize_for_match,
    find_paperpile_pdfs,
    match_paper_to_pdf,
)
from incite.models import Paper

# ---------------------------------------------------------------------------
# Sample BibTeX for testing
# ---------------------------------------------------------------------------

SAMPLE_BIBTEX = """\
@article{Smith2020deep,
  author = {Smith, John and Doe, Jane},
  title = {Deep Learning for Climate Science},
  journal = {Nature Climate Change},
  year = {2020},
  doi = {10.1038/s41558-020-0001-1},
  abstract = {We present a deep learning approach to climate modeling.},
}

@inproceedings{Jones2019survey,
  author = {Jones, Alice and Brown, Bob},
  title = {A Survey of Neural Networks},
  booktitle = {Proceedings of ICML},
  year = {2019},
}

@article{NoTitle2021,
  author = {Nobody, Someone},
  year = {2021},
}
"""


# ---------------------------------------------------------------------------
# Tests for _normalize_for_match
# ---------------------------------------------------------------------------


class TestNormalizeForMatch:
    def test_lowercases(self):
        assert _normalize_for_match("Hello World") == "hello world"

    def test_strips_accents(self):
        assert _normalize_for_match("Müller") == "muller"
        assert _normalize_for_match("García") == "garcia"

    def test_removes_punctuation(self):
        assert _normalize_for_match("Smith, J.") == "smith j"

    def test_collapses_whitespace(self):
        assert _normalize_for_match("  lots   of   space  ") == "lots of space"


# ---------------------------------------------------------------------------
# Tests for _extract_author_year_title_from_filename
# ---------------------------------------------------------------------------


class TestExtractAuthorYearTitle:
    def test_standard_format_with_dash(self):
        author, year, title = _extract_author_year_title_from_filename(
            "Smith 2020 - Deep Learning for NLP.pdf"
        )
        assert author == "smith"
        assert year == "2020"
        assert title.startswith("deep learning")

    def test_format_with_et_al(self):
        author, year, title = _extract_author_year_title_from_filename(
            "Smith et al. 2021 - Some Title.pdf"
        )
        assert author == "smith"
        assert year == "2021"

    def test_format_without_dash(self):
        author, year, title = _extract_author_year_title_from_filename(
            "Jones 2019 A Survey of Methods.pdf"
        )
        assert author == "jones"
        assert year == "2019"

    def test_no_match_returns_title_fallback(self):
        author, year, title = _extract_author_year_title_from_filename(
            "random_file_name.pdf"
        )
        assert author == ""
        assert year == ""
        assert len(title) > 0

    def test_em_dash_separator(self):
        author, year, title = _extract_author_year_title_from_filename(
            "Smith 2020\u2014Deep Learning.pdf"
        )
        assert author == "smith"
        assert year == "2020"


# ---------------------------------------------------------------------------
# Tests for find_paperpile_pdfs
# ---------------------------------------------------------------------------


class TestFindPaperpilePdfs:
    def test_finds_pdfs_in_subfolders(self, tmp_path):
        # Simulate Paperpile's alphabetical folder structure
        (tmp_path / "S").mkdir()
        (tmp_path / "S" / "Smith 2020 - Deep Learning.pdf").touch()
        (tmp_path / "J").mkdir()
        (tmp_path / "J" / "Jones 2019 - A Survey.pdf").touch()

        index = find_paperpile_pdfs(tmp_path)
        assert len(index) == 2

    def test_empty_folder(self, tmp_path):
        index = find_paperpile_pdfs(tmp_path)
        assert index == {}

    def test_nonexistent_folder(self, tmp_path):
        index = find_paperpile_pdfs(tmp_path / "nonexistent")
        assert index == {}

    def test_keys_contain_author_year(self, tmp_path):
        (tmp_path / "Smith 2020 - Title.pdf").touch()
        index = find_paperpile_pdfs(tmp_path)
        assert len(index) == 1
        key = list(index.keys())[0]
        assert "smith" in key
        assert "2020" in key

    def test_files_without_pattern_get_file_prefix(self, tmp_path):
        (tmp_path / "random_name.pdf").touch()
        index = find_paperpile_pdfs(tmp_path)
        assert len(index) == 1
        key = list(index.keys())[0]
        assert key.startswith("file|")


# ---------------------------------------------------------------------------
# Tests for match_paper_to_pdf
# ---------------------------------------------------------------------------


class TestMatchPaperToPdf:
    def _make_index(self, tmp_path):
        """Create a small PDF index for testing."""
        pdf1 = tmp_path / "Smith 2020 - Deep Learning for Climate.pdf"
        pdf2 = tmp_path / "Jones 2019 - A Survey of Methods.pdf"
        pdf1.touch()
        pdf2.touch()
        return find_paperpile_pdfs(tmp_path)

    def test_exact_match(self, tmp_path):
        index = self._make_index(tmp_path)
        paper = Paper(
            id="test1",
            title="Deep Learning for Climate Science",
            authors=["John Smith"],
            year=2020,
        )
        result = match_paper_to_pdf(paper, index)
        assert result is not None
        assert "Smith" in result.name

    def test_author_year_match(self, tmp_path):
        index = self._make_index(tmp_path)
        paper = Paper(
            id="test2",
            title="A Survey of Methods in Deep Learning",
            authors=["Alice Jones"],
            year=2019,
        )
        result = match_paper_to_pdf(paper, index)
        assert result is not None
        assert "Jones" in result.name

    def test_no_match_returns_none(self, tmp_path):
        index = self._make_index(tmp_path)
        paper = Paper(
            id="test3",
            title="Completely Different Paper",
            authors=["Nobody Known"],
            year=2025,
        )
        result = match_paper_to_pdf(paper, index)
        assert result is None

    def test_empty_index(self):
        paper = Paper(id="test4", title="Some Paper", authors=["Smith"], year=2020)
        result = match_paper_to_pdf(paper, {})
        assert result is None

    def test_last_first_author_format(self, tmp_path):
        """Paper with 'Last, First' author format should match."""
        index = self._make_index(tmp_path)
        paper = Paper(
            id="test5",
            title="Deep Learning for Climate",
            authors=["Smith, John"],
            year=2020,
        )
        result = match_paper_to_pdf(paper, index)
        assert result is not None


# ---------------------------------------------------------------------------
# Tests for PaperpileSource.__init__
# ---------------------------------------------------------------------------


class TestPaperpileSourceInit:
    def test_requires_url_or_path(self):
        with pytest.raises(ValueError, match="Must provide either"):
            PaperpileSource()

    def test_accepts_url(self):
        source = PaperpileSource(bibtex_url="https://example.com/library.bib")
        assert source.name == "paperpile"

    def test_accepts_path(self, tmp_path):
        bib = tmp_path / "library.bib"
        bib.write_text(SAMPLE_BIBTEX)
        source = PaperpileSource(bibtex_path=bib)
        assert source.name == "paperpile"

    def test_cache_key(self):
        source = PaperpileSource(bibtex_url="https://example.com/library.bib")
        assert source.cache_key() == "paperpile"


# ---------------------------------------------------------------------------
# Tests for PaperpileSource.load_papers (from local .bib)
# ---------------------------------------------------------------------------


class TestPaperpileSourceLoadPapers:
    def test_loads_from_local_bib(self, tmp_path, monkeypatch):
        bib = tmp_path / "library.bib"
        bib.write_text(SAMPLE_BIBTEX)

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(Path, "home", lambda: cache_dir)

        source = PaperpileSource(bibtex_path=bib)
        papers = source.load_papers()

        # Should parse 2 entries (NoTitle2021 is skipped by BibTeXParser)
        assert len(papers) == 2
        titles = {p.title for p in papers}
        assert "Deep Learning for Climate Science" in titles
        assert "A Survey of Neural Networks" in titles

    def test_preserves_bibtex_key(self, tmp_path, monkeypatch):
        bib = tmp_path / "library.bib"
        bib.write_text(SAMPLE_BIBTEX)

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(Path, "home", lambda: cache_dir)

        source = PaperpileSource(bibtex_path=bib)
        papers = source.load_papers()

        keys = {p.bibtex_key for p in papers}
        assert "Smith2020deep" in keys
        assert "Jones2019survey" in keys

    def test_abstract_from_bibtex(self, tmp_path, monkeypatch):
        bib = tmp_path / "library.bib"
        bib.write_text(SAMPLE_BIBTEX)

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(Path, "home", lambda: cache_dir)

        source = PaperpileSource(bibtex_path=bib)
        papers = source.load_papers()

        smith = [p for p in papers if p.bibtex_key == "Smith2020deep"][0]
        assert "deep learning" in smith.abstract.lower()

    def test_caches_corpus(self, tmp_path, monkeypatch):
        bib = tmp_path / "library.bib"
        bib.write_text(SAMPLE_BIBTEX)

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(Path, "home", lambda: cache_dir)

        source = PaperpileSource(bibtex_path=bib)
        source.load_papers()

        corpus_path = cache_dir / ".incite" / "paperpile_corpus.jsonl"
        assert corpus_path.exists()

    def test_incremental_reuses_cached_papers(self, tmp_path, monkeypatch):
        bib = tmp_path / "library.bib"
        bib.write_text(SAMPLE_BIBTEX)

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(Path, "home", lambda: cache_dir)

        source = PaperpileSource(bibtex_path=bib)
        papers1 = source.load_papers()

        # Second load should reuse cached papers
        papers2 = source.load_papers()
        assert len(papers2) == len(papers1)
        # IDs should be stable
        ids1 = {p.id for p in papers1}
        ids2 = {p.id for p in papers2}
        assert ids1 == ids2


# ---------------------------------------------------------------------------
# Tests for PaperpileSource.needs_refresh
# ---------------------------------------------------------------------------


class TestPaperpileSourceNeedsRefresh:
    def test_true_when_no_cache(self, tmp_path, monkeypatch):
        bib = tmp_path / "library.bib"
        bib.write_text(SAMPLE_BIBTEX)

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(Path, "home", lambda: cache_dir)

        source = PaperpileSource(bibtex_path=bib)
        assert source.needs_refresh() is True

    def test_false_after_load(self, tmp_path, monkeypatch):
        bib = tmp_path / "library.bib"
        bib.write_text(SAMPLE_BIBTEX)

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(Path, "home", lambda: cache_dir)

        source = PaperpileSource(bibtex_path=bib)
        source.load_papers()

        assert source.needs_refresh() is False

    def test_true_when_bib_modified(self, tmp_path, monkeypatch):
        import time

        bib = tmp_path / "library.bib"
        bib.write_text(SAMPLE_BIBTEX)

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(Path, "home", lambda: cache_dir)

        source = PaperpileSource(bibtex_path=bib)
        source.load_papers()

        # Modify bib file
        time.sleep(0.05)
        bib.write_text(SAMPLE_BIBTEX + "\n% modified\n")

        assert source.needs_refresh() is True

    def test_url_source_checks_etag(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(Path, "home", lambda: cache_dir)

        source = PaperpileSource(bibtex_url="https://example.com/lib.bib")

        # No cache file → needs refresh
        assert source.needs_refresh() is True


# ---------------------------------------------------------------------------
# Tests for PDF matching integration
# ---------------------------------------------------------------------------


class TestPdfMatchingIntegration:
    def test_load_papers_with_pdf_folder(self, tmp_path, monkeypatch):
        bib = tmp_path / "library.bib"
        bib.write_text(SAMPLE_BIBTEX)

        # Create matching PDF files
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "Smith 2020 - Deep Learning for Climate Science.pdf").touch()
        (pdf_dir / "Jones 2019 - A Survey of Neural Networks.pdf").touch()

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(Path, "home", lambda: cache_dir)

        source = PaperpileSource(bibtex_path=bib, pdf_folder=pdf_dir)
        papers = source.load_papers()

        matched = [p for p in papers if p.source_file]
        assert len(matched) >= 1  # At least the Smith paper should match

    def test_load_papers_without_pdf_folder(self, tmp_path, monkeypatch):
        bib = tmp_path / "library.bib"
        bib.write_text(SAMPLE_BIBTEX)

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(Path, "home", lambda: cache_dir)

        source = PaperpileSource(bibtex_path=bib)
        papers = source.load_papers()

        # No PDF matching, so no source_file set
        assert all(p.source_file is None for p in papers)
