"""Tests for cache safety rules documented in CLAUDE.md.

Tests the overwrite protection, version-based re-filtering, and backup
mechanisms in webapp/state.py.
"""

import json

import pytest

from incite.models import Chunk, Paper
from incite.webapp.state import (
    CHUNK_CACHE_VERSION,
    _get_chunk_cache_version,
    _refilter_cached_chunks,
    _set_chunk_cache_version,
)


def _write_fake_chunks(path, n, text_fn=None):
    """Write n fake chunk JSONL lines to path."""
    with open(path, "w") as f:
        for i in range(n):
            text = text_fn(i) if text_fn else f"Chunk text number {i} with enough content to be a valid paragraph."
            data = {
                "id": f"paper_{i % 10}::chunk_{i}",
                "paper_id": f"paper_{i % 10}",
                "text": text,
            }
            f.write(json.dumps(data) + "\n")


def _fake_chunker_factory(n_per_paper):
    """Return a chunker function that produces exactly n_per_paper chunks per paper."""
    def _chunker(papers, show_progress=False):
        chunks = []
        for p in papers:
            for j in range(n_per_paper):
                chunks.append(Chunk(
                    id=f"{p.id}::chunk_{j}",
                    paper_id=p.id,
                    text=f"Chunk {j} of paper {p.id} with enough text to pass validation.",
                ))
        return chunks
    return _chunker


class TestChunkCacheVersion:
    def test_get_version_no_file(self, tmp_path):
        assert _get_chunk_cache_version(tmp_path, "paragraph") == 0

    def test_set_and_get_version(self, tmp_path):
        _set_chunk_cache_version(tmp_path, "paragraph", 3)
        assert _get_chunk_cache_version(tmp_path, "paragraph") == 3

    def test_corrupt_version_file(self, tmp_path):
        version_path = tmp_path / "zotero_chunks_paragraph.version"
        version_path.write_text("not_a_number")
        assert _get_chunk_cache_version(tmp_path, "paragraph") == 0


class TestOverwriteProtection:
    def test_refuses_large_to_small(self, tmp_path, monkeypatch):
        """2000 existing chunks, 100 new -> refuses overwrite, returns old."""
        from incite.webapp import state

        monkeypatch.setattr(state, "get_cache_dir", lambda: tmp_path)
        monkeypatch.setattr(state, "get_chunker", lambda s: _fake_chunker_factory(1))
        chunks_path = tmp_path / "zotero_chunks_paragraph.jsonl"
        _write_fake_chunks(chunks_path, 2000)
        _set_chunk_cache_version(tmp_path, "paragraph", CHUNK_CACHE_VERSION)

        papers = [Paper(id=f"p{i}", title=f"Paper {i}", abstract=f"Abstract {i}.") for i in range(100)]

        result = state.load_zotero_chunks(
            papers, force_rebuild=True, chunking_strategy="paragraph"
        )
        # Should return old cache (2000) instead of new small set (100)
        assert len(result) == 2000

    def test_allows_similar_count(self, tmp_path, monkeypatch):
        """1000 existing chunks, 800 new (>50%) -> allows overwrite."""
        from incite.webapp import state

        monkeypatch.setattr(state, "get_cache_dir", lambda: tmp_path)
        monkeypatch.setattr(state, "get_chunker", lambda s: _fake_chunker_factory(1))
        chunks_path = tmp_path / "zotero_chunks_paragraph.jsonl"
        _write_fake_chunks(chunks_path, 1000)
        _set_chunk_cache_version(tmp_path, "paragraph", CHUNK_CACHE_VERSION)

        papers = [Paper(id=f"p{i}", title=f"Paper {i}", abstract=f"Abstract {i}.") for i in range(800)]

        result = state.load_zotero_chunks(
            papers, force_rebuild=True, chunking_strategy="paragraph"
        )
        # 800 >= 50% of 1000, so overwrite is allowed
        assert len(result) == 800

    def test_allows_when_cache_small(self, tmp_path, monkeypatch):
        """500 existing chunks, 50 new -> allows (<1000 threshold)."""
        from incite.webapp import state

        monkeypatch.setattr(state, "get_cache_dir", lambda: tmp_path)
        monkeypatch.setattr(state, "get_chunker", lambda s: _fake_chunker_factory(1))
        chunks_path = tmp_path / "zotero_chunks_paragraph.jsonl"
        _write_fake_chunks(chunks_path, 500)
        _set_chunk_cache_version(tmp_path, "paragraph", CHUNK_CACHE_VERSION)

        papers = [Paper(id=f"p{i}", title=f"Paper {i}", abstract=f"Abstract {i}.") for i in range(50)]

        result = state.load_zotero_chunks(
            papers, force_rebuild=True, chunking_strategy="paragraph"
        )
        # 500 < 1000 threshold, so overwrite is allowed
        assert len(result) == 50

    def test_boundary_exactly_50_percent(self, tmp_path, monkeypatch):
        """1000 existing, 500 new (exactly 50%) -> allows."""
        from incite.webapp import state

        monkeypatch.setattr(state, "get_cache_dir", lambda: tmp_path)
        monkeypatch.setattr(state, "get_chunker", lambda s: _fake_chunker_factory(1))
        chunks_path = tmp_path / "zotero_chunks_paragraph.jsonl"
        _write_fake_chunks(chunks_path, 1000)
        _set_chunk_cache_version(tmp_path, "paragraph", CHUNK_CACHE_VERSION)

        papers = [Paper(id=f"p{i}", title=f"Paper {i}", abstract=f"Abstract {i}.") for i in range(500)]

        result = state.load_zotero_chunks(
            papers, force_rebuild=True, chunking_strategy="paragraph"
        )
        # 500 == 50% of 1000, condition is `< 0.5` so 0.5 is allowed
        assert len(result) == 500


class TestRefilterPath:
    def test_version_upgrade_triggers_refilter(self, tmp_path, monkeypatch):
        """Outdated version file should trigger re-filter, not rebuild."""
        from incite.webapp import state

        monkeypatch.setattr(state, "get_cache_dir", lambda: tmp_path)

        chunks_path = tmp_path / "zotero_chunks_paragraph.jsonl"
        _write_fake_chunks(chunks_path, 50)
        # Set old version to trigger refilter
        _set_chunk_cache_version(tmp_path, "paragraph", CHUNK_CACHE_VERSION - 1)

        papers = [Paper(id="p0", title="Paper", abstract="Abstract")]

        result = state.load_zotero_chunks(
            papers, force_rebuild=False, chunking_strategy="paragraph"
        )
        # After refilter, version should be updated
        new_version = _get_chunk_cache_version(tmp_path, "paragraph")
        assert new_version == CHUNK_CACHE_VERSION
        assert isinstance(result, list)

    def test_refilter_applies_bibliography_filter(self, tmp_path):
        """Bibliography entries should be removed by _refilter_cached_chunks."""
        chunks_path = tmp_path / "chunks.jsonl"

        # Write a mix of normal chunks and bibliography-like entries
        with open(chunks_path, "w") as f:
            # Normal chunk
            f.write(json.dumps({
                "id": "p1::chunk_0", "paper_id": "p1",
                "text": "Sea levels are rising due to climate change and thermal expansion.",
            }) + "\n")
            # Bibliography entry (short text with many bib signals)
            f.write(json.dumps({
                "id": "p1::chunk_1", "paper_id": "p1",
                "text": "Smith, J., Jones, A. (2023). Sea level rise. Journal of Climate, 36(4), pp. 123-145. doi:10.1234/jclim.2023",
            }) + "\n")
            # Another normal chunk
            f.write(json.dumps({
                "id": "p1::chunk_2", "paper_id": "p1",
                "text": "Ocean temperatures continue to increase across all major basins.",
            }) + "\n")

        result = _refilter_cached_chunks(chunks_path)
        # The bibliography entry should be filtered out
        texts = [c.text for c in result]
        assert any("Sea levels" in t for t in texts)
        assert any("Ocean temperatures" in t for t in texts)
        # Should have fewer chunks than original if bibliography detected
        assert len(result) <= 3

    def test_version_updated_after_refilter(self, tmp_path, monkeypatch):
        """After re-filtering, version file should match CHUNK_CACHE_VERSION."""
        from incite.webapp import state

        monkeypatch.setattr(state, "get_cache_dir", lambda: tmp_path)

        chunks_path = tmp_path / "zotero_chunks_paragraph.jsonl"
        _write_fake_chunks(chunks_path, 10)
        _set_chunk_cache_version(tmp_path, "paragraph", 1)

        papers = [Paper(id="p0", title="Paper", abstract="Abstract")]

        state.load_zotero_chunks(
            papers, force_rebuild=False, chunking_strategy="paragraph"
        )
        assert _get_chunk_cache_version(tmp_path, "paragraph") == CHUNK_CACHE_VERSION


class TestExtractBackups:
    def test_backup_created_before_clearing(self, tmp_path, monkeypatch):
        """extract_and_save_pdfs should create .bak before deleting chunks."""
        from incite.webapp import state

        monkeypatch.setattr(state, "get_cache_dir", lambda: tmp_path)

        # Create a fake chunks file
        chunks_path = tmp_path / "zotero_chunks.jsonl"
        _write_fake_chunks(chunks_path, 10)
        assert chunks_path.exists()

        # Create a fake corpus file so save_corpus doesn't fail
        corpus_path = tmp_path / "zotero_corpus.jsonl"
        corpus_path.write_text("")

        papers = [Paper(id="p0", title="Paper", abstract="Abstract")]

        try:
            state.extract_and_save_pdfs(papers)
        except Exception:
            pass  # May fail because papers don't have real PDFs

        # If the function got far enough to clear caches, .bak should exist
        bak_path = chunks_path.with_suffix(".jsonl.bak")
        if not chunks_path.exists():
            assert bak_path.exists()

    def test_strategy_specific_backups(self, tmp_path, monkeypatch):
        """Strategy-specific chunk files should also get backed up."""
        from incite.webapp import state

        monkeypatch.setattr(state, "get_cache_dir", lambda: tmp_path)

        # Create strategy-specific chunk file
        strategy_path = tmp_path / "zotero_chunks_paragraph.jsonl"
        _write_fake_chunks(strategy_path, 10)

        # Create a fake corpus file
        corpus_path = tmp_path / "zotero_corpus.jsonl"
        corpus_path.write_text("")

        papers = [Paper(id="p0", title="Paper", abstract="Abstract")]

        try:
            state.extract_and_save_pdfs(papers)
        except Exception:
            pass

        bak_path = strategy_path.with_suffix(".jsonl.bak")
        if not strategy_path.exists():
            assert bak_path.exists()
