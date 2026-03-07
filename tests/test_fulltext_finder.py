"""Tests for cloud/fulltext_finder.py — all mocked, no real API/DB calls."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

# ---------------------------------------------------------------------------
# TestDiscoverOaPdfUrl
# ---------------------------------------------------------------------------


class TestDiscoverOaPdfUrl:
    @patch("cloud.fulltext_finder.time.sleep")
    @patch("cloud.fulltext_finder.requests.get")
    def test_s2_oa_pdf_found_by_doi(self, mock_get: MagicMock, mock_sleep: MagicMock) -> None:
        """S2 returns openAccessPdf by DOI — no further sources queried."""
        from cloud.fulltext_finder import _discover_oa_pdf_url

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "openAccessPdf": {"url": "https://arxiv.org/pdf/2301.00001.pdf"},
            },
        )

        result = _discover_oa_pdf_url(doi="10.1234/test", title="Some Paper")

        assert result == "https://arxiv.org/pdf/2301.00001.pdf"
        # Only one S2 call made (DOI lookup), no Unpaywall
        mock_get.assert_called_once()
        assert "DOI:10.1234/test" in mock_get.call_args[0][0]

    @patch("cloud.fulltext_finder.time.sleep")
    @patch("cloud.fulltext_finder.UnpaywallClient")
    @patch("cloud.fulltext_finder.requests.get")
    @patch.dict("os.environ", {"OPENALEX_EMAIL": "test@example.com"})
    def test_s2_no_oa_falls_through_to_unpaywall(
        self, mock_get: MagicMock, mock_unpaywall_cls: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """S2 has no OA PDF — falls through to Unpaywall which finds it."""
        from cloud.fulltext_finder import _discover_oa_pdf_url

        # S2 DOI lookup: no openAccessPdf
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"openAccessPdf": None},
        )

        # Unpaywall finds the PDF
        mock_client = MagicMock()
        mock_unpaywall_cls.return_value = mock_client
        mock_client.lookup.return_value = MagicMock(
            best_oa_url="https://europepmc.org/pdf/12345.pdf"
        )

        result = _discover_oa_pdf_url(doi="10.1234/closed", title="Closed Paper")

        assert result == "https://europepmc.org/pdf/12345.pdf"
        mock_client.lookup.assert_called_once_with("10.1234/closed")

    @patch("cloud.fulltext_finder.time.sleep")
    @patch("cloud.fulltext_finder.requests.get")
    def test_no_doi_uses_s2_title_search(self, mock_get: MagicMock, mock_sleep: MagicMock) -> None:
        """No DOI — uses S2 title search instead of DOI lookup."""
        from cloud.fulltext_finder import _discover_oa_pdf_url

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "data": [{"openAccessPdf": {"url": "https://arxiv.org/pdf/title-hit.pdf"}}]
            },
        )

        result = _discover_oa_pdf_url(doi=None, title="Deep Learning Survey")

        assert result == "https://arxiv.org/pdf/title-hit.pdf"
        # Should hit the search endpoint, not DOI endpoint
        assert "search" in mock_get.call_args[0][0]

    @patch("cloud.fulltext_finder.time.sleep")
    @patch("cloud.fulltext_finder.UnpaywallClient")
    @patch("cloud.fulltext_finder.requests.get")
    @patch.dict("os.environ", {"OPENALEX_EMAIL": "test@example.com"})
    def test_both_sources_fail_returns_none(
        self, mock_get: MagicMock, mock_unpaywall_cls: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Both S2 and Unpaywall fail — returns None."""
        from cloud.fulltext_finder import _discover_oa_pdf_url

        # S2 returns no OA
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"openAccessPdf": None},
        )

        # Unpaywall returns None
        mock_client = MagicMock()
        mock_unpaywall_cls.return_value = mock_client
        mock_client.lookup.return_value = None

        result = _discover_oa_pdf_url(doi="10.1234/nope", title="No Access Paper")

        assert result is None


# ---------------------------------------------------------------------------
# TestDownloadPdf
# ---------------------------------------------------------------------------


class TestDownloadPdf:
    @patch("cloud.fulltext_finder.requests.get")
    def test_download_valid_pdf(self, mock_get: MagicMock, tmp_path: Path) -> None:
        """Downloads and validates a PDF file."""
        from cloud.fulltext_finder import _download_pdf

        pdf_content = b"%PDF-1.4 fake pdf content here"
        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": str(len(pdf_content))}
        mock_resp.iter_content = MagicMock(return_value=[pdf_content])
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_get.return_value = mock_resp

        dest = tmp_path / "paper.pdf"
        ok = _download_pdf("https://arxiv.org/pdf/2301.00001.pdf", dest)

        assert ok is True
        assert dest.exists()
        assert dest.read_bytes() == pdf_content

    @patch("cloud.fulltext_finder.requests.get")
    def test_download_invalid_header(self, mock_get: MagicMock, tmp_path: Path) -> None:
        """Rejects non-PDF content (invalid header)."""
        from cloud.fulltext_finder import _download_pdf

        html_content = b"<html>Not a PDF</html>"
        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": str(len(html_content))}
        mock_resp.iter_content = MagicMock(return_value=[html_content])
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_get.return_value = mock_resp

        dest = tmp_path / "bad.pdf"
        ok = _download_pdf("https://example.com/fake.pdf", dest)

        assert ok is False
        assert not dest.exists()

    @patch("cloud.fulltext_finder.requests.get")
    def test_download_multi_chunk_pdf(self, mock_get: MagicMock, tmp_path: Path) -> None:
        """Multi-chunk PDF: header check only on first chunk, second chunk is valid body."""
        from cloud.fulltext_finder import _download_pdf

        chunk1 = b"%PDF-1.4 header"
        chunk2 = b"\x00\x01\x02 binary body content"
        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": str(len(chunk1) + len(chunk2))}
        mock_resp.iter_content = MagicMock(return_value=[chunk1, chunk2])
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_get.return_value = mock_resp

        dest = tmp_path / "multi.pdf"
        ok = _download_pdf("https://example.com/multi.pdf", dest)

        assert ok is True
        assert dest.exists()
        assert dest.read_bytes() == chunk1 + chunk2

    @patch("cloud.fulltext_finder.requests.get")
    def test_download_timeout(self, mock_get: MagicMock, tmp_path: Path) -> None:
        """Handles timeout gracefully."""
        from cloud.fulltext_finder import _download_pdf

        mock_get.side_effect = requests.Timeout("timed out")

        dest = tmp_path / "timeout.pdf"
        ok = _download_pdf("https://example.com/slow.pdf", dest)

        assert ok is False
        assert not dest.exists()


# ---------------------------------------------------------------------------
# TestFindAndDownloadFulltext
# ---------------------------------------------------------------------------


class TestFindAndDownloadFulltext:
    @patch("cloud.fulltext_finder._discover_oa_pdf_url")
    @patch("cloud.fulltext_finder._download_pdf")
    @patch("cloud.fulltext_finder._get_papers_without_body_vectors")
    def test_skips_papers_with_existing_chunks(
        self, mock_get_missing: MagicMock, mock_download: MagicMock, mock_discover: MagicMock
    ) -> None:
        """Papers already having body_vectors are skipped entirely."""
        from cloud.fulltext_finder import find_and_download_fulltext

        papers = [
            {"canonical_id": "aaa", "doi": "10.1/a", "title": "Paper A"},
            {"canonical_id": "bbb", "doi": "10.1/b", "title": "Paper B"},
        ]
        # Only paper bbb needs processing (aaa already has chunks)
        mock_get_missing.return_value = [papers[1]]

        # Discover returns None for bbb (no PDF found)
        mock_discover.return_value = None

        user_dir = Path("/tmp/fake_user_dir")
        db = MagicMock()

        result = find_and_download_fulltext(papers, library_id=1, user_dir=user_dir, db=db)

        assert result == {}
        # Discover was called only for bbb, not aaa
        mock_discover.assert_called_once_with("10.1/b", "Paper B")

    @patch("cloud.fulltext_finder.time.sleep")
    @patch("cloud.fulltext_finder._discover_oa_pdf_url")
    @patch("cloud.fulltext_finder._download_pdf")
    @patch("cloud.fulltext_finder._get_papers_without_body_vectors")
    def test_end_to_end_one_paper(
        self,
        mock_get_missing: MagicMock,
        mock_download: MagicMock,
        mock_discover: MagicMock,
        mock_sleep: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Full flow: filter → discover → download → return path."""
        from cloud.fulltext_finder import find_and_download_fulltext

        papers = [{"canonical_id": "ccc", "doi": "10.1/c", "title": "Paper C"}]
        mock_get_missing.return_value = papers

        mock_discover.return_value = "https://arxiv.org/pdf/paper-c.pdf"
        mock_download.return_value = True

        user_dir = tmp_path / "user"
        user_dir.mkdir()
        db = MagicMock()

        result = find_and_download_fulltext(papers, library_id=1, user_dir=user_dir, db=db)

        assert "ccc" in result
        assert result["ccc"].name.endswith(".pdf")
        mock_download.assert_called_once()


# ---------------------------------------------------------------------------
# TestRunFulltextPipeline
# ---------------------------------------------------------------------------


class TestRunFulltextPipeline:
    @patch("cloud.library_worker._index_new_papers")
    @patch("cloud.library_worker._run_modal_grobid_extraction")
    @patch("cloud.fulltext_finder.find_and_download_fulltext")
    def test_pipeline_calls_grobid_and_index(
        self,
        mock_find: MagicMock,
        mock_grobid: MagicMock,
        mock_index: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Pipeline calls GROBID extraction and indexing after downloading PDFs."""
        from cloud.fulltext_finder import run_fulltext_pipeline

        pdf_path = tmp_path / "pdfs" / "abc123.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        mock_find.return_value = {"abc123": pdf_path}
        mock_grobid.return_value = {"abc123"}
        mock_index.return_value = (1, 5, {})

        db = MagicMock()
        papers = [
            {
                "canonical_id": "abc123",
                "title": "Test Paper",
                "abstract": "An abstract",
                "authors": ["Smith"],
                "year": 2023,
                "journal": "Nature",
            }
        ]
        paper_map = {"abc123": "zotero_key_1"}

        run_fulltext_pipeline(
            library_id=1,
            user_dir=tmp_path,
            db=db,
            papers=papers,
            paper_map=paper_map,
            embedder_type="granite-ft",
            job_id="job-001",
        )

        mock_find.assert_called_once_with(papers, 1, tmp_path, db)
        mock_grobid.assert_called_once_with(tmp_path, {"abc123": pdf_path}, db, "job-001")
        mock_index.assert_called_once()
        # Verify key kwargs passed to _index_new_papers
        call_kwargs = mock_index.call_args[1]
        assert call_kwargs["library_id"] == 1
        assert call_kwargs["embedder_type"] == "granite-ft"
        assert call_kwargs["extracted_ids"] == {"abc123"}
        assert call_kwargs["cache_dir"] == tmp_path / "grobid_cache"

    @patch("cloud.library_worker._index_new_papers")
    @patch("cloud.library_worker._run_modal_grobid_extraction")
    @patch("cloud.fulltext_finder.find_and_download_fulltext")
    def test_pipeline_noop_when_no_pdfs_found(
        self,
        mock_find: MagicMock,
        mock_grobid: MagicMock,
        mock_index: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Pipeline short-circuits when no OA PDFs are discovered."""
        from cloud.fulltext_finder import run_fulltext_pipeline

        mock_find.return_value = {}

        db = MagicMock()
        papers = [{"canonical_id": "xyz", "title": "No PDF", "authors": [], "year": 2020}]

        run_fulltext_pipeline(
            library_id=1,
            user_dir=tmp_path,
            db=db,
            papers=papers,
            paper_map={},
            embedder_type="granite-ft",
            job_id="job-002",
        )

        mock_find.assert_called_once()
        mock_grobid.assert_not_called()
        mock_index.assert_not_called()
