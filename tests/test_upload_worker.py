"""Tests for cloud/upload_worker.py: BibTeX dedup and enrichment."""

from unittest.mock import MagicMock, patch

from incite.models import Paper


class TestEnrichPapers:
    """Test _enrich_papers best-effort enrichment."""

    def test_skips_papers_with_abstracts(self):
        """Papers that already have abstracts should not be enriched."""
        from cloud.upload_worker import _enrich_papers

        papers = [
            Paper(id="p1", title="Paper One", abstract="Has an abstract"),
        ]
        db = MagicMock()
        count = _enrich_papers(papers, db, "job1")
        assert count == 0

    def test_skips_when_no_api_keys(self):
        """Without API keys, enrichment returns 0."""
        from cloud.upload_worker import _enrich_papers

        papers = [
            Paper(id="p1", title="Paper One", abstract=""),
        ]
        db = MagicMock()
        with patch.dict("os.environ", {}, clear=True):
            count = _enrich_papers(papers, db, "job1")
        assert count == 0

    @patch("cloud.upload_worker._enrich_papers.__module__", "cloud.upload_worker")
    def test_enriches_via_doi(self):
        """Papers with DOI should be enriched via enrich_from_doi."""
        from cloud.upload_worker import _enrich_papers

        papers = [
            Paper(id="p1", title="Paper One", abstract="", doi="10.1234/test"),
        ]

        mock_result = Paper(
            id="enriched",
            title="Paper One",
            abstract="Enriched abstract from S2",
            year=2023,
        )

        db = MagicMock()
        with (
            patch.dict("os.environ", {"SEMANTIC_SCHOLAR_API_KEY": "fake_key"}),
            patch(
                "incite.corpus.enrichment.MetadataEnricher.enrich_from_doi",
                return_value=mock_result,
            ),
        ):
            count = _enrich_papers(papers, db, "job1")

        assert count == 1
        assert papers[0].abstract == "Enriched abstract from S2"
        assert papers[0].year == 2023

    def test_enrichment_error_does_not_crash(self):
        """Exceptions during enrichment should be caught per-paper."""
        from cloud.upload_worker import _enrich_papers

        papers = [
            Paper(id="p1", title="Paper One", abstract="", doi="10.1234/test"),
            Paper(id="p2", title="Paper Two", abstract=""),
        ]

        db = MagicMock()
        with (
            patch.dict("os.environ", {"SEMANTIC_SCHOLAR_API_KEY": "fake_key"}),
            patch(
                "incite.corpus.enrichment.MetadataEnricher.enrich_from_doi",
                side_effect=Exception("API error"),
            ),
        ):
            # Should not raise
            count = _enrich_papers(papers, db, "job1")

        assert count == 0


class TestBibtexPaperDedup:
    """Test that BibTeX upload path does not create duplicate papers."""

    def test_bibtex_papers_not_duplicated_with_pdfs(self, tmp_path):
        """When bibtex_papers are provided, matched PDFs should attach
        GROBID results to the BibTeX paper ID, not create new papers."""
        from cloud.upload_worker import process_uploads

        papers = [
            Paper(
                id="bib_abc123",
                title="Test Paper",
                abstract="An abstract",
                authors=["Author One"],
                doi="10.1234/test",
                bibtex_key="Author2023",
            ),
        ]
        pdf_match = {"bib_abc123": "Author2023.pdf"}
        mock_db = MagicMock()

        with (
            patch("cloud.upload_worker.DATA_DIR", tmp_path),
            patch("cloud.database.get_db", return_value=mock_db),
            patch("cloud.database.update_library"),
            patch("cloud.database.update_processing_job"),
            patch("cloud.upload_worker._load_existing_corpus", return_value=[]),
            patch("cloud.upload_worker._load_existing_chunks", return_value=[]),
            patch("cloud.upload_worker._save_corpus") as mock_save,
            patch(
                "cloud.upload_worker._process_single_pdf",
                return_value=(None, {"sections": []}),
            ),
            patch("cloud.upload_worker._enrich_papers", return_value=0),
            patch("cloud.library_worker._build_paper_index"),
            patch("cloud.library_worker._build_chunk_index"),
            patch("cloud.library_worker._create_chunks", return_value=([], {"grobid_fulltext_papers": 0, "grobid_fulltext_chunks": 0, "abstract_only_papers": 0})),
            patch("cloud.library_worker._save_chunks"),
            patch("cloud.library_worker._update_progress"),
        ):
            process_uploads(
                library_id=1,
                job_id="job1",
                bibtex_papers=papers,
                pdf_match=pdf_match,
                pdf_filename_to_path={},
            )

            # Verify corpus was saved with exactly 1 paper (no duplicate)
            assert mock_save.called
            saved_papers = mock_save.call_args[0][1]
            assert len(saved_papers) == 1
            assert saved_papers[0].id == "bib_abc123"
