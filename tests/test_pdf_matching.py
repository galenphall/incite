"""Tests for PDF-to-paper matching (enrichment.py) and BibTeX upload pipeline (upload_worker.py)."""

import os
from unittest.mock import patch

import pytest

from incite.corpus.enrichment import _word_set, match_pdfs_to_papers
from incite.models import Paper

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_paper(**kwargs) -> Paper:
    """Create a Paper with sensible defaults."""
    defaults = dict(id="p1", title="Untitled", abstract="")
    defaults.update(kwargs)
    return Paper(**defaults)


# ---------------------------------------------------------------------------
# _word_set helper
# ---------------------------------------------------------------------------


class TestWordSet:
    def test_basic_tokenization(self):
        assert _word_set("Climate Change Impacts") == {"climate", "change", "impacts"}

    def test_filters_short_words(self):
        # "AI" and "of" are < 3 chars
        result = _word_set("AI Impacts of Climate")
        assert "climate" in result
        assert "impacts" in result
        assert "ai" not in result
        assert "of" not in result

    def test_strips_punctuation(self):
        result = _word_set("climate-change_2023 (impacts)")
        assert result == {"climate", "change", "2023", "impacts"}

    def test_empty_string(self):
        assert _word_set("") == set()

    def test_all_short_words(self):
        assert _word_set("A I of to") == set()

    def test_numeric_tokens(self):
        result = _word_set("year2023 v100")
        assert "year2023" in result
        assert "v100" in result


# ---------------------------------------------------------------------------
# Pass 1: Exact bibtex key match (existing, verify still works)
# ---------------------------------------------------------------------------


class TestBibtexKeyMatch:
    def test_exact_match(self):
        paper = _make_paper(id="p1", bibtex_key="Smith2020")
        result = match_pdfs_to_papers(["Smith2020.pdf"], [paper])
        assert result == {"p1": "Smith2020.pdf"}

    def test_case_insensitive(self):
        paper = _make_paper(id="p1", bibtex_key="Smith2020")
        result = match_pdfs_to_papers(["smith2020.pdf"], [paper])
        assert result == {"p1": "smith2020.pdf"}

    def test_no_bibtex_key_skips(self):
        paper = _make_paper(id="p1", bibtex_key=None)
        result = match_pdfs_to_papers(["anything.pdf"], [paper])
        assert result == {}


# ---------------------------------------------------------------------------
# Pass 2: DOI-in-filename match (NEW)
# ---------------------------------------------------------------------------


class TestDoiFilenameMatch:
    def test_doi_with_underscores(self):
        paper = _make_paper(id="p1", doi="10.1038/s41558-023-01234-5")
        result = match_pdfs_to_papers(["10.1038_s41558-023-01234-5.pdf"], [paper])
        assert result == {"p1": "10.1038_s41558-023-01234-5.pdf"}

    def test_doi_with_hyphens_replaced(self):
        paper = _make_paper(id="p1", doi="10.1038/nclimate2100")
        result = match_pdfs_to_papers(["10.1038_nclimate2100.pdf"], [paper])
        assert result == {"p1": "10.1038_nclimate2100.pdf"}

    def test_doi_embedded_in_longer_filename(self):
        paper = _make_paper(id="p1", doi="10.1038/s41558-023-01234-5")
        result = match_pdfs_to_papers(["downloaded_10.1038_s41558_023_01234_5_final.pdf"], [paper])
        assert result == {"p1": "downloaded_10.1038_s41558_023_01234_5_final.pdf"}

    def test_no_doi_skips(self):
        paper = _make_paper(id="p1", doi=None)
        result = match_pdfs_to_papers(["10.1038_something.pdf"], [paper])
        assert result == {}

    def test_short_doi_skips(self):
        paper = _make_paper(id="p1", doi="10.1")
        result = match_pdfs_to_papers(["10.1.pdf"], [paper])
        assert result == {}

    def test_doi_takes_priority_over_title(self):
        """DOI match (pass 2) should match before title substring (pass 3)."""
        paper = _make_paper(
            id="p1",
            title="A Very Long Title About Climate Change",
            doi="10.1038/nclimate2100",
        )
        files = ["10.1038_nclimate2100.pdf", "averylongtitleaboutclimatechange.pdf"]
        result = match_pdfs_to_papers(files, [paper])
        assert result["p1"] == "10.1038_nclimate2100.pdf"


# ---------------------------------------------------------------------------
# Pass 3: Title substring match (existing, verify still works)
# ---------------------------------------------------------------------------


class TestTitleSubstringMatch:
    def test_title_prefix_in_filename(self):
        paper = _make_paper(id="p1", title="Climate Change Impacts on Agriculture")
        result = match_pdfs_to_papers(["climatechangeimpactsonagriculture.pdf"], [paper])
        assert result == {"p1": "climatechangeimpactsonagriculture.pdf"}

    def test_short_title_skipped(self):
        paper = _make_paper(id="p1", title="Short")
        result = match_pdfs_to_papers(["short.pdf"], [paper])
        assert result == {}


# ---------------------------------------------------------------------------
# Pass 4: Fuzzy Jaccard title match (NEW)
# ---------------------------------------------------------------------------


class TestJaccardTitleMatch:
    def test_reordered_words(self):
        """Jaccard should match even when words are in different order."""
        paper = _make_paper(id="p1", title="Climate Change Impacts on Global Agriculture Systems")
        result = match_pdfs_to_papers(["global-agriculture-systems-climate-impacts.pdf"], [paper])
        assert result == {"p1": "global-agriculture-systems-climate-impacts.pdf"}

    def test_partial_overlap_above_threshold(self):
        """4 out of 6 unique words = Jaccard ~0.57, should match."""
        paper = _make_paper(id="p1", title="Machine Learning Methods for Climate Prediction")
        # Shares: machine, learning, climate, prediction (4 words)
        # Union: machine, learning, methods, climate, prediction, applied, weather (7)
        # Jaccard = 4/7 ≈ 0.57, overlap = 4 >= 3
        result = match_pdfs_to_papers(
            ["applied-machine-learning-climate-prediction-weather.pdf"], [paper]
        )
        assert result == {"p1": "applied-machine-learning-climate-prediction-weather.pdf"}

    def test_low_overlap_rejected(self):
        """Only 2 shared words should not match (< 3 overlap requirement)."""
        paper = _make_paper(id="p1", title="Climate Change Impacts on Global Agriculture Systems")
        # Only shares "climate" and "impacts" (2 words) — below the 3-word minimum
        result = match_pdfs_to_papers(["climate-impacts-review.pdf"], [paper])
        assert result == {}

    def test_low_jaccard_rejected(self):
        """Even with 3+ shared words, Jaccard < 0.5 should not match."""
        paper = _make_paper(id="p1", title="Climate Change Impacts on Agriculture")
        # Title words (>= 3 chars): climate, change, impacts, agriculture (4)
        # Filename adds many extra: climate, change, impacts, plus 6 more = 10 unique
        # Jaccard = 3/10 = 0.3, below 0.5
        result = match_pdfs_to_papers(
            ["climate-change-impacts-extra-words-that-dilute-the-similarity-heavily.pdf"],
            [paper],
        )
        assert result == {}

    def test_short_title_skipped(self):
        """Titles with < 3 words (>= 3 chars each) should be skipped."""
        paper = _make_paper(id="p1", title="AI Today")
        result = match_pdfs_to_papers(["ai-today-review.pdf"], [paper])
        assert result == {}


# ---------------------------------------------------------------------------
# Pass 5: Author last name + year (existing, verify still works)
# ---------------------------------------------------------------------------


class TestAuthorYearMatch:
    def test_author_year_match(self):
        paper = _make_paper(id="p1", authors=["John Smith"], year=2021)
        result = match_pdfs_to_papers(["Smith_2021_something.pdf"], [paper])
        assert result == {"p1": "Smith_2021_something.pdf"}

    def test_missing_author_skips(self):
        paper = _make_paper(id="p1", authors=[], year=2021)
        result = match_pdfs_to_papers(["Smith_2021.pdf"], [paper])
        assert result == {}

    def test_missing_year_skips(self):
        paper = _make_paper(id="p1", authors=["John Smith"], year=None)
        result = match_pdfs_to_papers(["Smith_2021.pdf"], [paper])
        assert result == {}


# ---------------------------------------------------------------------------
# Multi-pass priority and deduplication
# ---------------------------------------------------------------------------


class TestMultiPassPriority:
    def test_earlier_pass_wins(self):
        """A file matched by bibtex key (pass 1) should not be reused by DOI (pass 2)."""
        paper1 = _make_paper(id="p1", bibtex_key="Smith2020")
        paper2 = _make_paper(id="p2", doi="10.1038/smith2020")
        result = match_pdfs_to_papers(["Smith2020.pdf"], [paper1, paper2])
        assert result == {"p1": "Smith2020.pdf"}
        assert "p2" not in result

    def test_file_consumed_once(self):
        """Each file can only match one paper."""
        paper1 = _make_paper(id="p1", doi="10.1038/abc")
        paper2 = _make_paper(id="p2", doi="10.1038/abc")
        result = match_pdfs_to_papers(["10.1038_abc.pdf"], [paper1, paper2])
        assert len(result) == 1

    def test_multiple_papers_multiple_files(self):
        papers = [
            _make_paper(id="p1", bibtex_key="Key1"),
            _make_paper(id="p2", doi="10.1038/nature123"),
            _make_paper(
                id="p3",
                title="Deep Learning for Natural Language Processing Tasks",
            ),
            _make_paper(id="p4", authors=["Jane Doe"], year=2022),
        ]
        files = [
            "Key1.pdf",
            "10.1038_nature123.pdf",
            "natural-language-processing-deep-learning-tasks.pdf",
            "Doe_2022_review.pdf",
        ]
        result = match_pdfs_to_papers(files, papers)
        assert result["p1"] == "Key1.pdf"
        assert result["p2"] == "10.1038_nature123.pdf"
        assert result["p3"] == "natural-language-processing-deep-learning-tasks.pdf"
        assert result["p4"] == "Doe_2022_review.pdf"

    def test_no_match_returns_empty(self):
        paper = _make_paper(id="p1", title="Quantum Entanglement")
        result = match_pdfs_to_papers(["totally_unrelated.pdf"], [paper])
        assert result == {}


# ---------------------------------------------------------------------------
# BibTeX + PDF upload pipeline (upload_worker.process_uploads)
# ---------------------------------------------------------------------------


# Set env vars required by cloud modules
os.environ.setdefault("ENCRYPTION_KEY", "7gRTU890D-m7JaPp6-ks4KMHLpXs5-ugRvtjfAXwZPE=")
os.environ.setdefault("INVITE_CODE", "test-invite-123")


@pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — skipping Postgres-dependent tests",
)
class TestProcessUploadsBibtex:
    """Test that process_uploads correctly uses bibtex_papers as source of truth."""

    @pytest.fixture
    def setup_db(self, tmp_path):
        """Create a Postgres connection with a user, library, and job."""
        import uuid

        from cloud.database import (
            create_library_manual,
            create_processing_job,
            create_user,
            get_db,
            init_db,
            return_conn,
        )

        init_db()
        db = get_db()

        email = f"test-pdf-{uuid.uuid4().hex[:8]}@example.com"
        uid = create_user(db, "Test User", email, "hashed_pw")
        lib_id = create_library_manual(db, uid)
        job_id = create_processing_job(db, lib_id)

        # Library dir that process_uploads will use
        lib_dir = tmp_path / str(lib_id)
        lib_dir.mkdir(parents=True, exist_ok=True)

        yield db, lib_id, job_id, lib_dir, tmp_path
        return_conn(db)

    @pytest.fixture
    def sample_papers(self):
        return [
            _make_paper(
                id="bib_paper1",
                title="Climate Impacts",
                doi="10.1038/climate1",
                bibtex_key="Smith2020",
                abstract="Abstract about climate.",
            ),
            _make_paper(
                id="bib_paper2",
                title="Ocean Warming",
                doi="10.1038/ocean1",
                abstract="Abstract about oceans.",
            ),
        ]

    def _run_process_uploads(
        self,
        setup_db,
        sample_papers,
        pdf_match=None,
        fname_to_path=None,
        mock_grobid_return=(None, None),
    ):
        """Helper to run process_uploads with all the right mocks."""
        import json

        from cloud.upload_worker import process_uploads

        db, lib_id, job_id, lib_dir, tmp_path = setup_db

        with (
            patch("cloud.database.get_db", return_value=db),
            patch("cloud.upload_worker.DATA_DIR", tmp_path),
            patch(
                "cloud.upload_worker._process_single_pdf", return_value=mock_grobid_return
            ) as mock_grobid,
            patch("cloud.upload_worker._enrich_papers"),
            patch("cloud.library_worker._build_paper_index"),
            patch("cloud.library_worker._build_chunk_index"),
            patch(
                "cloud.library_worker._create_chunks",
                return_value=(
                    [],
                    {
                        "grobid_fulltext_papers": 0,
                        "grobid_fulltext_chunks": 0,
                        "abstract_only_papers": 0,
                    },
                ),
            ),
            patch("cloud.library_worker._save_chunks"),
        ):
            process_uploads(
                library_id=lib_id,
                job_id=job_id,
                bibtex_papers=sample_papers,
                pdf_match=pdf_match or {},
                pdf_filename_to_path=fname_to_path or {},
            )

        corpus_path = lib_dir / "corpus.jsonl"
        if corpus_path.exists():
            papers_saved = [
                json.loads(line) for line in corpus_path.read_text().splitlines() if line.strip()
            ]
        else:
            papers_saved = []

        return papers_saved, mock_grobid

    def test_bibtex_papers_are_source_of_truth(self, setup_db, sample_papers):
        """BibTeX papers should be added to corpus even without PDF matches."""
        papers_saved, _ = self._run_process_uploads(setup_db, sample_papers)

        paper_ids = {p["id"] for p in papers_saved}
        assert "bib_paper1" in paper_ids
        assert "bib_paper2" in paper_ids

    def test_matched_pdfs_processed_through_grobid(self, setup_db, sample_papers):
        """Matched PDFs should be sent through GROBID with the BibTeX paper ID."""
        db, lib_id, job_id, lib_dir, tmp_path = setup_db

        fake_pdf = lib_dir / "Smith2020.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        grobid_result = {
            "title": "Climate Impacts",
            "abstract": "Full abstract from GROBID",
            "sections": [
                {"heading": "Introduction", "text": "Intro text.", "section_type": "body"}
            ],
        }

        from cloud.upload_worker import process_uploads

        with (
            patch("cloud.database.get_db", return_value=db),
            patch("cloud.upload_worker.DATA_DIR", tmp_path),
            patch(
                "cloud.upload_worker._process_single_pdf", return_value=(None, grobid_result)
            ) as mock_grobid,
            patch("cloud.upload_worker._enrich_papers"),
            patch("cloud.library_worker._build_paper_index"),
            patch("cloud.library_worker._build_chunk_index"),
            patch(
                "cloud.library_worker._create_chunks",
                return_value=(
                    [],
                    {
                        "grobid_fulltext_papers": 0,
                        "grobid_fulltext_chunks": 0,
                        "abstract_only_papers": 0,
                    },
                ),
            ),
            patch("cloud.library_worker._save_chunks"),
        ):
            process_uploads(
                library_id=lib_id,
                job_id=job_id,
                bibtex_papers=sample_papers,
                pdf_match={"bib_paper1": "Smith2020.pdf"},
                pdf_filename_to_path={"Smith2020.pdf": fake_pdf},
            )
            mock_grobid.assert_called_once()
            call_args = mock_grobid.call_args
            assert call_args[0] == (fake_pdf, "http://grobid:8070")

    def test_no_duplicate_papers_from_matched_pdfs(self, setup_db, sample_papers):
        """Matched PDFs should NOT create duplicate papers — BibTeX paper is the only entry."""
        db, lib_id, job_id, lib_dir, tmp_path = setup_db

        fake_pdf = lib_dir / "Smith2020.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        grobid_result = {
            "title": "Climate Impacts",
            "abstract": "Full abstract",
            "sections": [],
        }

        papers_saved, _ = self._run_process_uploads(
            setup_db,
            sample_papers,
            pdf_match={"bib_paper1": "Smith2020.pdf"},
            fname_to_path={"Smith2020.pdf": fake_pdf},
            mock_grobid_return=(None, grobid_result),
        )

        paper_ids = [p["id"] for p in papers_saved]
        assert len(paper_ids) == len(set(paper_ids)), f"Duplicate paper IDs: {paper_ids}"
        assert len(paper_ids) == 2  # exactly the 2 BibTeX papers

    def test_unmatched_pdfs_create_new_papers(self, setup_db, sample_papers):
        """PDFs that don't match any BibTeX paper should create new papers via GROBID."""
        db, lib_id, job_id, lib_dir, tmp_path = setup_db

        fake_pdf = lib_dir / "unknown_paper.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        new_paper = _make_paper(id="upload_abc123", title="Unknown Paper from PDF")
        grobid_result = {
            "title": "Unknown Paper from PDF",
            "abstract": "Abstract from GROBID",
            "sections": [],
        }

        papers_saved, _ = self._run_process_uploads(
            setup_db,
            sample_papers,
            pdf_match={},
            fname_to_path={"unknown_paper.pdf": fake_pdf},
            mock_grobid_return=(new_paper, grobid_result),
        )

        paper_ids = {p["id"] for p in papers_saved}
        assert len(paper_ids) == 3  # 2 BibTeX + 1 from unmatched PDF
        assert "upload_abc123" in paper_ids
