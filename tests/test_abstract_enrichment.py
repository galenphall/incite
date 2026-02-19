"""Tests for abstract enrichment pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from incite.corpus.enrichment import EnrichmentResult, enrich_abstracts_batch
from incite.models import Paper


def _make_paper(id: str, title: str, abstract: str = "", doi: str | None = None) -> Paper:
    return Paper(id=id, title=title, abstract=abstract, authors=["Auth"], year=2020, doi=doi)


class TestEnrichAbstractsBatch:
    """Tests for enrich_abstracts_batch()."""

    def test_skip_papers_with_abstracts(self):
        """Papers that already have abstracts are not touched."""
        papers = [_make_paper("1", "Paper One", abstract="Has abstract", doi="10.1/a")]
        result = enrich_abstracts_batch(papers)
        assert result.total_missing == 0
        assert result.still_missing == 0
        assert papers[0].abstract == "Has abstract"

    def test_doi_lookup_via_s2_batch(self):
        """Papers with DOIs get abstracts from S2 batch endpoint."""
        papers = [
            _make_paper("1", "Paper One", doi="10.1/a"),
            _make_paper("2", "Paper Two", doi="10.1/b"),
        ]

        s2 = MagicMock()
        s2.get_papers_batch.return_value = {
            "DOI:10.1/a": Paper(
                id="s2_1", title="Paper One", abstract="Found abstract A", authors=[], year=2020
            ),
            "DOI:10.1/b": Paper(
                id="s2_2", title="Paper Two", abstract="Found abstract B", authors=[], year=2020
            ),
        }

        result = enrich_abstracts_batch(papers, s2_client=s2)
        assert result.found_by_doi == 2
        assert result.still_missing == 0
        assert papers[0].abstract == "Found abstract A"
        assert papers[1].abstract == "Found abstract B"
        s2.get_papers_batch.assert_called_once()

    def test_s2_miss_falls_through_to_openalex(self):
        """S2 misses fall through to OpenAlex batch lookup."""
        papers = [_make_paper("1", "Paper One", doi="10.1/a")]

        s2 = MagicMock()
        s2.get_papers_batch.return_value = {}  # S2 found nothing

        oa = MagicMock()
        oa.get_works_batch_by_doi.return_value = {
            "10.1/a": Paper(
                id="oa_1", title="Paper One", abstract="OpenAlex abstract", authors=[], year=2020
            ),
        }

        result = enrich_abstracts_batch(papers, s2_client=s2, openalex_client=oa)
        assert result.found_by_doi == 1
        assert papers[0].abstract == "OpenAlex abstract"
        oa.get_works_batch_by_doi.assert_called_once()

    def test_no_doi_uses_title_search(self):
        """Papers without DOIs use S2 title search with fuzzy matching."""
        papers = [_make_paper("1", "Machine Learning in Healthcare")]

        s2 = MagicMock()
        s2.get_papers_batch.return_value = {}  # Not called for no-DOI papers
        s2.search_papers.return_value = [
            Paper(
                id="s2_1",
                title="Machine Learning in Healthcare",
                abstract="Title search abstract",
                authors=[],
                year=2020,
            ),
        ]

        result = enrich_abstracts_batch(papers, s2_client=s2)
        assert result.found_by_title == 1
        assert papers[0].abstract == "Title search abstract"

    def test_title_search_fuzzy_match_rejects_low_similarity(self):
        """Title search rejects candidates with low Jaccard similarity."""
        papers = [_make_paper("1", "Machine Learning in Healthcare")]

        s2 = MagicMock()
        s2.get_papers_batch.return_value = {}
        s2.search_papers.return_value = [
            Paper(
                id="s2_1",
                title="Totally Different Paper About Physics",
                abstract="Wrong abstract",
                authors=[],
                year=2020,
            ),
        ]

        result = enrich_abstracts_batch(papers, s2_client=s2)
        assert result.found_by_title == 0
        assert papers[0].abstract == ""

    def test_api_failures_propagate(self):
        """API failures propagate (caller is responsible for catching)."""
        papers = [_make_paper("1", "Paper One", doi="10.1/a")]

        s2 = MagicMock()
        s2.get_papers_batch.side_effect = Exception("API error")

        with pytest.raises(Exception, match="API error"):
            enrich_abstracts_batch(papers, s2_client=s2)

    def test_no_clients_returns_early(self):
        """With no API clients, returns immediately."""
        papers = [_make_paper("1", "Paper One", doi="10.1/a")]
        result = enrich_abstracts_batch(papers)
        assert result.total_missing == 1
        assert result.still_missing == 1

    def test_progress_callback_called(self):
        """Progress callback receives status messages."""
        papers = [_make_paper("1", "Paper One", doi="10.1/a")]

        s2 = MagicMock()
        s2.get_papers_batch.return_value = {}
        callback = MagicMock()

        enrich_abstracts_batch(papers, s2_client=s2, progress_callback=callback)
        assert callback.call_count >= 2  # At least start + summary

    def test_mixed_doi_and_no_doi(self):
        """Mix of DOI and no-DOI papers handled correctly."""
        papers = [
            _make_paper("1", "With DOI", doi="10.1/a"),
            _make_paper("2", "Without DOI"),
            _make_paper("3", "Has Abstract", abstract="Already here", doi="10.1/b"),
        ]

        s2 = MagicMock()
        s2.get_papers_batch.return_value = {
            "DOI:10.1/a": Paper(
                id="s2_1", title="With DOI", abstract="S2 abstract", authors=[], year=2020
            ),
        }
        s2.search_papers.return_value = []

        result = enrich_abstracts_batch(papers, s2_client=s2)
        assert result.total_missing == 2  # Only 2 missing (3rd has abstract)
        assert result.found_by_doi == 1
        assert papers[0].abstract == "S2 abstract"
        assert papers[2].abstract == "Already here"

    def test_enrichment_result_frozen(self):
        """EnrichmentResult is immutable."""
        result = EnrichmentResult(
            total_missing=5, found_by_doi=3, found_by_title=1, still_missing=1
        )
        with pytest.raises(AttributeError):
            result.total_missing = 10


class TestSemanticScholarBatch:
    """Tests for SemanticScholarClient.get_papers_batch()."""

    @patch("incite.corpus.semantic_scholar.requests.post")
    def test_batch_returns_papers(self, mock_post):
        from incite.corpus.semantic_scholar import SemanticScholarClient

        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "paperId": "abc123",
                "title": "Test Paper",
                "abstract": "Test abstract",
                "authors": [{"name": "Alice"}],
                "year": 2023,
                "externalIds": {"DOI": "10.1/test"},
            },
        ]
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = SemanticScholarClient(api_key="test", delay=0)
        results = client.get_papers_batch(["DOI:10.1/test"])

        assert "DOI:10.1/test" in results
        assert results["DOI:10.1/test"].abstract == "Test abstract"

    @patch("incite.corpus.semantic_scholar.requests.post")
    def test_batch_skips_null_entries(self, mock_post):
        from incite.corpus.semantic_scholar import SemanticScholarClient

        mock_response = MagicMock()
        mock_response.json.return_value = [None, None]
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = SemanticScholarClient(delay=0)
        results = client.get_papers_batch(["DOI:10.1/a", "DOI:10.1/b"])
        assert len(results) == 0

    @patch("incite.corpus.semantic_scholar.requests.post")
    def test_batch_skips_no_abstract(self, mock_post):
        from incite.corpus.semantic_scholar import SemanticScholarClient

        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "paperId": "abc",
                "title": "No Abstract Paper",
                "abstract": None,
                "authors": [],
                "year": 2023,
                "externalIds": {},
            },
        ]
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = SemanticScholarClient(delay=0)
        results = client.get_papers_batch(["DOI:10.1/a"])
        assert len(results) == 0


class TestOpenAlexBatchByDoi:
    """Tests for OpenAlexClient.get_works_batch_by_doi()."""

    @patch("incite.corpus.openalex.requests.get")
    def test_batch_by_doi_returns_papers(self, mock_get):
        from incite.corpus.openalex import OpenAlexClient

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "id": "https://openalex.org/W123",
                    "title": "Test Paper",
                    "abstract_inverted_index": {"Test": [0], "abstract": [1]},
                    "authorships": [{"author": {"display_name": "Bob"}}],
                    "publication_year": 2023,
                    "doi": "https://doi.org/10.1/test",
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        client = OpenAlexClient(email="test@test.com", delay=0)
        results = client.get_works_batch_by_doi(["10.1/test"])

        assert "10.1/test" in results
        assert results["10.1/test"].abstract == "Test abstract"
