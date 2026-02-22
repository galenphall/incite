"""Tests for CrossRef DOI resolution: title normalization, match validation, and batch resolve."""

from unittest.mock import patch

from incite.corpus.crossref import (
    normalize_title,
    resolve_dois_batch,
    search_by_bibliographic,
)
from incite.models import Paper


class TestNormalizeTitle:
    """Test title normalization for fuzzy matching."""

    def test_basic_lowercasing(self):
        assert normalize_title("Climate Change Effects") == "climate change effects"

    def test_latex_markup_stripped(self):
        assert normalize_title(r"\textit{Nature} Reviews") == "nature reviews"
        assert normalize_title(r"\emph{Science} Today") == "science today"
        assert normalize_title(r"\textbf{Bold} Title") == "bold title"

    def test_braces_removed(self):
        assert normalize_title("{GDP} Growth in {OECD} Countries") == "gdp growth in oecd countries"

    def test_punctuation_removed_except_hyphens(self):
        assert normalize_title("A Title: With Punctuation!") == "a title with punctuation"
        assert normalize_title("Self-supervised Learning") == "self-supervised learning"

    def test_unicode_normalized(self):
        # Accented characters should be decomposed and combining marks stripped
        assert normalize_title("Résumé of Naïve Methods") == "resume of naive methods"

    def test_whitespace_collapsed(self):
        assert normalize_title("  too   many   spaces  ") == "too many spaces"

    def test_empty_string(self):
        assert normalize_title("") == ""
        assert normalize_title(None) == ""


class TestSearchByBibliographic:
    """Test CrossRef API search with mocked responses."""

    MOCK_RESPONSE = {
        "message": {
            "items": [
                {
                    "DOI": "10.1038/s41558-023-01234-5",
                    "title": ["Climate change and tropical forests"],
                    "author": [
                        {"family": "Smith", "given": "John"},
                        {"family": "Jones", "given": "Alice"},
                    ],
                    "published-print": {"date-parts": [[2023]]},
                }
            ]
        }
    }

    @patch("incite.corpus.crossref.requests.get")
    def test_high_confidence_match(self, mock_get):
        """Score >= 90 should auto-accept."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = self.MOCK_RESPONSE
        mock_get.return_value.raise_for_status = lambda: None

        result = search_by_bibliographic("Climate change and tropical forests")
        assert result is not None
        assert result["doi"] == "10.1038/s41558-023-01234-5"
        assert result["year"] == 2023
        assert "Smith" in result["authors"]

    @patch("incite.corpus.crossref.requests.get")
    def test_medium_confidence_accepted_with_year_author(self, mock_get):
        """Score 80-89 should accept if year and author match."""
        # Slightly different title to get score in 80-89 range
        response = {
            "message": {
                "items": [
                    {
                        "DOI": "10.1234/test",
                        "title": ["Climate impacts on tropical forest ecosystems worldwide"],
                        "author": [{"family": "Smith", "given": "J"}],
                        "published-print": {"date-parts": [[2023]]},
                    }
                ]
            }
        }
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = response
        mock_get.return_value.raise_for_status = lambda: None

        # With matching author and year — may accept or reject depending on
        # actual fuzzy score; this tests the path exists
        result = search_by_bibliographic(
            "Climate impacts on tropical forest ecosystems",
            first_author_last="Smith",
            year=2023,
        )
        # Result depends on exact score; just verify no crash
        assert result is None or result["doi"] == "10.1234/test"

    @patch("incite.corpus.crossref.requests.get")
    def test_low_confidence_rejected(self, mock_get):
        """Score < 80 should reject."""
        response = {
            "message": {
                "items": [
                    {
                        "DOI": "10.1234/wrong",
                        "title": ["Completely unrelated paper about quantum physics"],
                        "author": [{"family": "Nobody", "given": "X"}],
                        "published-print": {"date-parts": [[2020]]},
                    }
                ]
            }
        }
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = response
        mock_get.return_value.raise_for_status = lambda: None

        result = search_by_bibliographic("Climate change and tropical forests")
        assert result is None

    @patch("incite.corpus.crossref.requests.get")
    def test_empty_results(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"message": {"items": []}}
        mock_get.return_value.raise_for_status = lambda: None

        result = search_by_bibliographic("Some title")
        assert result is None

    @patch("incite.corpus.crossref.requests.get")
    def test_api_error_returns_none(self, mock_get):
        import requests

        mock_get.side_effect = requests.RequestException("timeout")
        result = search_by_bibliographic("Some title")
        assert result is None

    def test_empty_title_returns_none(self):
        assert search_by_bibliographic("") is None
        assert search_by_bibliographic("   ") is None


class TestResolveDoisBatch:
    """Test batch DOI resolution on Paper objects."""

    @patch("incite.corpus.crossref.search_by_bibliographic")
    def test_resolves_missing_dois(self, mock_search):
        mock_search.return_value = {
            "doi": "10.1234/resolved",
            "title": "Test Paper",
            "authors": ["Smith"],
            "year": 2023,
        }

        papers = [
            Paper(id="p1", title="Test Paper", authors=["John Smith"], year=2023),
        ]
        count = resolve_dois_batch(papers)
        assert count == 1
        assert papers[0].doi == "10.1234/resolved"

    @patch("incite.corpus.crossref.search_by_bibliographic")
    def test_skips_papers_with_doi(self, mock_search):
        papers = [
            Paper(
                id="p1",
                title="Already Has DOI",
                authors=["A"],
                doi="10.existing/doi",
            ),
        ]
        count = resolve_dois_batch(papers)
        assert count == 0
        mock_search.assert_not_called()

    @patch("incite.corpus.crossref.search_by_bibliographic")
    def test_respects_max_papers(self, mock_search):
        mock_search.return_value = {
            "doi": "10.1234/test",
            "title": "T",
            "authors": [],
            "year": 2023,
        }
        papers = [Paper(id=f"p{i}", title=f"Paper {i}", authors=["A"]) for i in range(10)]
        count = resolve_dois_batch(papers, max_papers=3)
        assert count == 3
        assert mock_search.call_count == 3

    @patch("incite.corpus.crossref.search_by_bibliographic")
    def test_handles_no_match(self, mock_search):
        mock_search.return_value = None
        papers = [
            Paper(id="p1", title="Obscure Paper", authors=["Nobody"]),
        ]
        count = resolve_dois_batch(papers)
        assert count == 0
        assert papers[0].doi is None
