"""Tests for cloud.library_search — multi-field library search."""

import pytest

from cloud.library_search import SearchToken, paper_matches_search, parse_search_query

# -- Fixtures --


@pytest.fixture
def newman_paper():
    return {
        "title": "Fast algorithm for detecting community structure in networks",
        "authors": ["Mark Newman", "Michelle Girvan"],
        "abstract": "We propose an algorithm for detecting community structure.",
        "year": 2004,
        "journal": "Physical Review E",
    }


@pytest.fixture
def watts_paper():
    return {
        "title": "Collective dynamics of small-world networks",
        "authors": ["Duncan Watts", "Steven Strogatz"],
        "abstract": "We explore a model of small-world networks with clustering.",
        "year": 1998,
        "journal": "Nature",
    }


@pytest.fixture
def sparse_paper():
    """Paper with minimal metadata — no abstract, journal, or year."""
    return {
        "title": "Some Working Paper",
        "authors": ["Jane Doe"],
    }


# -- parse_search_query tests --


class TestParseSearchQuery:
    def test_empty_query(self):
        assert parse_search_query("") == []
        assert parse_search_query("   ") == []

    def test_single_word(self):
        tokens = parse_search_query("newman")
        assert len(tokens) == 1
        assert tokens[0] == SearchToken(text="newman", field=None)

    def test_multiple_words(self):
        tokens = parse_search_query("Newman fast")
        assert len(tokens) == 2
        assert tokens[0] == SearchToken(text="newman", field=None)
        assert tokens[1] == SearchToken(text="fast", field=None)

    def test_quoted_phrase(self):
        tokens = parse_search_query('"fast algorithm"')
        assert len(tokens) == 1
        assert tokens[0] == SearchToken(text="fast algorithm", field=None)

    def test_quoted_phrase_with_other_words(self):
        tokens = parse_search_query('Newman "fast algorithm"')
        assert len(tokens) == 2
        assert tokens[0] == SearchToken(text="newman", field=None)
        assert tokens[1] == SearchToken(text="fast algorithm", field=None)

    def test_field_prefix_author(self):
        tokens = parse_search_query("author:Newman")
        assert len(tokens) == 1
        assert tokens[0] == SearchToken(text="newman", field="author")

    def test_field_prefix_title(self):
        tokens = parse_search_query("title:fast")
        assert len(tokens) == 1
        assert tokens[0] == SearchToken(text="fast", field="title")

    def test_field_prefix_year(self):
        tokens = parse_search_query("year:2004")
        assert len(tokens) == 1
        assert tokens[0] == SearchToken(text="2004", field="year")

    def test_field_prefix_journal(self):
        tokens = parse_search_query("journal:nature")
        assert len(tokens) == 1
        assert tokens[0] == SearchToken(text="nature", field="journal")

    def test_mixed_prefixed_and_unprefixed(self):
        tokens = parse_search_query("author:Newman community")
        assert len(tokens) == 2
        assert tokens[0] == SearchToken(text="newman", field="author")
        assert tokens[1] == SearchToken(text="community", field=None)

    def test_unknown_prefix_treated_as_plain_text(self):
        tokens = parse_search_query("doi:10.1234")
        assert len(tokens) == 1
        # Not a recognized prefix, so the whole thing is treated as text
        assert tokens[0] == SearchToken(text="doi:10.1234", field=None)

    def test_bare_prefix_with_no_value_skipped(self):
        tokens = parse_search_query("author: newman")
        # "author:" has no value attached (space separates), so it's skipped
        # "newman" is a plain token
        # Actually shlex will parse "author:" as one token and "newman" as another
        assert len(tokens) == 1
        assert tokens[0] == SearchToken(text="newman", field=None)

    def test_unmatched_quote_fallback(self):
        """Unmatched quotes should not crash, should fall back gracefully."""
        tokens = parse_search_query('"fast algorithm')
        # Should still produce at least the words
        assert len(tokens) >= 1
        texts = [t.text for t in tokens]
        assert "fast" in texts or "fast algorithm" in texts

    def test_case_insensitive(self):
        tokens = parse_search_query("NEWMAN Fast")
        assert tokens[0].text == "newman"
        assert tokens[1].text == "fast"

    def test_field_prefix_case_insensitive(self):
        tokens = parse_search_query("Author:Newman")
        assert tokens[0] == SearchToken(text="newman", field="author")


# -- paper_matches_search tests --


class TestPaperMatchesSearch:
    def test_empty_tokens_matches_everything(self, newman_paper):
        assert paper_matches_search(newman_paper, []) is True

    def test_single_word_in_title(self, newman_paper):
        tokens = parse_search_query("fast")
        assert paper_matches_search(newman_paper, tokens) is True

    def test_single_word_in_authors(self, newman_paper):
        tokens = parse_search_query("newman")
        assert paper_matches_search(newman_paper, tokens) is True

    def test_single_word_in_abstract(self, newman_paper):
        tokens = parse_search_query("community")
        assert paper_matches_search(newman_paper, tokens) is True

    def test_single_word_not_found(self, newman_paper):
        tokens = parse_search_query("blockchain")
        assert paper_matches_search(newman_paper, tokens) is False

    def test_and_semantics_both_match(self, newman_paper):
        """'Newman fast' should match: newman in authors, fast in title."""
        tokens = parse_search_query("Newman fast")
        assert paper_matches_search(newman_paper, tokens) is True

    def test_and_semantics_one_missing(self, newman_paper):
        """'Newman blockchain' should NOT match: blockchain not found."""
        tokens = parse_search_query("Newman blockchain")
        assert paper_matches_search(newman_paper, tokens) is False

    def test_field_prefix_author_match(self, newman_paper):
        tokens = parse_search_query("author:newman")
        assert paper_matches_search(newman_paper, tokens) is True

    def test_field_prefix_author_no_match(self, newman_paper):
        tokens = parse_search_query("author:fast")
        assert paper_matches_search(newman_paper, tokens) is False

    def test_field_prefix_title_match(self, newman_paper):
        tokens = parse_search_query("title:fast")
        assert paper_matches_search(newman_paper, tokens) is True

    def test_field_prefix_title_no_match(self, newman_paper):
        tokens = parse_search_query("title:newman")
        assert paper_matches_search(newman_paper, tokens) is False

    def test_field_prefix_year_match(self, newman_paper):
        tokens = parse_search_query("year:2004")
        assert paper_matches_search(newman_paper, tokens) is True

    def test_field_prefix_year_no_match(self, newman_paper):
        tokens = parse_search_query("year:2020")
        assert paper_matches_search(newman_paper, tokens) is False

    def test_field_prefix_journal_match(self, newman_paper):
        tokens = parse_search_query("journal:physical")
        assert paper_matches_search(newman_paper, tokens) is True

    def test_field_prefix_journal_no_match(self, watts_paper):
        tokens = parse_search_query("journal:science")
        assert paper_matches_search(watts_paper, tokens) is False

    def test_mixed_field_and_global(self, newman_paper):
        """author:Newman + 'community' (global) -- both should match."""
        tokens = parse_search_query("author:Newman community")
        assert paper_matches_search(newman_paper, tokens) is True

    def test_mixed_field_and_global_no_match(self, newman_paper):
        """author:Newman + 'blockchain' (global) -- second should fail."""
        tokens = parse_search_query("author:Newman blockchain")
        assert paper_matches_search(newman_paper, tokens) is False

    def test_quoted_phrase_match(self, newman_paper):
        tokens = parse_search_query('"fast algorithm"')
        assert paper_matches_search(newman_paper, tokens) is True

    def test_quoted_phrase_no_match(self, newman_paper):
        tokens = parse_search_query('"algorithm fast"')
        assert paper_matches_search(newman_paper, tokens) is False

    def test_year_as_unprefixed_token(self, newman_paper):
        """Year should be searchable even without the year: prefix."""
        tokens = parse_search_query("2004")
        assert paper_matches_search(newman_paper, tokens) is True

    def test_journal_as_unprefixed_token(self, newman_paper):
        """Journal should be searchable even without the journal: prefix."""
        tokens = parse_search_query("physical")
        assert paper_matches_search(newman_paper, tokens) is True

    def test_sparse_paper_no_crash(self, sparse_paper):
        """Papers with missing fields should not crash."""
        tokens = parse_search_query("doe working")
        assert paper_matches_search(sparse_paper, tokens) is True

    def test_sparse_paper_year_search(self, sparse_paper):
        """Searching by year on a paper with no year should not match."""
        tokens = parse_search_query("year:2020")
        assert paper_matches_search(sparse_paper, tokens) is False

    def test_the_original_issue_newman_fast(self, newman_paper, watts_paper):
        """The motivating example: 'Newman fast' should match Newman's
        paper but not Watts's paper."""
        tokens = parse_search_query("Newman fast")
        assert paper_matches_search(newman_paper, tokens) is True
        assert paper_matches_search(watts_paper, tokens) is False

    def test_cross_field_match(self, newman_paper):
        """'girvan detecting' -- girvan in authors, detecting in title."""
        tokens = parse_search_query("girvan detecting")
        assert paper_matches_search(newman_paper, tokens) is True

    def test_substring_match(self, newman_paper):
        """Tokens should match as substrings, not whole words."""
        tokens = parse_search_query("commun")
        assert paper_matches_search(newman_paper, tokens) is True

    def test_multiple_field_prefixes(self, newman_paper):
        """Multiple field-restricted tokens should all match."""
        tokens = parse_search_query("author:newman year:2004")
        assert paper_matches_search(newman_paper, tokens) is True

    def test_multiple_field_prefixes_one_wrong(self, newman_paper):
        tokens = parse_search_query("author:newman year:2020")
        assert paper_matches_search(newman_paper, tokens) is False
