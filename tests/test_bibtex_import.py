"""Tests for BibTeX import: parsing, Paper conversion, and PDF matching."""

from incite.corpus.enrichment import (
    BibTeXParser,
    bibtex_entries_to_papers,
    match_pdfs_to_papers,
)

SAMPLE_BIB = """
@article{Smith2023,
    title = {Climate change and tropical forests},
    author = {Smith, John and Jones, Alice},
    year = {2023},
    journal = {Nature Climate Change},
    doi = {10.1038/s41558-023-01234-5},
    abstract = {We examine the effects of rising temperatures on tropical forest ecosystems.}
}

@inproceedings{Lee2022Neural,
    title = {Neural approaches to citation recommendation},
    author = {Lee, Wei and Zhang, Ming},
    year = {2022},
    booktitle = {Proceedings of ACL 2022},
    abstract = {We propose a neural model for recommending citations in scientific writing.}
}

@misc{NoTitle2021,
    author = {Nobody, Someone},
    year = {2021}
}

@article{Bare2020,
    title = {A paper with minimal fields},
    year = {2020}
}
"""


class TestBibTeXParserJournal:
    """Test journal/booktitle extraction in _extract_entry."""

    def test_journal_extracted(self):
        entries = BibTeXParser.parse_string(SAMPLE_BIB)
        smith = next(e for e in entries if e["key"] == "Smith2023")
        assert smith["journal"] == "Nature Climate Change"

    def test_booktitle_fallback(self):
        entries = BibTeXParser.parse_string(SAMPLE_BIB)
        lee = next(e for e in entries if e["key"] == "Lee2022Neural")
        assert lee["journal"] == "Proceedings of ACL 2022"

    def test_no_journal(self):
        entries = BibTeXParser.parse_string(SAMPLE_BIB)
        bare = next(e for e in entries if e["key"] == "Bare2020")
        assert bare["journal"] is None

    def test_entries_without_title_skipped(self):
        entries = BibTeXParser.parse_string(SAMPLE_BIB)
        keys = [e["key"] for e in entries]
        assert "NoTitle2021" not in keys


PAPERPILE_BIB = """
@article{Doe2021,
    title = {Paperpile DOI in bdsk-url},
    author = {Doe, Jane},
    year = {2021},
    bdsk-url-1 = {https://doi.org/10.1234/paperpile-test}
}

@article{Roe2022,
    title = {DOI in url field only},
    author = {Roe, Richard},
    year = {2022},
    url = {https://dx.doi.org/10.5678/url-field-test}
}

@article{Both2023,
    title = {Has both doi and bdsk-url},
    author = {Both, Alex},
    year = {2023},
    doi = {10.9999/standard-doi},
    bdsk-url-1 = {https://doi.org/10.9999/should-be-ignored}
}

@article{NoDoi2020,
    title = {No DOI anywhere},
    author = {None, Nobody},
    year = {2020},
    url = {https://example.com/not-a-doi}
}
"""


class TestBibTeXParserDoiFallback:
    """Test fallback DOI extraction from bdsk-url and url fields."""

    def test_doi_from_bdsk_url_1(self):
        entries = BibTeXParser.parse_string(PAPERPILE_BIB)
        doe = next(e for e in entries if e["key"] == "Doe2021")
        assert doe["doi"] == "10.1234/paperpile-test"

    def test_doi_from_url_field(self):
        entries = BibTeXParser.parse_string(PAPERPILE_BIB)
        roe = next(e for e in entries if e["key"] == "Roe2022")
        assert roe["doi"] == "10.5678/url-field-test"

    def test_standard_doi_preferred_over_fallback(self):
        entries = BibTeXParser.parse_string(PAPERPILE_BIB)
        both = next(e for e in entries if e["key"] == "Both2023")
        assert both["doi"] == "10.9999/standard-doi"

    def test_no_doi_when_url_not_doi(self):
        entries = BibTeXParser.parse_string(PAPERPILE_BIB)
        nodoi = next(e for e in entries if e["key"] == "NoDoi2020")
        assert nodoi["doi"] is None

    def test_fallback_doi_flows_to_paper(self):
        entries = BibTeXParser.parse_string(PAPERPILE_BIB)
        papers = bibtex_entries_to_papers(entries)
        doe = next(p for p in papers if p.bibtex_key == "Doe2021")
        assert doe.doi == "10.1234/paperpile-test"


class TestBibtexEntriesToPapers:
    """Test conversion of parsed BibTeX dicts to Paper objects."""

    def test_basic_conversion(self):
        entries = BibTeXParser.parse_string(SAMPLE_BIB)
        papers = bibtex_entries_to_papers(entries)
        assert len(papers) == 3  # NoTitle2021 is skipped by parser

        smith = next(p for p in papers if p.bibtex_key == "Smith2023")
        assert smith.title == "Climate change and tropical forests"
        assert smith.authors == ["John Smith", "Alice Jones"]
        assert smith.year == 2023
        assert smith.journal == "Nature Climate Change"
        assert smith.doi == "10.1038/s41558-023-01234-5"
        assert smith.source_file == "paperpile"
        assert smith.id.startswith("bib_")

    def test_deterministic_ids(self):
        entries = BibTeXParser.parse_string(SAMPLE_BIB)
        papers1 = bibtex_entries_to_papers(entries)
        papers2 = bibtex_entries_to_papers(entries)
        ids1 = sorted(p.id for p in papers1)
        ids2 = sorted(p.id for p in papers2)
        assert ids1 == ids2

    def test_empty_entries(self):
        assert bibtex_entries_to_papers([]) == []

    def test_entry_without_title_skipped(self):
        entries = [{"key": "foo", "authors": ["A"], "year": 2020}]
        assert bibtex_entries_to_papers(entries) == []


class TestMatchPdfsToPapers:
    """Test PDF filename → paper matching heuristics."""

    def setup_method(self):
        entries = BibTeXParser.parse_string(SAMPLE_BIB)
        self.papers = bibtex_entries_to_papers(entries)

    def test_exact_key_match(self):
        matches = match_pdfs_to_papers(["Smith2023.pdf"], self.papers)
        smith = next(p for p in self.papers if p.bibtex_key == "Smith2023")
        assert matches.get(smith.id) == "Smith2023.pdf"

    def test_title_substring_match(self):
        matches = match_pdfs_to_papers(["Climate_change_and_tropical_forests.pdf"], self.papers)
        smith = next(p for p in self.papers if p.bibtex_key == "Smith2023")
        assert smith.id in matches

    def test_author_year_match(self):
        matches = match_pdfs_to_papers(["Lee_2022_neural.pdf"], self.papers)
        lee = next(p for p in self.papers if p.bibtex_key == "Lee2022Neural")
        assert lee.id in matches

    def test_no_match(self):
        matches = match_pdfs_to_papers(["random_file.pdf"], self.papers)
        assert len(matches) == 0

    def test_empty_inputs(self):
        assert match_pdfs_to_papers([], self.papers) == {}
        assert match_pdfs_to_papers(["foo.pdf"], []) == {}

    def test_doi_in_filename_match(self):
        """Pass 2: DOI embedded in filename with _ replacing /."""
        matches = match_pdfs_to_papers(["10.1038_s41558-023-01234-5.pdf"], self.papers)
        smith = next(p for p in self.papers if p.bibtex_key == "Smith2023")
        assert matches.get(smith.id) == "10.1038_s41558-023-01234-5.pdf"

    def test_fuzzy_jaccard_match(self):
        """Pass 4: Jaccard similarity catches reordered/partial title words."""
        from incite.models import Paper

        papers = [
            Paper(
                id="test_1",
                title="Deep learning approaches for natural language processing tasks",
                authors=["Author One"],
            )
        ]
        # Reordered words in filename
        matches = match_pdfs_to_papers(
            ["natural_language_processing_deep_learning_approaches.pdf"], papers
        )
        assert "test_1" in matches

    def test_jaccard_no_false_positive_on_short_overlap(self):
        """Jaccard should not match when only 1-2 words overlap."""
        from incite.models import Paper

        papers = [
            Paper(
                id="test_1",
                title="Deep learning for NLP",
                authors=["Author"],
            )
        ]
        # Only "deep" and "learning" would overlap — fewer than 3 words
        matches = match_pdfs_to_papers(["deep_learning_completely_different_topic.pdf"], papers)
        assert "test_1" not in matches


class TestWordSet:
    """Test the _word_set helper."""

    def test_basic(self):
        from incite.corpus.enrichment import _word_set

        result = _word_set("Deep Learning for NLP tasks 2023")
        assert "deep" in result
        assert "learning" in result
        assert "tasks" in result
        assert "2023" in result
        # "for" and "nlp" are exactly 3 chars, so they ARE included (>= 3)
        assert "for" in result
        assert "nlp" in result

    def test_empty(self):
        from incite.corpus.enrichment import _word_set

        assert _word_set("") == set()
        assert _word_set("a b") == set()
