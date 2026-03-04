"""Tests for citation export formats (BibTeX, RIS, Bibliography, CSV, JSON)."""

from __future__ import annotations

import csv
import io
import json as json_mod

from incite.export import FORMATS, ExportFormat
from incite.export.bibtex import BibTeXFormat
from incite.export.ris import RISFormat
from incite.models import Paper


def _make_paper(**overrides) -> Paper:
    """Create a Paper with sensible defaults."""
    defaults = {
        "id": "test_001",
        "title": "A Study of Testing",
        "abstract": "We test things.",
        "authors": ["Smith, John", "Jones, Alice"],
        "year": 2024,
        "doi": "10.1234/test.2024",
        "journal": "Journal of Tests",
        "bibtex_key": "",
    }
    defaults.update(overrides)
    return Paper(**defaults)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_bibtex_registered(self):
        assert "bibtex" in FORMATS

    def test_ris_registered(self):
        assert "ris" in FORMATS

    def test_formats_implement_protocol(self):
        assert isinstance(FORMATS["bibtex"], ExportFormat)
        assert isinstance(FORMATS["ris"], ExportFormat)

    def test_format_metadata(self):
        bib = FORMATS["bibtex"]
        assert bib.file_extension == ".bib"
        assert bib.mime_type == "application/x-bibtex"

        ris = FORMATS["ris"]
        assert ris.file_extension == ".ris"
        assert ris.mime_type == "application/x-research-info-systems"


# ---------------------------------------------------------------------------
# BibTeX
# ---------------------------------------------------------------------------


class TestBibTeX:
    def test_single_export(self):
        fmt = BibTeXFormat()
        paper = _make_paper(bibtex_key="smith2024study")
        result = fmt.export_single(paper)

        assert "@article{smith2024study," in result
        assert "title = {A Study of Testing}" in result
        assert "author = {Smith, John and Jones, Alice}" in result
        assert "year = {2024}" in result
        assert "doi = {10.1234/test.2024}" in result
        assert "journal = {Journal of Tests}" in result

    def test_multi_export(self):
        fmt = BibTeXFormat()
        papers = [_make_paper(id="p1", bibtex_key="a"), _make_paper(id="p2", bibtex_key="b")]
        result = fmt.export_items(papers)

        assert result.count("@article{") == 2
        assert "\n\n" in result  # entries separated by blank line

    def test_generated_key_when_no_bibtex_key(self):
        fmt = BibTeXFormat()
        paper = _make_paper(bibtex_key="")
        result = fmt.export_single(paper)

        # Key should be generated from author+year+title word
        assert "@article{smith2024study," in result

    def test_latex_special_chars_escaped(self):
        fmt = BibTeXFormat()
        paper = _make_paper(title="10% of A & B: C#D")
        result = fmt.export_single(paper)

        assert r"10\% of A \& B: C\#D" in result

    def test_missing_optional_fields(self):
        fmt = BibTeXFormat()
        paper = _make_paper(doi=None, journal=None, abstract=None, authors=[])
        result = fmt.export_single(paper)

        assert "doi" not in result
        assert "journal" not in result
        assert "abstract" not in result
        assert "author" not in result
        assert "title = {A Study of Testing}" in result

    def test_braces_escaped(self):
        fmt = BibTeXFormat()
        paper = _make_paper(title="Graph {Neural} Networks")
        result = fmt.export_single(paper)

        assert r"Graph \{Neural\} Networks" in result


# ---------------------------------------------------------------------------
# RIS
# ---------------------------------------------------------------------------


class TestRIS:
    def test_single_export(self):
        fmt = RISFormat()
        paper = _make_paper()
        result = fmt.export_single(paper)

        assert "TY  - JOUR" in result
        assert "TI  - A Study of Testing" in result
        assert "AU  - Smith, John" in result
        assert "AU  - Jones, Alice" in result
        assert "PY  - 2024" in result
        assert "DO  - 10.1234/test.2024" in result
        assert "JO  - Journal of Tests" in result
        assert "AB  - We test things." in result
        assert "ER  - " in result

    def test_multi_export(self):
        fmt = RISFormat()
        papers = [_make_paper(id="p1"), _make_paper(id="p2")]
        result = fmt.export_items(papers)

        assert result.count("TY  - JOUR") == 2
        assert result.count("ER  - ") == 2

    def test_missing_optional_fields(self):
        fmt = RISFormat()
        paper = _make_paper(doi=None, journal=None, abstract=None)
        result = fmt.export_single(paper)

        assert "DO  -" not in result
        assert "JO  -" not in result
        assert "AB  -" not in result
        assert "TY  - JOUR" in result
        assert "ER  - " in result

    def test_record_starts_with_ty_ends_with_er(self):
        fmt = RISFormat()
        paper = _make_paper()
        result = fmt.export_single(paper)
        lines = result.split("\n")

        assert lines[0] == "TY  - JOUR"
        assert lines[-1].startswith("ER  -")

    def test_multiple_authors(self):
        fmt = RISFormat()
        paper = _make_paper(authors=["A", "B", "C"])
        result = fmt.export_single(paper)

        assert result.count("AU  - ") == 3


# ---------------------------------------------------------------------------
# Bibliography (CSL-formatted)
# ---------------------------------------------------------------------------


class TestBibliography:
    def test_apa_registered(self):
        assert "apa" in FORMATS

    def test_mla_registered(self):
        assert "mla" in FORMATS

    def test_chicago_registered(self):
        assert "chicago" in FORMATS

    def test_harvard_registered(self):
        assert "harvard" in FORMATS

    def test_apa_format_metadata(self):
        fmt = FORMATS["apa"]
        assert fmt.file_extension == ".txt"
        assert fmt.mime_type == "text/plain"

    def test_apa_single_export_contains_author_and_year(self):
        fmt = FORMATS["apa"]
        paper = _make_paper()
        result = fmt.export_single(paper)
        assert "Smith" in result
        assert "2024" in result
        assert "A Study of Testing" in result

    def test_apa_multi_export(self):
        fmt = FORMATS["apa"]
        papers = [
            _make_paper(id="p1", title="First Paper"),
            _make_paper(id="p2", title="Second Paper"),
        ]
        result = fmt.export_items(papers)
        assert "First Paper" in result
        assert "Second Paper" in result

    def test_bibliography_missing_fields(self):
        fmt = FORMATS["apa"]
        paper = _make_paper(doi=None, journal=None, authors=[], year=None)
        result = fmt.export_single(paper)
        # Should still produce something with the title
        assert "A Study of Testing" in result

    def test_implements_protocol(self):
        for key in ("apa", "mla", "chicago", "harvard"):
            assert isinstance(FORMATS[key], ExportFormat)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


class TestCSV:
    def test_csv_registered(self):
        assert "csv" in FORMATS

    def test_csv_format_metadata(self):
        fmt = FORMATS["csv"]
        assert fmt.file_extension == ".csv"
        assert fmt.mime_type == "text/csv"

    def test_csv_single_export(self):
        fmt = FORMATS["csv"]
        paper = _make_paper()
        result = fmt.export_single(paper)
        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["title"] == "A Study of Testing"
        assert rows[0]["year"] == "2024"

    def test_csv_multi_export(self):
        fmt = FORMATS["csv"]
        papers = [_make_paper(id="p1"), _make_paper(id="p2")]
        result = fmt.export_items(papers)
        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) == 2

    def test_csv_with_recommendation_data(self):
        from incite.export.csv_export import CSVFormat

        fmt = CSVFormat()
        paper = _make_paper()
        result = fmt.export_single(paper, confidence=0.85, matched_paragraph="Evidence text.")
        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)
        assert rows[0]["confidence"] == "0.85"
        assert rows[0]["matched_paragraph"] == "Evidence text."


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


class TestJSON:
    def test_json_registered(self):
        assert "json" in FORMATS

    def test_json_format_metadata(self):
        fmt = FORMATS["json"]
        assert fmt.file_extension == ".json"
        assert fmt.mime_type == "application/json"

    def test_json_single_export(self):
        fmt = FORMATS["json"]
        paper = _make_paper()
        result = fmt.export_single(paper)
        data = json_mod.loads(result)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["title"] == "A Study of Testing"

    def test_json_multi_export(self):
        fmt = FORMATS["json"]
        papers = [_make_paper(id="p1"), _make_paper(id="p2")]
        result = fmt.export_items(papers)
        data = json_mod.loads(result)
        assert len(data) == 2

    def test_json_with_recommendation_data(self):
        from incite.export.json_export import JSONFormat

        fmt = JSONFormat()
        paper = _make_paper()
        result = fmt.export_single(paper, confidence=0.85, matched_paragraph="Evidence.")
        data = json_mod.loads(result)
        assert data[0]["confidence"] == 0.85
        assert data[0]["matched_paragraph"] == "Evidence."
