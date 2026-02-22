"""Tests for HTML preprocessing module."""

from incite.corpus.html_preprocessing import (
    _clean_html_paragraph,
    preprocess_html_text,
)


class TestCleanHtmlParagraph:
    """Tests for _clean_html_paragraph."""

    def test_strips_inline_citations(self):
        assert (
            _clean_html_paragraph("Some text [1] with citations [2,3].")
            == "Some text with citations."
        )

    def test_strips_range_citations(self):
        assert _clean_html_paragraph("Evidence [1-3] supports this.") == "Evidence supports this."

    def test_strips_superscript_digits(self):
        assert _clean_html_paragraph("Some text¹²³ here.") == "Some text here."

    def test_replaces_nbsp(self):
        result = _clean_html_paragraph("Non\u00a0breaking\u200bspace")
        assert "\u00a0" not in result
        assert "\u200b" not in result

    def test_collapses_spaces(self):
        assert _clean_html_paragraph("Too   many    spaces") == "Too many spaces"

    def test_normal_text_unchanged(self):
        text = "This is a normal paragraph with no special characters."
        assert _clean_html_paragraph(text) == text


class TestPreprocessHtmlText:
    """Tests for preprocess_html_text."""

    def test_structured_text_basic(self):
        structured = {
            "sections": [
                {
                    "heading": "Introduction",
                    "paragraphs": [
                        "First paragraph of intro.",
                        "Second paragraph [1] with citation.",
                    ],
                },
                {
                    "heading": "Methods",
                    "paragraphs": ["Methods paragraph here."],
                },
            ]
        }
        paragraphs, sections = preprocess_html_text(None, structured)

        assert len(paragraphs) == 3
        assert sections[0] == "Introduction"
        assert sections[1] == "Introduction"
        assert sections[2] == "Methods"
        # Citation should be stripped
        assert "[1]" not in paragraphs[1]

    def test_structured_text_no_heading(self):
        structured = {
            "sections": [
                {"paragraphs": ["A paragraph without a heading."]},
            ]
        }
        paragraphs, sections = preprocess_html_text(None, structured)

        assert len(paragraphs) == 1
        assert sections[0] is None

    def test_fallback_to_full_text(self):
        full_text = "First paragraph.\n\nSecond paragraph."
        paragraphs, sections = preprocess_html_text(full_text, None)

        assert len(paragraphs) == 2
        assert paragraphs[0] == "First paragraph."
        assert all(s is None for s in sections)

    def test_structured_text_preferred_over_full_text(self):
        """When both are provided, structured_text wins."""
        structured = {"sections": [{"paragraphs": ["From structured."]}]}
        paragraphs, _ = preprocess_html_text("From full text.", structured)

        assert paragraphs == ["From structured."]

    def test_empty_paragraphs_filtered(self):
        structured = {
            "sections": [
                {"paragraphs": ["", "  ", "Real content."]},
            ]
        }
        paragraphs, _ = preprocess_html_text(None, structured)

        assert paragraphs == ["Real content."]

    def test_none_inputs_returns_empty(self):
        paragraphs, sections = preprocess_html_text(None, None)
        assert paragraphs == []
        assert sections == []

    def test_invalid_structured_text_fallback(self):
        """Malformed structured_text falls through to full_text."""
        paragraphs, _ = preprocess_html_text("Fallback text.", {"not_sections": "bad"})
        assert paragraphs == ["Fallback text."]


class TestPaywallDetection:
    """Tests for _is_likely_paywalled."""

    def test_short_text_is_paywalled(self):
        from cloud.library_api import _is_likely_paywalled

        assert _is_likely_paywalled("Short text under 1000 chars.", "An abstract.")

    def test_long_text_not_paywalled(self):
        from cloud.library_api import _is_likely_paywalled

        full_text = "A " * 600  # ~1200 chars
        assert not _is_likely_paywalled(full_text, "Different abstract text.")

    def test_abstract_overlap_detected(self):
        from cloud.library_api import _is_likely_paywalled

        abstract = " ".join(f"word{i}" for i in range(200))
        # full_text is just the abstract with minor padding
        full_text = abstract + " extra"
        assert _is_likely_paywalled(full_text, abstract)
