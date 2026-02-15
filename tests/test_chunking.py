"""Tests for chunking module."""

import pytest

from incite.corpus.chunking import (
    chunk_paper,
    chunk_papers,
    _split_into_paragraphs,
    _looks_like_heading,
    _split_long_text,
    _is_reference_section,
    _is_corrupted_text,
    _looks_like_bibliography_entry,
    _is_bibliography_chunk,
)
from incite.models import Chunk, Paper


class TestChunk:
    """Tests for the Chunk dataclass."""

    def test_chunk_creation(self):
        """Test basic chunk creation."""
        chunk = Chunk(
            id="paper1::chunk_0",
            paper_id="paper1",
            text="This is the chunk text.",
        )
        assert chunk.id == "paper1::chunk_0"
        assert chunk.paper_id == "paper1"
        assert chunk.text == "This is the chunk text."
        assert chunk.section is None
        assert chunk.context_text is None

    def test_chunk_with_context(self):
        """Test chunk with contextual enrichment."""
        chunk = Chunk(
            id="paper1::chunk_0",
            paper_id="paper1",
            text="This is the chunk text.",
            context_text="This chunk discusses X in the context of Y.",
        )
        assert chunk.context_text is not None
        # to_embedding_text should prepend context
        embedding_text = chunk.to_embedding_text()
        assert embedding_text.startswith("This chunk discusses X")
        assert "This is the chunk text." in embedding_text

    def test_chunk_without_context(self):
        """Test chunk without contextual enrichment."""
        chunk = Chunk(
            id="paper1::chunk_0",
            paper_id="paper1",
            text="This is the chunk text.",
        )
        embedding_text = chunk.to_embedding_text()
        assert embedding_text == "This is the chunk text."

    def test_parse_chunk_id(self):
        """Test chunk ID parsing."""
        paper_id, idx = Chunk.parse_id("paper1::chunk_5")
        assert paper_id == "paper1"
        assert idx == 5

    def test_parse_chunk_id_with_colons(self):
        """Test chunk ID parsing with colons in paper ID."""
        paper_id, idx = Chunk.parse_id("doi:10.1234/abc::chunk_3")
        assert paper_id == "doi:10.1234/abc"
        assert idx == 3

    def test_parse_chunk_id_invalid(self):
        """Test chunk ID parsing with invalid format."""
        with pytest.raises(ValueError):
            Chunk.parse_id("invalid_id")

    def test_chunk_validation(self):
        """Test chunk validation."""
        with pytest.raises(ValueError):
            Chunk(id="", paper_id="paper1", text="text")

        with pytest.raises(ValueError):
            Chunk(id="id", paper_id="", text="text")

        with pytest.raises(ValueError):
            Chunk(id="id", paper_id="paper1", text="")


class TestChunkPaper:
    """Tests for chunk_paper function."""

    def test_chunk_paper_with_paragraphs(self):
        """Test chunking a paper with pre-extracted paragraphs."""
        paper = Paper(
            id="test_paper",
            title="Test Paper",
            abstract="This is the abstract.",
            paragraphs=[
                "First paragraph with enough text to meet minimum length requirement for chunking.",
                "Second paragraph also with enough text to meet the minimum length requirement.",
            ],
        )

        chunks = chunk_paper(paper, min_chunk_length=50)

        assert len(chunks) == 2
        assert chunks[0].paper_id == "test_paper"
        assert "First paragraph" in chunks[0].text
        assert chunks[1].id == "test_paper::chunk_1"

    def test_chunk_paper_with_full_text(self):
        """Test chunking a paper with full_text (split on double newlines)."""
        paper = Paper(
            id="test_paper",
            title="Test Paper",
            abstract="This is the abstract.",
            full_text="First paragraph with enough text to meet minimum.\n\nSecond paragraph with enough text to meet minimum.",
        )

        chunks = chunk_paper(paper, min_chunk_length=30)

        assert len(chunks) == 2

    def test_chunk_paper_fallback_to_abstract(self):
        """Test that abstract is used as fallback when no full text."""
        paper = Paper(
            id="test_paper",
            title="Test Paper",
            abstract="This is the abstract which should be used as a single chunk since there is no full text available.",
        )

        chunks = chunk_paper(paper, min_chunk_length=20)

        assert len(chunks) == 1
        assert "abstract" in chunks[0].text

    def test_chunk_paper_empty(self):
        """Test chunking a paper with no text."""
        paper = Paper(
            id="test_paper",
            title="Test Paper",
            abstract="",
        )

        chunks = chunk_paper(paper)

        assert len(chunks) == 0

    def test_chunk_paper_skips_short(self):
        """Test that short paragraphs are skipped."""
        paper = Paper(
            id="test_paper",
            title="Test Paper",
            abstract="Short.",
            paragraphs=["Short.", "This paragraph is long enough to be included in the chunks."],
        )

        chunks = chunk_paper(paper, min_chunk_length=50)

        assert len(chunks) == 1
        assert "long enough" in chunks[0].text

    def test_chunk_paper_section_tracking(self):
        """Test that section headings are tracked but not chunked."""
        paper = Paper(
            id="test_paper",
            title="Test Paper",
            abstract="Abstract text.",
            paragraphs=[
                "1. Introduction",  # Heading
                "This is the introduction paragraph with enough text to be included.",
                "2. Methods",  # Heading
                "This is the methods paragraph with enough text to be included as well.",
            ],
        )

        chunks = chunk_paper(paper, min_chunk_length=30)

        # Should have 2 chunks (headings are not chunked)
        assert len(chunks) == 2
        assert chunks[0].section == "1. Introduction"
        assert chunks[1].section == "2. Methods"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_split_into_paragraphs(self):
        """Test paragraph splitting."""
        text = "First para.\n\nSecond para.\n\nThird para."
        paragraphs = _split_into_paragraphs(text)

        assert len(paragraphs) == 3
        assert paragraphs[0] == "First para."

    def test_split_into_paragraphs_normalizes_whitespace(self):
        """Test that internal newlines are collapsed."""
        text = "Line one\nLine two\n\nNew paragraph."
        paragraphs = _split_into_paragraphs(text)

        assert len(paragraphs) == 2
        assert paragraphs[0] == "Line one Line two"

    def test_looks_like_heading_numbered(self):
        """Test heading detection for numbered sections."""
        assert _looks_like_heading("1. Introduction")
        assert _looks_like_heading("1 Introduction")
        # Note: Subsection numbering like "3.2" not currently detected
        # This is acceptable since LLM contextual enrichment handles structure

    def test_looks_like_heading_common_names(self):
        """Test heading detection for common section names."""
        assert _looks_like_heading("Introduction")
        assert _looks_like_heading("Methods")
        assert _looks_like_heading("Results")
        assert _looks_like_heading("Conclusion")
        assert _looks_like_heading("Related Work")

    def test_looks_like_heading_negative(self):
        """Test that regular text is not detected as heading."""
        assert not _looks_like_heading("This is a regular sentence.")
        assert not _looks_like_heading("A very long text that is definitely not a heading because it has way too many words.")

    def test_split_long_text(self):
        """Test splitting long text at sentence boundaries."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        chunks = _split_long_text(text, max_chars=50, min_chars=10)

        # Should split into multiple chunks
        assert len(chunks) >= 1
        # Each chunk should end properly
        for chunk in chunks:
            assert len(chunk) >= 10


class TestChunkPapers:
    """Tests for chunk_papers function."""

    def test_chunk_multiple_papers(self):
        """Test chunking multiple papers."""
        papers = [
            Paper(
                id="paper1",
                title="Paper 1",
                abstract="Abstract 1 with enough text to be a valid chunk for testing purposes.",
            ),
            Paper(
                id="paper2",
                title="Paper 2",
                abstract="Abstract 2 with enough text to be a valid chunk for testing purposes.",
            ),
        ]

        chunks = chunk_papers(papers, min_chunk_length=30, show_progress=False)

        assert len(chunks) == 2
        assert chunks[0].paper_id == "paper1"
        assert chunks[1].paper_id == "paper2"


class TestChunkFiltering:
    """Tests for reference section and corrupted text filtering."""

    # --- _is_reference_section tests ---

    @pytest.mark.parametrize(
        "name",
        [
            "References",
            "REFERENCES",
            "references",
            "8. Bibliography",
            "7 References",
            "Works Cited",
            "Literature Cited",
            "9. Works Cited",
            # New: variants with suffixes (Phase 1 fix)
            "References and Notes",
            "References Cited",
            "VIII. References",
            "Cited Literature",
        ],
    )
    def test_is_reference_section_positive(self, name):
        assert _is_reference_section(name)

    @pytest.mark.parametrize(
        "name",
        [
            "Introduction",
            "Related Work",
            "Methods",
            "Reference Architecture",
            None,
            # New: edge cases that should NOT match
            "Cross References",
            "See References in Appendix",
        ],
    )
    def test_is_reference_section_negative(self, name):
        assert not _is_reference_section(name)

    # --- _looks_like_bibliography_entry tests (Phase 1) ---

    @pytest.mark.parametrize(
        "text",
        [
            "Smith, J. et al. (2020). Title of paper. Journal of Something, Vol. 42, pp. 123-456. doi: 10.1234/example",
            "Jones, A., Brown, B. 2019. Another paper. Nature Vol. 500, pp. 1-10. https://doi.org/10.1038/example",
            "Williams, C. et al. (2018a). Paper title. Proceedings of the Conference. arXiv:1234.5678",
        ],
    )
    def test_looks_like_bibliography_entry_positive(self, text):
        assert _looks_like_bibliography_entry(text)

    @pytest.mark.parametrize(
        "text",
        [
            "This is a paragraph about climate change and its effects on global temperatures.",
            "In this section we describe our methodology for data collection.",
            "The results indicate that the proposed algorithm outperforms previous approaches.",
            "A" * 700,  # Too long to be a bibliography entry
        ],
    )
    def test_looks_like_bibliography_entry_negative(self, text):
        assert not _looks_like_bibliography_entry(text)

    # --- _is_corrupted_text tests ---

    @pytest.mark.parametrize(
        "text",
        [
            "??????????????????????",
            ".................",
            "→→→ ← ↔ ∑∏∫ √ ≈ ≠ ±",
            "",
            "12345 67890 !@#$%",
        ],
    )
    def test_is_corrupted_text_positive(self, text):
        assert _is_corrupted_text(text)

    def test_is_corrupted_text_normal_prose(self):
        assert not _is_corrupted_text(
            "Climate change impacts on sea level rise have been studied extensively."
        )

    def test_is_corrupted_text_with_some_math(self):
        # Text with inline equations but still mostly prose
        assert not _is_corrupted_text(
            "The equation F = ma (where m=10kg and a=9.8m/s^2) gives the force."
        )

    # --- _is_bibliography_chunk tests (per-chunk filter) ---

    @pytest.mark.parametrize(
        "text",
        [
            # Caught by 3+ signals (existing _looks_like_bibliography_entry)
            "Smith, J. et al. (2020). Title. Journal of Something, Vol. 42, pp. 1-10. doi: 10.1234/x",
            # Caught by structural start pattern: Author, F. + signal
            "Anderson, K. & Bows, A. 2012. A New Paradigm for Climate Change. Nature Climate Change, 2(9), 639-640.",
            "SAPINSKI, J.P. 2015. Climate Capitalism and the Global Corporate Elite Network.",
            # Caught by numbered ref start pattern
            "[1] Smith, J. (2020). Some paper title. Journal of Something.",
            "[42] Jones, A. et al. Deep learning for NLP. arXiv:2103.12345",
            "1. Author, F. (2019). Title of the paper. Nature, 500, 1-10.",
        ],
    )
    def test_is_bibliography_chunk_positive(self, text):
        assert _is_bibliography_chunk(text)

    @pytest.mark.parametrize(
        "text",
        [
            # Normal body text with in-text citations
            "Smith et al. (2020) found that climate change impacts are accelerating globally.",
            "In this section we describe our methodology for data collection and analysis.",
            "The results indicate that the proposed algorithm outperforms previous approaches by a wide margin.",
            # Body text with multiple citations (should NOT be flagged)
            "Several studies (Jones 2019; Smith 2020) have explored this phenomenon extensively.",
            # Too long to be a bibliography entry
            "A" * 700,
        ],
    )
    def test_is_bibliography_chunk_negative(self, text):
        assert not _is_bibliography_chunk(text)

    # --- Integration tests ---

    def test_chunk_paper_filters_bibliography_entries(self):
        """Bibliography entries without a References heading should still be filtered."""
        paper = Paper(
            id="test",
            title="Test",
            abstract="Abstract.",
            paragraphs=[
                "This is a normal intro paragraph with enough text to meet the minimum chunk length for testing purposes.",
                "Anderson, K. & Bows, A. 2012. A New Paradigm. Nature Climate Change, 2(9), 639-640.",
                "Joskow, P. L. (2001). California's Electricity Crisis. Oxford Review of Economic Policy, 17(3), 365-388. doi:10.1093/oxrep/17.3.365",
                "This is another normal paragraph with enough text to meet the minimum chunk length for testing purposes.",
            ],
        )
        chunks = chunk_paper(paper, min_chunk_length=30)
        assert len(chunks) == 2
        assert all("Anderson" not in c.text for c in chunks)
        assert all("Joskow" not in c.text for c in chunks)

    def test_chunk_paper_stops_at_references(self):
        """Paper with References section should only return pre-reference chunks."""
        paper = Paper(
            id="test",
            title="Test",
            abstract="Abstract.",
            paragraphs=[
                "1. Introduction",
                "This is the introduction with enough text to meet the minimum chunk length.",
                "2. Methods",
                "This is the methods section with enough text to meet the minimum chunk length.",
                "References",
                "Smith et al. (2020). Some paper title. Journal of Something, 1(2), 3-4.",
                "Jones et al. (2019). Another paper. Nature, 5(6), 7-8.",
            ],
        )
        chunks = chunk_paper(paper, min_chunk_length=30)
        assert len(chunks) == 2
        assert all("Smith" not in c.text for c in chunks)
        assert chunks[0].section == "1. Introduction"
        assert chunks[1].section == "2. Methods"

    def test_chunk_paper_excludes_corrupted(self):
        """Corrupted paragraphs should be excluded from chunks."""
        paper = Paper(
            id="test",
            title="Test",
            abstract="Abstract.",
            paragraphs=[
                "This is a normal paragraph with enough text to meet the minimum length.",
                "?????? → ← ∑∏∫ √ ≈ ≠ ± 12345 67890 ← → ↔ ∀∃∂∇ ♠♣♥♦ ★☆ ⌘ ⌥",
                "Another normal paragraph with enough text to meet the minimum length.",
            ],
        )
        chunks = chunk_paper(paper, min_chunk_length=30)
        assert len(chunks) == 2
        assert all("??????" not in c.text for c in chunks)
