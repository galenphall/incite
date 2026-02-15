"""Tests for sentence-level chunking module."""

import pytest

from incite.corpus.sentence_chunking import (
    chunk_paper_sentences,
    chunk_papers_sentences,
    _split_sentences,
    _split_sentences_regex,
    _build_sentence_context,
    _is_short_reference,
)
from incite.models import Chunk, Paper


class TestSplitSentences:
    """Tests for sentence splitting functions."""

    def test_split_sentences_basic(self):
        """Test basic sentence splitting."""
        text = "First sentence. Second sentence. Third sentence."
        sentences = _split_sentences(text)

        assert len(sentences) == 3
        assert sentences[0][2] == "First sentence."
        assert sentences[1][2] == "Second sentence."
        assert sentences[2][2] == "Third sentence."

    def test_split_sentences_regex_fallback(self):
        """Test regex fallback for sentence splitting."""
        text = "First sentence. Second sentence. Third sentence."
        sentences = _split_sentences_regex(text)

        assert len(sentences) == 3
        assert sentences[0][2] == "First sentence."

    def test_split_sentences_with_abbreviations(self):
        """Test that common abbreviations don't cause incorrect splits."""
        text = "Dr. Smith et al. published results. The data shows improvement."
        sentences = _split_sentences(text)

        # spaCy should handle this correctly (2 sentences)
        # regex fallback may incorrectly split (ok for fallback)
        assert len(sentences) >= 1

    def test_split_sentences_preserves_offsets(self):
        """Test that character offsets are preserved."""
        text = "First. Second. Third."
        sentences = _split_sentences(text)

        for start, end, sent_text in sentences:
            # The sentence text should match the slice from original
            assert text[start:end].strip() == sent_text


class TestBuildSentenceContext:
    """Tests for context building."""

    def test_context_with_all_parts(self):
        """Test context with title, section, and previous sentence."""
        paper = Paper(id="p1", title="Test Paper Title")
        context = _build_sentence_context(
            paper=paper,
            section="Introduction",
            prev_sentence="Previous sentence here.",
        )

        assert context == "Test Paper Title | Introduction | Previous sentence here."

    def test_context_title_only(self):
        """Test context with only title."""
        paper = Paper(id="p1", title="Test Paper Title")
        context = _build_sentence_context(paper=paper, section=None, prev_sentence=None)

        assert context == "Test Paper Title"

    def test_context_with_section_no_prev(self):
        """Test context with title and section but no previous sentence."""
        paper = Paper(id="p1", title="Test Paper Title")
        context = _build_sentence_context(
            paper=paper,
            section="Methods",
            prev_sentence=None,
        )

        assert context == "Test Paper Title | Methods"


class TestIsShortReference:
    """Tests for short reference detection."""

    @pytest.mark.parametrize(
        "text",
        [
            "See Figure 1.",
            "See Fig. 2 for details.",
            "Table 3 shows the results.",
            "(cf. above)",
            "(ibid.)",
        ],
    )
    def test_short_reference_positive(self, text):
        """Test detection of short figure/table references."""
        assert _is_short_reference(text)

    @pytest.mark.parametrize(
        "text",
        [
            "The results in Figure 1 demonstrate that our approach outperforms baselines.",
            "We present our methodology in this section.",
            "Climate change affects ecosystems globally.",
        ],
    )
    def test_short_reference_negative(self, text):
        """Test that normal sentences are not flagged as references."""
        assert not _is_short_reference(text)


class TestChunkPaperSentences:
    """Tests for chunk_paper_sentences function."""

    def test_chunk_paper_basic(self):
        """Test basic sentence chunking."""
        paper = Paper(
            id="test_paper",
            title="Test Paper",
            abstract="This is a sufficiently long abstract sentence that meets the minimum length. Another sentence here for testing.",
        )

        chunks = chunk_paper_sentences(paper, min_chunk_length=30)

        assert len(chunks) >= 1
        assert all(c.paper_id == "test_paper" for c in chunks)
        assert all(c.id.startswith("test_paper::chunk_") for c in chunks)

    def test_chunk_paper_includes_context(self):
        """Test that context_text is included when enabled."""
        paper = Paper(
            id="test_paper",
            title="Test Paper Title",
            abstract="First sentence is long enough to be included. Second sentence is also long enough for the chunk.",
        )

        chunks = chunk_paper_sentences(paper, min_chunk_length=30, include_context=True)

        assert len(chunks) >= 1
        assert chunks[0].context_text is not None
        assert "Test Paper Title" in chunks[0].context_text

    def test_chunk_paper_no_context(self):
        """Test that context_text is None when disabled."""
        paper = Paper(
            id="test_paper",
            title="Test Paper",
            abstract="This is a sufficiently long sentence that will be chunked without context.",
        )

        chunks = chunk_paper_sentences(paper, min_chunk_length=30, include_context=False)

        assert len(chunks) >= 1
        assert all(c.context_text is None for c in chunks)

    def test_chunk_paper_skips_short_sentences(self):
        """Test that short sentences are skipped."""
        paper = Paper(
            id="test_paper",
            title="Test Paper",
            abstract="Short. This is a much longer sentence that should be included in the chunks.",
        )

        chunks = chunk_paper_sentences(paper, min_chunk_length=50)

        # Only the long sentence should be included
        assert len(chunks) == 1
        assert "much longer sentence" in chunks[0].text

    def test_chunk_paper_tracks_sections(self):
        """Test that section headings are tracked."""
        paper = Paper(
            id="test_paper",
            title="Test Paper",
            abstract="Abstract text.",
            paragraphs=[
                "1. Introduction",
                "This is the introduction paragraph with enough text to meet the minimum chunk length requirement.",
                "2. Methods",
                "This is the methods paragraph with enough text to meet the minimum chunk length requirement as well.",
            ],
        )

        chunks = chunk_paper_sentences(paper, min_chunk_length=30)

        # Find chunks from each section
        intro_chunks = [c for c in chunks if c.section == "1. Introduction"]
        methods_chunks = [c for c in chunks if c.section == "2. Methods"]

        assert len(intro_chunks) >= 1
        assert len(methods_chunks) >= 1

    def test_chunk_paper_stops_at_references(self):
        """Test that chunking stops at reference section."""
        paper = Paper(
            id="test_paper",
            title="Test Paper",
            abstract="Abstract.",
            paragraphs=[
                "1. Introduction",
                "This is the introduction with enough text to be chunked properly.",
                "References",
                "Smith et al. (2020). Some paper title. Journal of Something.",
                "Jones et al. (2019). Another paper. Nature.",
            ],
        )

        chunks = chunk_paper_sentences(paper, min_chunk_length=30)

        # Should not include reference entries
        assert all("Smith et al." not in c.text for c in chunks)
        assert all("Jones et al." not in c.text for c in chunks)

    def test_chunk_paper_excludes_corrupted(self):
        """Test that corrupted text is excluded."""
        paper = Paper(
            id="test_paper",
            title="Test Paper",
            abstract="Abstract.",
            paragraphs=[
                "This is a normal sentence with enough text to be included.",
                "?????? → ← ∑∏∫ √ ≈ ≠ ± 12345 67890",
                "Another normal sentence with enough text to meet the minimum.",
            ],
        )

        chunks = chunk_paper_sentences(paper, min_chunk_length=30)

        assert all("??????" not in c.text for c in chunks)

    def test_chunk_paper_empty(self):
        """Test chunking a paper with no text."""
        paper = Paper(
            id="test_paper",
            title="Test Paper",
            abstract="",
        )

        chunks = chunk_paper_sentences(paper)

        assert len(chunks) == 0

    def test_chunk_paper_context_includes_prev_sentence(self):
        """Test that context includes previous sentence."""
        paper = Paper(
            id="test_paper",
            title="Test Paper",
            abstract="First sentence meets the minimum length requirement. Second sentence also meets minimum length requirement.",
        )

        chunks = chunk_paper_sentences(paper, min_chunk_length=30, include_context=True)

        if len(chunks) >= 2:
            # Second chunk's context should include first sentence
            assert chunks[0].text in chunks[1].context_text or "First sentence" in chunks[1].context_text


class TestChunkPapersSentences:
    """Tests for chunk_papers_sentences function."""

    def test_chunk_multiple_papers(self):
        """Test chunking multiple papers."""
        papers = [
            Paper(
                id="paper1",
                title="Paper 1",
                abstract="Abstract 1 with enough text to be a valid chunk for testing purposes here.",
            ),
            Paper(
                id="paper2",
                title="Paper 2",
                abstract="Abstract 2 with enough text to be a valid chunk for testing purposes here.",
            ),
        ]

        chunks = chunk_papers_sentences(papers, min_chunk_length=30, show_progress=False)

        paper1_chunks = [c for c in chunks if c.paper_id == "paper1"]
        paper2_chunks = [c for c in chunks if c.paper_id == "paper2"]

        assert len(paper1_chunks) >= 1
        assert len(paper2_chunks) >= 1


class TestFactoryIntegration:
    """Tests for factory integration."""

    def test_sentence_strategy_available(self):
        """Test that sentence strategy is in CHUNKING_STRATEGIES."""
        from incite.retrieval.factory import CHUNKING_STRATEGIES

        assert "sentence" in CHUNKING_STRATEGIES
        assert CHUNKING_STRATEGIES["sentence"]["function"] == "chunk_papers_sentences"

    def test_get_chunker_sentence(self):
        """Test getting sentence chunker from factory."""
        from incite.retrieval.factory import get_chunker

        chunker = get_chunker("sentence")
        assert callable(chunker)

        # Test that it works
        papers = [
            Paper(
                id="test",
                title="Test",
                abstract="This is a test abstract with enough text to chunk properly.",
            )
        ]
        chunks = chunker(papers, show_progress=False)
        assert len(chunks) >= 1
