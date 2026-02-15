"""Tests for core data models."""

import pytest

from incite.models import (
    Chunk,
    CitationContext,
    EvaluationResult,
    Paper,
    RetrievalResult,
    clean_citation_markers,
    format_author_string,
    format_paper_embedding_text,
    format_paper_metadata_prefix,
    format_passage_embedding_text,
)


class TestPaper:
    def test_create_paper(self):
        paper = Paper(
            id="abc123",
            title="Test Paper",
            abstract="This is a test abstract.",
            authors=["Smith, John", "Doe, Jane"],
            year=2023,
        )
        assert paper.id == "abc123"
        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2

    def test_paper_requires_id(self):
        with pytest.raises(ValueError, match="must have an id"):
            Paper(id="", title="Test", abstract="")

    def test_paper_requires_title(self):
        with pytest.raises(ValueError, match="must have a title"):
            Paper(id="abc", title="", abstract="")

    def test_paper_abstract_defaults_empty(self):
        paper = Paper(id="1", title="Test")
        assert paper.abstract == ""

    def test_paper_all_fields(self):
        paper = Paper(
            id="abc",
            title="Title",
            abstract="Abstract",
            authors=["A", "B"],
            year=2024,
            doi="10.1234/test",
            bibtex_key="smith2024",
            journal="Nature",
            full_text="Full text",
            paragraphs=["p1", "p2"],
            source_file="/path/to/file.pdf",
            llm_description="LLM description",
        )
        assert paper.doi == "10.1234/test"
        assert paper.bibtex_key == "smith2024"
        assert paper.journal == "Nature"
        assert paper.source_file == "/path/to/file.pdf"
        assert paper.llm_description == "LLM description"
        assert len(paper.paragraphs) == 2

    def test_author_lastnames(self):
        paper = Paper(
            id="1",
            title="Test",
            abstract="",
            authors=["Smith, John", "Jane Doe", "Bob"],
        )
        # "Bob" is 3 chars so included, but single-char names filtered
        assert paper.author_lastnames == ["Smith", "Doe", "Bob"]

    def test_author_lastnames_filters_single_char(self):
        paper = Paper(
            id="1",
            title="Test",
            authors=["J", "Li Wei", "A B"],
        )
        lastnames = paper.author_lastnames
        # "J" is single char, should be filtered out
        assert "J" not in lastnames
        # "Wei" and "B" — "B" is single char, filtered
        assert "Wei" in lastnames
        assert "B" not in lastnames

    def test_to_embedding_text(self):
        paper = Paper(
            id="1",
            title="Test Title",
            abstract="Test abstract here.",
        )
        # Default includes metadata (but paper has none) and abstract
        assert paper.to_embedding_text() == "Test Title. Test abstract here."
        assert paper.to_embedding_text(include_abstract=False) == "Test Title"
        assert paper.to_embedding_text(include_metadata=False) == "Test Title. Test abstract here."

    def test_to_embedding_text_with_metadata(self):
        paper = Paper(
            id="1",
            title="Test Title",
            abstract="Test abstract here.",
            authors=["John Smith", "Jane Doe"],
            year=2023,
            journal="Nature",
        )
        # With metadata: "Title. Authors. Year. Journal. Abstract"
        expected = "Test Title. Smith and Doe. 2023. Nature. Test abstract here."
        assert paper.to_embedding_text() == expected

        # Without metadata
        assert paper.to_embedding_text(include_metadata=False) == "Test Title. Test abstract here."

        # Single author
        paper.authors = ["John Smith"]
        expected = "Test Title. Smith. 2023. Nature. Test abstract here."
        assert paper.to_embedding_text() == expected

        # Three+ authors (et al.)
        paper.authors = ["John Smith", "Jane Doe", "Bob Wilson"]
        expected = "Test Title. Smith et al.. 2023. Nature. Test abstract here."
        assert paper.to_embedding_text() == expected

    def test_to_embedding_text_with_llm_description(self):
        paper = Paper(
            id="1",
            title="Test Title",
            abstract="Test abstract.",
            llm_description="This paper examines X.",
        )
        text = paper.to_embedding_text()
        assert "This paper examines X." in text
        assert "Test Title" in text
        assert "Test abstract." in text

    def test_has_full_text(self):
        paper = Paper(id="1", title="Test", abstract="")
        assert not paper.has_full_text

        paper.full_text = "Full text content"
        assert paper.has_full_text

        # Empty string counts as no full text
        paper.full_text = ""
        assert not paper.has_full_text


class TestCitationContext:
    def test_create_context(self):
        context = CitationContext(
            id="ctx1",
            local_context="This method has been widely used [CITATION].",
            section_context="Methods section",
            global_context="Paper about machine learning",
        )
        assert context.id == "ctx1"
        assert "widely used" in context.local_context

    def test_context_requires_id(self):
        with pytest.raises(ValueError, match="must have an id"):
            CitationContext(id="", local_context="text")

    def test_context_requires_local(self):
        with pytest.raises(ValueError, match="must have local_context"):
            CitationContext(id="1", local_context="")

    def test_get_query_scales(self):
        context = CitationContext(
            id="1",
            local_context="Local text.",
            narrow_context="Narrow text.",
            broad_context="Broad text.",
            section_context="Section text.",
            global_context="Global text.",
        )
        assert context.get_query("local") == "Local text."
        assert context.get_query("narrow") == "Narrow text."
        assert context.get_query("broad") == "Broad text."
        assert "Section text." in context.get_query("section")
        assert "Global text." in context.get_query("global")

    def test_get_query_narrow_fallback(self):
        context = CitationContext(id="1", local_context="Local only.")
        assert context.get_query("narrow") == "Local only."
        assert context.get_query("broad") == "Local only."

    def test_get_query_reformulated_scale(self):
        context = CitationContext(
            id="1",
            local_context="Local text.",
            narrow_context="Narrow text.",
            reformulated_query="This hypothetical paper discusses...",
        )
        result = context.get_query("reformulated")
        assert result == "This hypothetical paper discusses..."

    def test_get_query_reformulated_fallback(self):
        context = CitationContext(
            id="1",
            local_context="Local [CITE] text.",
            narrow_context="Narrow context.",
        )
        # No reformulated_query → falls back to narrow (cleaned)
        result = context.get_query("reformulated")
        assert result == "Narrow context."

    def test_get_query_prefix_section(self):
        context = CitationContext(
            id="1",
            local_context="Local text.",
            section_context="Methods",
        )
        result = context.get_query("local", prefix_section=True)
        assert "Methods" in result
        assert "Local text." in result

    def test_get_query_invalid_scale(self):
        context = CitationContext(id="1", local_context="text")
        with pytest.raises(ValueError, match="Unknown scale"):
            context.get_query("invalid")

    def test_get_query_cleaning(self):
        context = CitationContext(
            id="1",
            local_context="This uses RNNs [CITE] and {{cite:abc123}} methods.",
        )
        # Default is clean=True
        cleaned = context.get_query("local")
        assert "[CITE]" not in cleaned
        assert "{{cite:" not in cleaned
        assert "RNNs" in cleaned
        assert "methods" in cleaned

        # With clean=False, markers preserved
        raw = context.get_query("local", clean=False)
        assert "[CITE]" in raw
        assert "{{cite:abc123}}" in raw


class TestCleanCitationMarkers:
    """Tests for all 7 regex patterns in clean_citation_markers."""

    def test_cite_marker(self):
        assert "text methods" in clean_citation_markers("text [CITE] methods")
        assert "[CITE]" not in clean_citation_markers("text [CITE] methods")

    def test_cite_hash(self):
        result = clean_citation_markers("text {{cite:abc123}} more")
        assert "{{cite:" not in result
        assert "text" in result
        assert "more" in result

    def test_cite_hash_truncated(self):
        # Truncated (missing closing braces)
        result = clean_citation_markers("text {{cite:abc123 more")
        assert "{{cite:" not in result

    def test_formula_uuid(self):
        result = clean_citation_markers("equation {{formula:a1b2-c3d4}} holds")
        assert "{{formula:" not in result
        assert "equation" in result
        assert "holds" in result

    def test_empty_parentheses(self):
        result = clean_citation_markers("text ( ) more")
        assert "( )" not in result
        assert "()" not in result

    def test_parentheses_with_abbrev(self):
        result = clean_citation_markers("text (e.g., ) more")
        assert "(e.g., )" not in result

    def test_double_commas(self):
        result = clean_citation_markers("Smith , , Jones found")
        assert ", ," not in result
        assert ",," not in result

    def test_comma_period(self):
        result = clean_citation_markers("text , . more")
        assert ", ." not in result

    def test_whitespace_normalization(self):
        result = clean_citation_markers("text   with   extra   spaces")
        assert "  " not in result
        assert "text with extra spaces" == result

    def test_combined_patterns(self):
        text = "Previous work [CITE] by {{cite:abc}} and ( ) showed results , , that."
        result = clean_citation_markers(text)
        assert "[CITE]" not in result
        assert "{{cite:" not in result
        assert "( )" not in result
        assert ", ," not in result
        assert "Previous" in result
        assert "that." in result


class TestRetrievalResult:
    def test_create_result(self):
        result = RetrievalResult(
            paper_id="paper1",
            score=0.95,
            rank=1,
        )
        assert result.paper_id == "paper1"
        assert result.score == 0.95
        assert result.rank == 1

    def test_result_comparison(self):
        r1 = RetrievalResult(paper_id="a", score=0.9, rank=1)
        r2 = RetrievalResult(paper_id="b", score=0.8, rank=2)
        # Higher score should be "less than" for sorting
        assert r1 < r2

    def test_result_comparison_reverse(self):
        r1 = RetrievalResult(paper_id="a", score=0.5, rank=2)
        r2 = RetrievalResult(paper_id="b", score=0.9, rank=1)
        assert not r1 < r2

    def test_default_display_mode(self):
        result = RetrievalResult(paper_id="p1", score=0.9, rank=1)
        assert result.display_mode == "paper"

    def test_score_breakdown_default(self):
        result = RetrievalResult(paper_id="p1", score=0.9, rank=1)
        assert result.score_breakdown == {}


class TestGetDisplayMode:
    """Tests for all 3 return paths of get_display_mode."""

    def test_paper_when_no_paragraph(self):
        result = RetrievalResult(
            paper_id="p1",
            score=0.9,
            rank=1,
            matched_paragraph=None,
        )
        assert result.get_display_mode() == "paper"

    def test_paragraph_when_high_chunk_score(self):
        result = RetrievalResult(
            paper_id="p1",
            score=0.9,
            rank=1,
            matched_paragraph="Some paragraph text.",
            score_breakdown={"best_chunk_score": 0.8, "num_chunks_matched": 1},
        )
        assert result.get_display_mode(para_threshold=0.65) == "paragraph"

    def test_paper_with_summary_when_many_chunks(self):
        result = RetrievalResult(
            paper_id="p1",
            score=0.9,
            rank=1,
            matched_paragraph="Some text.",
            score_breakdown={"best_chunk_score": 0.3, "num_chunks_matched": 5},
        )
        assert (
            result.get_display_mode(para_threshold=0.65, multi_chunk_threshold=3)
            == "paper_with_summary"
        )

    def test_paper_when_low_score_few_chunks(self):
        result = RetrievalResult(
            paper_id="p1",
            score=0.9,
            rank=1,
            matched_paragraph="Some text.",
            score_breakdown={"best_chunk_score": 0.3, "num_chunks_matched": 1},
        )
        assert result.get_display_mode() == "paper"

    def test_custom_thresholds(self):
        result = RetrievalResult(
            paper_id="p1",
            score=0.9,
            rank=1,
            matched_paragraph="text",
            score_breakdown={"best_chunk_score": 0.5, "num_chunks_matched": 2},
        )
        # With low threshold, should show paragraph
        assert result.get_display_mode(para_threshold=0.4) == "paragraph"
        # With high threshold and low multi_chunk, should show paper
        assert result.get_display_mode(para_threshold=0.9, multi_chunk_threshold=10) == "paper"


class TestEvaluationResult:
    def test_to_dict(self):
        result = EvaluationResult(
            recall_at_1=0.5,
            recall_at_5=0.7,
            recall_at_10=0.8,
            recall_at_20=0.85,
            recall_at_50=0.9,
            mrr=0.6,
            ndcg_at_10=0.75,
            num_queries=100,
        )
        d = result.to_dict()
        assert d["recall@1"] == 0.5
        assert d["recall@20"] == 0.85
        assert d["recall@50"] == 0.9
        assert d["mrr"] == 0.6
        assert d["ndcg@10"] == 0.75
        assert d["num_queries"] == 100

    def test_str_output(self):
        result = EvaluationResult(recall_at_10=0.8, num_queries=50)
        s = str(result)
        assert "n=50" in s
        assert "0.800" in s
        assert "Recall@10" in s
        assert "MRR" in s


class TestFormatAuthorString:
    """Tests for the canonical format_author_string function."""

    def test_no_authors(self):
        assert format_author_string([]) == ""

    def test_single_author(self):
        assert format_author_string(["Smith"]) == "Smith"

    def test_two_authors(self):
        assert format_author_string(["Smith", "Jones"]) == "Smith and Jones"

    def test_three_authors(self):
        assert format_author_string(["Smith", "Jones", "Brown"]) == "Smith et al."

    def test_many_authors(self):
        assert format_author_string(["A", "B", "C", "D", "E"]) == "A et al."


class TestFormatPaperEmbeddingText:
    """Tests for the canonical format_paper_embedding_text function."""

    def test_title_only(self):
        assert format_paper_embedding_text("Test Title") == "Test Title"

    def test_title_and_abstract(self):
        result = format_paper_embedding_text("Title", abstract="Abstract text")
        assert result == "Title. Abstract text"

    def test_full_metadata(self):
        result = format_paper_embedding_text(
            title="Title",
            abstract="Abstract",
            author_lastnames=["Smith", "Jones"],
            year=2023,
            journal="Nature",
        )
        assert result == "Title. Smith and Jones. 2023. Nature. Abstract"

    def test_no_metadata(self):
        result = format_paper_embedding_text(
            title="Title",
            abstract="Abstract",
            author_lastnames=["Smith"],
            year=2023,
            include_metadata=False,
        )
        assert result == "Title. Abstract"

    def test_no_abstract(self):
        result = format_paper_embedding_text(
            title="Title",
            abstract="Abstract",
            author_lastnames=["Smith"],
            year=2023,
            include_abstract=False,
        )
        assert result == "Title. Smith. 2023"

    def test_with_llm_description(self):
        result = format_paper_embedding_text(
            title="Title",
            abstract="Abstract",
            llm_description="This examines X.",
        )
        assert result == "Title. Abstract. This examines X."

    def test_missing_optional_fields(self):
        result = format_paper_embedding_text(
            title="Title",
            abstract="Abstract",
            author_lastnames=None,
            year=None,
            journal=None,
        )
        assert result == "Title. Abstract"

    def test_matches_paper_to_embedding_text(self):
        """Paper.to_embedding_text() must match format_paper_embedding_text()."""
        paper = Paper(
            id="1",
            title="Test Title",
            abstract="Test abstract here.",
            authors=["John Smith", "Jane Doe"],
            year=2023,
            journal="Nature",
            llm_description="Examines X.",
        )
        expected = format_paper_embedding_text(
            title=paper.title,
            abstract=paper.abstract,
            author_lastnames=paper.author_lastnames,
            year=paper.year,
            journal=paper.journal,
            llm_description=paper.llm_description,
        )
        assert paper.to_embedding_text() == expected


class TestFormatPaperMetadataPrefix:
    """Tests for the canonical format_paper_metadata_prefix function."""

    def test_full_metadata(self):
        result = format_paper_metadata_prefix(
            title="Title",
            author_lastnames=["Smith", "Jones"],
            year=2023,
            journal="Nature",
        )
        assert result == "Title. Smith and Jones. 2023. Nature"

    def test_partial_metadata(self):
        result = format_paper_metadata_prefix(
            title="Title",
            author_lastnames=["Smith"],
        )
        assert result == "Title. Smith"

    def test_title_only(self):
        result = format_paper_metadata_prefix(title="Title")
        assert result == "Title"


class TestFormatPassageEmbeddingText:
    """Tests for the canonical format_passage_embedding_text function."""

    def test_with_prefix(self):
        result = format_passage_embedding_text("Chunk text here.", "Title. Smith. 2023")
        assert result == "Title. Smith. 2023\n\nChunk text here."

    def test_without_prefix(self):
        result = format_passage_embedding_text("Chunk text here.", None)
        assert result == "Chunk text here."

    def test_empty_prefix(self):
        result = format_passage_embedding_text("Chunk text here.", "")
        assert result == "Chunk text here."

    def test_matches_chunk_to_embedding_text(self):
        """Chunk.to_embedding_text() must match format_passage_embedding_text()."""
        chunk = Chunk(
            id="p1::chunk_0",
            paper_id="p1",
            text="Some paragraph text.",
            context_text="Title. Smith. 2023",
        )
        expected = format_passage_embedding_text(chunk.text, chunk.context_text)
        assert chunk.to_embedding_text() == expected

    def test_chunk_without_context(self):
        """Chunk without context_text returns raw text."""
        chunk = Chunk(
            id="p1::chunk_0",
            paper_id="p1",
            text="Some paragraph text.",
        )
        assert chunk.to_embedding_text() == "Some paragraph text."
