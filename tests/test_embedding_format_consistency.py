"""Regression tests for embedding text format consistency.

Ensures all paths that produce paper/passage embedding text use the same
canonical format from models.py. Guards against the train/eval distribution
shift that caused the v5 regression.
"""

from incite.corpus.chunking import _build_paper_metadata_prefix
from incite.finetuning.data_preparation import _paper_to_embedding_text
from incite.finetuning.data_sources import _format_positive
from incite.models import (
    Chunk,
    Paper,
    format_paper_embedding_text,
    format_paper_metadata_prefix,
    format_passage_embedding_text,
)


def _make_paper(**kwargs) -> Paper:
    """Create a Paper with sensible defaults for testing."""
    defaults = {
        "id": "test_paper_1",
        "title": "Test Paper Title",
        "abstract": "This is a test abstract about machine learning.",
        "authors": ["John Smith", "Jane Doe"],
        "year": 2023,
        "journal": "Nature",
    }
    defaults.update(kwargs)
    return Paper(**defaults)


class TestPaperEmbeddingConsistency:
    """Verify all paper embedding paths produce identical output."""

    def test_paper_method_matches_canonical(self):
        """Paper.to_embedding_text() must match format_paper_embedding_text()."""
        paper = _make_paper()
        canonical = format_paper_embedding_text(
            title=paper.title,
            abstract=paper.abstract,
            author_lastnames=paper.author_lastnames,
            year=paper.year,
            journal=paper.journal,
            llm_description=paper.llm_description,
        )
        assert paper.to_embedding_text() == canonical

    def test_data_preparation_matches_paper(self):
        """_paper_to_embedding_text() delegates to Paper.to_embedding_text()."""
        paper = _make_paper()
        assert _paper_to_embedding_text(paper) == paper.to_embedding_text()

    def test_format_positive_uses_dot_separator(self):
        """_format_positive() must use '. ' (not [SEP])."""
        result = _format_positive("Test Title", "Test abstract")
        assert "[SEP]" not in result
        assert ". " in result
        assert result == "Test Title. Test abstract"

    def test_format_positive_matches_canonical_no_metadata(self):
        """_format_positive(title, abstract) produces same as canonical with no metadata."""
        result = _format_positive("Test Title", "Test abstract")
        canonical = format_paper_embedding_text(
            title="Test Title",
            abstract="Test abstract",
            include_abstract=True,
            include_metadata=True,
        )
        assert result == canonical

    def test_format_positive_title_only(self):
        """_format_positive with empty abstract returns just title."""
        result = _format_positive("Test Title", "")
        assert result == "Test Title"


class TestAuthorFormatConsistency:
    """Verify 2-author format is consistent across all paths."""

    def test_two_authors_paper(self):
        paper = _make_paper(authors=["John Smith", "Jane Doe"])
        text = paper.to_embedding_text()
        assert "Smith and Doe" in text
        assert "et al." not in text

    def test_two_authors_chunking_prefix(self):
        paper = _make_paper(authors=["John Smith", "Jane Doe"])
        prefix = _build_paper_metadata_prefix(paper)
        assert "Smith and Doe" in prefix
        assert "et al." not in prefix

    def test_two_authors_canonical(self):
        prefix = format_paper_metadata_prefix(
            title="Title",
            author_lastnames=["Smith", "Doe"],
        )
        assert "Smith and Doe" in prefix
        assert "et al." not in prefix

    def test_three_authors_uses_et_al_everywhere(self):
        paper = _make_paper(authors=["John Smith", "Jane Doe", "Bob Wilson"])
        text = paper.to_embedding_text()
        prefix = _build_paper_metadata_prefix(paper)
        assert "Smith et al." in text
        assert "Smith et al." in prefix


class TestChunkMetadataPrefixConsistency:
    """Verify chunking metadata prefix matches canonical format."""

    def test_chunking_prefix_matches_canonical(self):
        paper = _make_paper()
        prefix = _build_paper_metadata_prefix(paper)
        canonical = format_paper_metadata_prefix(
            title=paper.title,
            author_lastnames=paper.author_lastnames,
            year=paper.year,
            journal=paper.journal,
        )
        assert prefix == canonical

    def test_chunking_prefix_used_by_chunk(self):
        """Chunk.to_embedding_text() uses the prefix via format_passage_embedding_text."""
        paper = _make_paper()
        prefix = _build_paper_metadata_prefix(paper)
        chunk = Chunk(
            id=f"{paper.id}::chunk_0",
            paper_id=paper.id,
            text="Some paragraph text.",
            context_text=prefix,
        )
        expected = format_passage_embedding_text("Some paragraph text.", prefix)
        assert chunk.to_embedding_text() == expected
        assert chunk.to_embedding_text().startswith(prefix)


class TestPassageFormatConsistency:
    """Verify passage training data includes metadata prefix."""

    def test_passage_embedding_with_prefix(self):
        prefix = format_paper_metadata_prefix(title="Title", author_lastnames=["Smith"], year=2023)
        result = format_passage_embedding_text("Paragraph text.", prefix)
        assert result.startswith("Title. Smith. 2023")
        assert "\n\n" in result
        assert result.endswith("Paragraph text.")

    def test_passage_embedding_without_prefix(self):
        result = format_passage_embedding_text("Paragraph text.", None)
        assert result == "Paragraph text."


class TestStripMetadataPrefix:
    """Verify strip_metadata_prefix round-trip behavior."""

    def test_strips_full_paper_embedding_text(self):
        """strip_metadata_prefix(format_paper_embedding_text(...)) returns the abstract."""
        from incite.models import strip_metadata_prefix

        abstract = (
            "We propose a novel method for ground-state cooling of mechanical "
            "resonators using quantum feedback control techniques."
        )
        full_text = format_paper_embedding_text(
            title="Ground-state cooling of mechanical resonators",
            abstract=abstract,
            author_lastnames=["Martin", "Zurek"],
            year=2004,
            journal="Physical Review Letters",
        )
        assert strip_metadata_prefix(full_text) == abstract

    def test_raw_abstract_unchanged(self):
        """strip_metadata_prefix(raw_abstract) returns the abstract unchanged."""
        from incite.models import strip_metadata_prefix

        abstract = (
            "We propose a novel method for ground-state cooling of mechanical "
            "resonators using quantum feedback control techniques."
        )
        assert strip_metadata_prefix(abstract) == abstract

    def test_single_author(self):
        """Works with single-author format."""
        from incite.models import strip_metadata_prefix

        abstract = (
            "This paper presents a comprehensive review of deep learning "
            "approaches for natural language processing tasks."
        )
        full_text = format_paper_embedding_text(
            title="Deep Learning for NLP",
            abstract=abstract,
            author_lastnames=["Smith"],
            year=2023,
        )
        assert strip_metadata_prefix(full_text) == abstract

    def test_et_al_format(self):
        """Works with 'et al.' author format."""
        from incite.models import strip_metadata_prefix

        abstract = (
            "We introduce a transformer-based architecture for scientific "
            "document understanding and citation recommendation."
        )
        full_text = format_paper_embedding_text(
            title="Transformers for Science",
            abstract=abstract,
            author_lastnames=["Smith", "Jones", "Wilson"],
            year=2023,
        )
        assert strip_metadata_prefix(full_text) == abstract

    def test_short_text_not_stripped(self):
        """Short text after prefix is not stripped (safety check)."""
        from incite.models import strip_metadata_prefix

        # With only a very short abstract, the function should not strip
        short = format_paper_embedding_text(
            title="Title",
            abstract="Short.",
            author_lastnames=["Smith"],
            year=2023,
        )
        assert strip_metadata_prefix(short) == short

    def test_format_variant_dedup(self):
        """Same paper in different formats: stripped versions match."""
        from incite.models import strip_metadata_prefix

        abstract = (
            "We propose a novel method for ground-state cooling of mechanical "
            "resonators using quantum feedback control techniques."
        )
        with_metadata = format_paper_embedding_text(
            title="Ground-state cooling",
            abstract=abstract,
            author_lastnames=["Martin"],
            year=2004,
        )
        without_metadata = abstract

        assert strip_metadata_prefix(with_metadata) == strip_metadata_prefix(without_metadata)
