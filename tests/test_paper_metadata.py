"""Tests for cloud.models DocumentMetadata and PaperMetadata persistence models."""

import pytest

pydantic = pytest.importorskip("pydantic", reason="pydantic required for cloud model tests")
ValidationError = pydantic.ValidationError

from cloud.models import DocumentMetadata, PaperMetadata


class TestDocumentMetadata:
    """Tests for the base DocumentMetadata model."""

    def test_requires_title(self):
        with pytest.raises(ValidationError):
            DocumentMetadata()

    def test_minimal_creation(self):
        doc = DocumentMetadata(title="A Title")
        assert doc.title == "A Title"
        assert doc.id == ""
        assert doc.year is None
        assert doc.document_type == "paper"

    def test_extra_fields_preserved(self):
        doc = DocumentMetadata(title="A Title", custom_field="hello", number=42)
        assert doc.custom_field == "hello"
        assert doc.number == 42
        dumped = doc.model_dump()
        assert dumped["custom_field"] == "hello"
        assert dumped["number"] == 42


class TestPaperMetadata:
    """Tests for the PaperMetadata persistence model."""

    def test_requires_title(self):
        with pytest.raises(ValidationError):
            PaperMetadata()

    def test_minimal_paper(self):
        paper = PaperMetadata(title="My Paper")
        assert paper.title == "My Paper"
        assert paper.authors == []
        assert paper.abstract == ""
        assert paper.document_type == "paper"

    def test_full_paper_with_all_fields(self):
        paper = PaperMetadata(
            id="abc123",
            title="Deep Learning for Citations",
            year=2024,
            authors=["Smith", "Jones"],
            abstract="We propose a new method.",
            doi="10.1234/test",
            journal="Nature",
            bibtex_key="smith2024deep",
            zotero_uri="zotero://select/items/ABC",
            url="https://example.com/paper",
            pdf_url="https://example.com/paper.pdf",
        )
        assert paper.id == "abc123"
        assert paper.title == "Deep Learning for Citations"
        assert paper.year == 2024
        assert paper.authors == ["Smith", "Jones"]
        assert paper.abstract == "We propose a new method."
        assert paper.doi == "10.1234/test"
        assert paper.journal == "Nature"
        assert paper.bibtex_key == "smith2024deep"
        assert paper.zotero_uri == "zotero://select/items/ABC"
        assert paper.url == "https://example.com/paper"
        assert paper.pdf_url == "https://example.com/paper.pdf"
        assert paper.document_type == "paper"

    def test_extra_fields_preserved(self):
        paper = PaperMetadata(
            title="Test Paper",
            authors=["Author"],
            structured_authors=[{"given": "John", "family": "Author"}],
            volume="12",
            issue="3",
            pages="100-110",
            item_type="journalArticle",
        )
        assert paper.structured_authors == [{"given": "John", "family": "Author"}]
        assert paper.volume == "12"
        assert paper.issue == "3"
        assert paper.pages == "100-110"
        assert paper.item_type == "journalArticle"

    def test_to_dict_includes_extras(self):
        paper = PaperMetadata(
            title="Test",
            authors=["A"],
            volume="5",
            custom="value",
        )
        d = paper.model_dump()
        assert d["title"] == "Test"
        assert d["authors"] == ["A"]
        assert d["volume"] == "5"
        assert d["custom"] == "value"
        assert d["document_type"] == "paper"

    def test_from_dict_round_trip(self):
        original = PaperMetadata(
            id="p1",
            title="Round Trip",
            year=2023,
            authors=["Smith"],
            abstract="Abstract text",
            doi="10.1/test",
            journal="Science",
            bibtex_key="smith2023",
            zotero_uri="zotero://abc",
            url="https://example.com",
            pdf_url="https://example.com/pdf",
            structured_authors=[{"given": "A", "family": "Smith"}],
            volume="1",
        )
        d = original.model_dump()
        restored = PaperMetadata(**d)
        assert restored.model_dump() == d

    def test_attribute_access(self):
        paper = PaperMetadata(
            id="x",
            title="Access Test",
            year=2025,
            authors=["Doe"],
        )
        assert paper.title == "Access Test"
        assert paper.id == "x"
        assert paper.year == 2025
        assert paper.authors == ["Doe"]
        assert paper.doi is None

    def test_paper_metadata_round_trip_no_field_loss(self):
        """Verify that all fields survive: dict -> PaperMetadata -> model_dump -> PaperMetadata."""
        # Simulate a rich Zotero paper with all possible fields
        original = {
            "id": "zotero_ABC123",
            "title": "Test Paper",
            "authors": ["Smith, John", "Jones, Jane"],
            "year": 2024,
            "abstract": "This is a test abstract.",
            "doi": "10.1234/test.2024",
            "journal": "Nature",
            "bibtex_key": "smith2024test",
            "zotero_uri": "zotero://select/items/ABC123",
            "url": "https://example.com/paper",
            "pdf_url": "https://example.com/paper.pdf",
            # Zotero-specific extras
            "structured_authors": [
                {"firstName": "John", "lastName": "Smith"},
                {"firstName": "Jane", "lastName": "Jones"},
            ],
            "volume": "42",
            "issue": "3",
            "pages": "100-110",
            "item_type": "journalArticle",
            "publisher": "Nature Publishing Group",
            "language": "en",
        }

        # Simulate: ingest -> validate -> serialize to DB
        validated = PaperMetadata.model_validate(original)
        stored = validated.model_dump(exclude_none=True)

        # Simulate: load from DB -> validate -> use
        loaded = PaperMetadata.model_validate(stored)
        final = loaded.model_dump(exclude_none=True)

        # Every original field must survive
        for key, value in original.items():
            assert key in final, f"Field '{key}' lost in round-trip"
            assert final[key] == value, f"Field '{key}' changed: {value!r} -> {final[key]!r}"

    def test_paper_metadata_attribute_access_after_round_trip(self):
        """ML pipeline uses attribute access -- verify it works after round-trip."""
        original = {
            "id": "test_123",
            "title": "Test Paper",
            "authors": ["Smith"],
            "year": 2024,
            "journal": "Nature",
            "abstract": "Abstract text",
            "volume": "42",
        }

        validated = PaperMetadata.model_validate(original)
        stored = validated.model_dump(exclude_none=True)
        loaded = PaperMetadata.model_validate(stored)

        # Standard fields
        assert loaded.title == "Test Paper"
        assert loaded.id == "test_123"
        assert loaded.year == 2024
        assert loaded.journal == "Nature"
        assert loaded.authors == ["Smith"]
        assert loaded.abstract == "Abstract text"

        # Extra field via attribute access
        assert loaded.volume == "42"
