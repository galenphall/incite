"""Tests for GROBID client and chunking modules."""

from incite.corpus.grobid import (
    GROBIDClient,
    GROBIDResult,
    GROBIDSection,
)


class TestGROBIDResult:
    """Tests for GROBIDResult dataclass."""

    def test_full_text_property(self):
        """Test concatenating sections into full text."""
        result = GROBIDResult(
            abstract="This is the abstract.",
            sections=[
                GROBIDSection(heading="Introduction", text="Intro paragraph."),
                GROBIDSection(heading="Methods", text="Methods paragraph."),
            ],
        )
        full_text = result.full_text
        assert "This is the abstract." in full_text
        assert "Intro paragraph." in full_text
        assert "Methods paragraph." in full_text

    def test_paragraphs_property(self):
        """Test extracting paragraphs from sections."""
        result = GROBIDResult(
            abstract="Abstract text.",
            sections=[
                GROBIDSection(
                    heading="Introduction",
                    text="First para.\n\nSecond para.",
                ),
            ],
        )
        paragraphs = result.paragraphs
        assert len(paragraphs) == 3  # abstract + 2 paragraphs
        assert paragraphs[0] == "Abstract text."
        assert paragraphs[1] == "First para."
        assert paragraphs[2] == "Second para."

    def test_empty_result(self):
        """Test empty result has empty full_text and paragraphs."""
        result = GROBIDResult()
        assert result.full_text == ""
        assert result.paragraphs == []


class TestGROBIDClient:
    """Tests for GROBIDClient."""

    def test_client_url_trailing_slash(self):
        """Test trailing slash is stripped from URL."""
        client = GROBIDClient(url="http://localhost:8070/")
        assert client.url == "http://localhost:8070"

    def test_is_available_when_offline(self):
        """Test is_available returns False when service is not running."""
        client = GROBIDClient(url="http://localhost:9999")  # Non-existent port
        assert client.is_available() is False


class TestGROBIDXMLParsing:
    """Tests for GROBID XML parsing."""

    def test_parse_simple_tei_xml(self):
        """Test parsing a simple TEI-XML document."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
            <teiHeader>
                <fileDesc>
                    <titleStmt>
                        <title>Test Paper Title</title>
                    </titleStmt>
                    <sourceDesc>
                        <biblStruct>
                            <analytic>
                                <author>
                                    <persName>
                                        <forename>John</forename>
                                        <surname>Smith</surname>
                                    </persName>
                                </author>
                            </analytic>
                        </biblStruct>
                    </sourceDesc>
                </fileDesc>
                <profileDesc>
                    <abstract>
                        <p>This is the abstract of the paper.</p>
                    </abstract>
                </profileDesc>
            </teiHeader>
            <text>
                <body>
                    <div type="section">
                        <head>1. Introduction</head>
                        <p>This is the introduction paragraph.</p>
                    </div>
                </body>
                <back>
                    <listBibl>
                        <biblStruct>
                            <analytic>
                                <title level="a">Referenced Paper</title>
                                <author>
                                    <persName>
                                        <forename>Jane</forename>
                                        <surname>Doe</surname>
                                    </persName>
                                </author>
                            </analytic>
                            <monogr>
                                <title level="j">Journal of Testing</title>
                                <imprint>
                                    <date when="2020"/>
                                </imprint>
                            </monogr>
                            <idno type="DOI">10.1234/test</idno>
                        </biblStruct>
                    </listBibl>
                </back>
            </text>
        </TEI>
        """

        client = GROBIDClient()
        result = client._parse_tei_xml(xml_content)

        assert result.title == "Test Paper Title"
        assert "John Smith" in result.authors
        assert result.abstract == "This is the abstract of the paper."
        assert len(result.sections) == 1
        assert result.sections[0].heading == "1. Introduction"
        assert "introduction paragraph" in result.sections[0].text
        assert len(result.references) == 1
        assert result.references[0].title == "Referenced Paper"
        assert result.references[0].doi == "10.1234/test"
        assert result.references[0].year == 2020

    def test_parse_malformed_xml(self):
        """Test parsing malformed XML returns empty result."""
        client = GROBIDClient()
        result = client._parse_tei_xml("<not valid xml")
        assert result.title is None
        assert result.sections == []

