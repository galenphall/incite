"""GROBID client for structured PDF extraction.

GROBID is a machine learning library for extracting, parsing, and
restructuring raw documents into structured TEI-XML. It provides
~90% accuracy on scholarly PDFs and is used in production at
Semantic Scholar, ResearchGate, and other major platforms.

Requires GROBID service running (via Docker):
    docker run --rm -p 8070:8070 grobid/grobid:0.8.0

Reference: https://grobid.readthedocs.io/
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import requests

logger = logging.getLogger(__name__)

# TEI namespace used by GROBID
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


@dataclass
class GROBIDSection:
    """A section extracted from a PDF by GROBID."""

    heading: Optional[str]
    text: str
    section_type: Optional[str] = None  # e.g., "abstract", "introduction", "methods"


@dataclass
class GROBIDReference:
    """A bibliographic reference extracted by GROBID."""

    raw_text: str
    title: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    journal: Optional[str] = None


@dataclass
class GROBIDFigure:
    """A figure or table extracted by GROBID."""

    caption: str
    label: Optional[str] = None
    figure_type: str = "figure"  # "figure" or "table"


@dataclass
class GROBIDResult:
    """Complete extraction result from GROBID."""

    title: Optional[str] = None
    abstract: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    sections: list[GROBIDSection] = field(default_factory=list)
    references: list[GROBIDReference] = field(default_factory=list)
    figures: list[GROBIDFigure] = field(default_factory=list)
    raw_xml: Optional[str] = None

    @property
    def full_text(self) -> str:
        """Concatenate all sections into full text."""
        parts = []
        if self.abstract:
            parts.append(self.abstract)
        for section in self.sections:
            if section.heading:
                parts.append(section.heading)
            parts.append(section.text)
        return "\n\n".join(parts)

    @property
    def paragraphs(self) -> list[str]:
        """Extract paragraphs from all sections."""
        paragraphs = []
        if self.abstract:
            paragraphs.append(self.abstract)
        for section in self.sections:
            # Split section text into paragraphs
            for para in section.text.split("\n\n"):
                para = para.strip()
                if para:
                    paragraphs.append(para)
        return paragraphs


class GROBIDClient:
    """Client for GROBID PDF extraction service.

    Usage:
        client = GROBIDClient()
        if client.is_available():
            result = client.extract_pdf("paper.pdf")
            print(result.title)
            for section in result.sections:
                print(f"{section.heading}: {len(section.text)} chars")
    """

    DEFAULT_URL = "http://localhost:8070"
    TIMEOUT = 180  # seconds per request (increased for cloud cold starts)

    def __init__(self, url: str = DEFAULT_URL):
        """Initialize GROBID client.

        Args:
            url: Base URL of GROBID service (default: http://localhost:8070)
        """
        self.url = url.rstrip("/")

    def is_available(self) -> bool:
        """Check if GROBID service is running and responding."""
        try:
            response = requests.get(
                f"{self.url}/api/isalive",
                timeout=5,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_version(self) -> Optional[str]:
        """Get GROBID version string."""
        try:
            response = requests.get(
                f"{self.url}/api/version",
                timeout=5,
            )
            if response.status_code == 200:
                return response.text.strip()
        except requests.RequestException:
            pass
        return None

    def extract_pdf(
        self,
        pdf_path: str | Path,
        include_raw_xml: bool = False,
    ) -> GROBIDResult:
        """Extract structured content from a PDF file.

        Args:
            pdf_path: Path to PDF file
            include_raw_xml: If True, include raw TEI-XML in result

        Returns:
            GROBIDResult with extracted structure

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            requests.RequestException: If GROBID request fails
            ValueError: If GROBID returns invalid response
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Call GROBID's fulltext endpoint
        with open(pdf_path, "rb") as f:
            response = requests.post(
                f"{self.url}/api/processFulltextDocument",
                files={"input": (pdf_path.name, f, "application/pdf")},
                data={
                    "consolidateCitations": "1",  # Improve reference extraction
                    "includeRawCitations": "1",
                },
                timeout=self.TIMEOUT,
            )

        if response.status_code != 200:
            raise ValueError(
                f"GROBID extraction failed: {response.status_code} - {response.text[:200]}"
            )

        xml_content = response.text
        return self._parse_tei_xml(xml_content, include_raw_xml=include_raw_xml)

    def _parse_tei_xml(
        self,
        xml_content: str,
        include_raw_xml: bool = False,
    ) -> GROBIDResult:
        """Parse GROBID's TEI-XML output into structured result.

        Args:
            xml_content: TEI-XML string from GROBID
            include_raw_xml: If True, include raw XML in result

        Returns:
            Parsed GROBIDResult
        """
        result = GROBIDResult()
        if include_raw_xml:
            result.raw_xml = xml_content

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.warning(f"Failed to parse GROBID XML: {e}")
            return result

        # Extract title
        title_elem = root.find(".//tei:titleStmt/tei:title", TEI_NS)
        if title_elem is not None and title_elem.text:
            result.title = self._clean_text(title_elem.text)

        # Extract authors
        for author in root.findall(".//tei:sourceDesc//tei:author", TEI_NS):
            name_parts = []
            forename = author.find(".//tei:forename", TEI_NS)
            surname = author.find(".//tei:surname", TEI_NS)
            if forename is not None and forename.text:
                name_parts.append(forename.text)
            if surname is not None and surname.text:
                name_parts.append(surname.text)
            if name_parts:
                result.authors.append(" ".join(name_parts))

        # Extract abstract
        abstract_elem = root.find(".//tei:profileDesc/tei:abstract", TEI_NS)
        if abstract_elem is not None:
            abstract_text = self._get_all_text(abstract_elem)
            if abstract_text:
                result.abstract = self._clean_text(abstract_text)

        # Extract body sections
        body = root.find(".//tei:body", TEI_NS)
        if body is not None:
            result.sections = self._extract_sections(body)

        # Extract references (from back matter - completely separate from body)
        back = root.find(".//tei:back", TEI_NS)
        if back is not None:
            result.references = self._extract_references(back)

        # Extract figures and tables
        for figure in root.findall(".//tei:figure", TEI_NS):
            fig_result = self._extract_figure(figure)
            if fig_result:
                result.figures.append(fig_result)

        return result

    def _extract_sections(self, body: ET.Element) -> list[GROBIDSection]:
        """Extract sections from TEI body element."""
        sections = []

        for div in body.findall(".//tei:div", TEI_NS):
            # Get section heading
            heading = None
            head_elem = div.find("tei:head", TEI_NS)
            if head_elem is not None:
                heading = self._clean_text(self._get_all_text(head_elem))

            # Get section type from @type attribute if present
            section_type = div.get("type")

            # Get section text (all paragraphs)
            paragraphs = []
            for p in div.findall("tei:p", TEI_NS):
                p_text = self._get_all_text(p)
                if p_text:
                    paragraphs.append(self._clean_text(p_text))

            if paragraphs:
                sections.append(
                    GROBIDSection(
                        heading=heading,
                        text="\n\n".join(paragraphs),
                        section_type=section_type,
                    )
                )

        return sections

    def _extract_references(self, back: ET.Element) -> list[GROBIDReference]:
        """Extract bibliographic references from TEI back element."""
        references = []

        for bibl in back.findall(".//tei:listBibl/tei:biblStruct", TEI_NS):
            ref = GROBIDReference(raw_text="")

            # Get raw text
            raw_parts = []
            for text in bibl.itertext():
                if text.strip():
                    raw_parts.append(text.strip())
            ref.raw_text = " ".join(raw_parts)

            # Get title
            title_elem = bibl.find(".//tei:title[@level='a']", TEI_NS)
            if title_elem is None:
                title_elem = bibl.find(".//tei:title", TEI_NS)
            if title_elem is not None and title_elem.text:
                ref.title = self._clean_text(title_elem.text)

            # Get authors
            for author in bibl.findall(".//tei:author", TEI_NS):
                name_parts = []
                forename = author.find(".//tei:forename", TEI_NS)
                surname = author.find(".//tei:surname", TEI_NS)
                if forename is not None and forename.text:
                    name_parts.append(forename.text)
                if surname is not None and surname.text:
                    name_parts.append(surname.text)
                if name_parts:
                    ref.authors.append(" ".join(name_parts))

            # Get year
            date_elem = bibl.find(".//tei:date[@when]", TEI_NS)
            if date_elem is not None:
                when = date_elem.get("when", "")
                if when and when[:4].isdigit():
                    ref.year = int(when[:4])

            # Get DOI
            doi_elem = bibl.find(".//tei:idno[@type='DOI']", TEI_NS)
            if doi_elem is not None and doi_elem.text:
                ref.doi = doi_elem.text.strip()

            # Get journal
            journal_elem = bibl.find(".//tei:title[@level='j']", TEI_NS)
            if journal_elem is not None and journal_elem.text:
                ref.journal = self._clean_text(journal_elem.text)

            references.append(ref)

        return references

    def _extract_figure(self, figure: ET.Element) -> Optional[GROBIDFigure]:
        """Extract figure or table from TEI figure element."""
        # Get caption (figDesc element)
        desc_elem = figure.find("tei:figDesc", TEI_NS)
        if desc_elem is None:
            return None

        caption = self._get_all_text(desc_elem)
        if not caption:
            return None

        # Get label (Figure 1, Table 2, etc.)
        label = None
        label_elem = figure.find("tei:label", TEI_NS)
        if label_elem is not None and label_elem.text:
            label = label_elem.text.strip()

        # Determine type
        fig_type = figure.get("type", "figure")
        if fig_type not in ("figure", "table"):
            fig_type = "figure"

        return GROBIDFigure(
            caption=self._clean_text(caption),
            label=label,
            figure_type=fig_type,
        )

    def _get_all_text(self, elem: ET.Element) -> str:
        """Get all text content from element and its children."""
        parts = []
        for text in elem.itertext():
            if text.strip():
                parts.append(text.strip())
        return " ".join(parts)

    def _clean_text(self, text: str) -> str:
        """Clean extracted text (normalize whitespace, etc.)."""
        import re

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()
