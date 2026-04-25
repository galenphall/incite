"""CSV export format for recommendation results."""

from __future__ import annotations

import csv
import io

from incite.export.base import FORMATS
from incite.models import Paper

_COLUMNS = [
    "title",
    "authors",
    "year",
    "journal",
    "doi",
    "confidence",
    "matched_paragraph",
]


def _paper_to_row(
    paper: Paper,
    confidence: float | None = None,
    matched_paragraph: str | None = None,
) -> dict[str, str]:
    """Convert a Paper to a CSV row dict."""
    return {
        "title": paper.title,
        "authors": "; ".join(paper.authors),
        "year": str(paper.year) if paper.year is not None else "",
        "journal": paper.journal or "",
        "doi": paper.doi or "",
        "confidence": str(confidence) if confidence is not None else "",
        "matched_paragraph": matched_paragraph or "",
    }


class CSVFormat:
    """CSV export format."""

    format_name: str = "CSV"
    file_extension: str = ".csv"
    mime_type: str = "text/csv"

    def export_items(
        self,
        papers: list[Paper],
        *,
        confidences: list[float] | None = None,
        matched_paragraphs: list[str | None] | None = None,
    ) -> str:
        """Export multiple papers to CSV format with optional confidence and evidence columns."""
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=_COLUMNS)
        writer.writeheader()
        for i, paper in enumerate(papers):
            conf = confidences[i] if confidences else None
            para = matched_paragraphs[i] if matched_paragraphs else None
            writer.writerow(_paper_to_row(paper, conf, para))
        return buf.getvalue()

    def export_single(
        self,
        paper: Paper,
        confidence: float | None = None,
        matched_paragraph: str | None = None,
    ) -> str:
        """Export a single paper to CSV format with optional confidence and evidence columns."""
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerow(_paper_to_row(paper, confidence, matched_paragraph))
        return buf.getvalue()


FORMATS["csv"] = CSVFormat()
