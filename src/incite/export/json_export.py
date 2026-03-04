"""JSON export format for recommendation results."""

from __future__ import annotations

import json

from incite.export.base import FORMATS
from incite.models import Paper


def _paper_to_dict(
    paper: Paper,
    confidence: float | None = None,
    matched_paragraph: str | None = None,
) -> dict:
    d: dict = {
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "journal": paper.journal,
        "doi": paper.doi,
    }
    if confidence is not None:
        d["confidence"] = confidence
    if matched_paragraph is not None:
        d["matched_paragraph"] = matched_paragraph
    return d


class JSONFormat:
    """JSON export format."""

    format_name: str = "JSON"
    file_extension: str = ".json"
    mime_type: str = "application/json"

    def export_items(
        self,
        papers: list[Paper],
        *,
        confidences: list[float] | None = None,
        matched_paragraphs: list[str | None] | None = None,
    ) -> str:
        items = []
        for i, paper in enumerate(papers):
            conf = confidences[i] if confidences else None
            para = matched_paragraphs[i] if matched_paragraphs else None
            items.append(_paper_to_dict(paper, conf, para))
        return json.dumps(items, indent=2, ensure_ascii=False)

    def export_single(
        self,
        paper: Paper,
        confidence: float | None = None,
        matched_paragraph: str | None = None,
    ) -> str:
        return json.dumps(
            [_paper_to_dict(paper, confidence, matched_paragraph)],
            indent=2,
            ensure_ascii=False,
        )


FORMATS["json"] = JSONFormat()
