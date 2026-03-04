"""CSL-formatted bibliography export using citeproc-py."""

from __future__ import annotations

import logging

from citeproc import (
    Citation,
    CitationItem,
    CitationStylesBibliography,
    CitationStylesStyle,
    formatter,
)
from citeproc.source.json import CiteProcJSON
from citeproc_styles import get_style_filepath

from incite.export.base import FORMATS
from incite.models import Paper

logger = logging.getLogger(__name__)


def _paper_to_csl_json(paper: Paper) -> dict:
    """Convert a Paper to a CSL-JSON item dict."""
    item: dict = {
        "id": paper.id,
        "type": "article-journal",
        "title": paper.title,
    }

    if paper.authors:
        csl_authors = []
        for author in paper.authors:
            if "," in author:
                parts = author.split(",", 1)
                csl_authors.append({"family": parts[0].strip(), "given": parts[1].strip()})
            else:
                name_parts = author.split()
                if len(name_parts) >= 2:
                    csl_authors.append(
                        {"family": name_parts[-1], "given": " ".join(name_parts[:-1])}
                    )
                else:
                    csl_authors.append({"family": author})
        item["author"] = csl_authors

    if paper.year is not None:
        item["issued"] = {"date-parts": [[paper.year]]}

    if paper.journal:
        item["container-title"] = paper.journal

    if paper.doi:
        item["DOI"] = paper.doi

    return item


def _render_bibliography(papers: list[Paper], style_name: str) -> str:
    """Render papers as a formatted bibliography using a CSL style."""
    if not papers:
        return ""

    try:
        csl_items = [_paper_to_csl_json(p) for p in papers]
        source = CiteProcJSON(csl_items)

        style_path = get_style_filepath(style_name)
        style = CitationStylesStyle(style_path, validate=False)

        bib = CitationStylesBibliography(style, source, formatter.plain)

        for item in csl_items:
            bib.register(Citation([CitationItem(item["id"])]))

        bib_entries = bib.bibliography()
        if not bib_entries:
            return ""

        return "\n".join(str(entry) for entry in bib_entries)
    except Exception:
        logger.exception("citeproc-py failed to render bibliography (style=%s)", style_name)
        raise


# Style name -> CSL style identifier used by citeproc-py-styles
_STYLE_MAP: dict[str, str] = {
    "apa": "apa",
    "mla": "modern-language-association",
    "chicago": "chicago-author-date",
    "harvard": "harvard-cite-them-right",
}


class BibliographyFormat:
    """CSL-formatted bibliography export."""

    format_name: str
    file_extension: str = ".txt"
    mime_type: str = "text/plain"

    _style_key: str  # CSL style identifier

    def __init__(self, format_name: str, style_key: str) -> None:
        self.format_name = format_name
        self._style_key = style_key

    def export_items(self, papers: list[Paper]) -> str:
        """Export multiple papers as a formatted bibliography."""
        return _render_bibliography(papers, self._style_key)

    def export_single(self, paper: Paper) -> str:
        """Export a single paper as a formatted bibliography entry."""
        return _render_bibliography([paper], self._style_key)


# Register all bibliography styles
for _key, _csl_id in _STYLE_MAP.items():
    FORMATS[_key] = BibliographyFormat(
        format_name=_key.upper(),
        style_key=_csl_id,
    )
