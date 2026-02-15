"""RIS export format."""

from __future__ import annotations

from incite.export.base import FORMATS
from incite.models import Paper


def _format_entry(paper: Paper) -> str:
    """Format a single paper as an RIS record."""
    lines: list[str] = []
    lines.append("TY  - JOUR")
    lines.append(f"TI  - {paper.title}")

    for author in paper.authors:
        lines.append(f"AU  - {author}")

    if paper.year is not None:
        lines.append(f"PY  - {paper.year}")

    if paper.doi:
        lines.append(f"DO  - {paper.doi}")

    if paper.journal:
        lines.append(f"JO  - {paper.journal}")

    if paper.abstract:
        lines.append(f"AB  - {paper.abstract}")

    lines.append("ER  - ")
    return "\n".join(lines)


class RISFormat:
    """RIS citation export format."""

    format_name: str = "RIS"
    file_extension: str = ".ris"
    mime_type: str = "application/x-research-info-systems"

    def export_items(self, papers: list[Paper]) -> str:
        """Export multiple papers to RIS format."""
        return "\n\n".join(_format_entry(p) for p in papers)

    def export_single(self, paper: Paper) -> str:
        """Export a single paper to RIS format."""
        return _format_entry(paper)


FORMATS["ris"] = RISFormat()
