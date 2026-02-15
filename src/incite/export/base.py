"""Base protocol and registry for citation export formats."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from incite.models import Paper


@runtime_checkable
class ExportFormat(Protocol):
    """Protocol for citation export formats."""

    format_name: str
    file_extension: str
    mime_type: str

    def export_items(self, papers: list[Paper]) -> str:
        """Export multiple papers to a formatted string."""
        ...

    def export_single(self, paper: Paper) -> str:
        """Export a single paper to a formatted string."""
        ...


FORMATS: dict[str, ExportFormat] = {}
