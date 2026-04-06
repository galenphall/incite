"""Base protocol and registry for citation export formats.

Defines the ``ExportFormat`` Protocol that all export format classes must
satisfy, and the ``FORMATS`` dict that acts as the global format registry.
Format modules (bibtex.py, ris.py, etc.) register themselves by inserting
into ``FORMATS`` at import time.

Available formats (populated by submodules):
- ``bibtex`` — BibTeX (.bib)
- ``ris`` — RIS (.ris)
- ``csv`` — Spreadsheet (.csv)
- ``json`` — JSON array (.json)
- ``apa``, ``mla``, ``chicago``, ``harvard`` — CSL-formatted plain text
"""

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
