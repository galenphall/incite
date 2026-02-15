"""BibTeX export format."""

from __future__ import annotations

import re

from incite.export.base import FORMATS
from incite.models import Paper

# Characters that need escaping in BibTeX field values
_LATEX_SPECIAL = str.maketrans(
    {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
)


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters in a string."""
    # Handle braces first (before translation table, since we add braces)
    text = text.replace("\\", r"\textbackslash{}")
    text = text.replace("{", r"\{")
    text = text.replace("}", r"\}")
    return text.translate(_LATEX_SPECIAL)


def _generate_key(paper: Paper) -> str:
    """Generate a BibTeX citation key from paper metadata."""
    # First author last name
    author = ""
    if paper.authors:
        first = paper.authors[0]
        if "," in first:
            author = first.split(",")[0].strip()
        else:
            parts = first.split()
            author = parts[-1] if parts else ""

    year = str(paper.year) if paper.year else ""

    # First substantial word of title (skip short words)
    title_word = ""
    if paper.title:
        for word in re.split(r"\s+", paper.title):
            cleaned = re.sub(r"[^a-zA-Z]", "", word)
            if len(cleaned) >= 3:
                title_word = cleaned
                break

    key = f"{author}{year}{title_word}".lower()
    # Remove any remaining non-alphanumeric chars
    return re.sub(r"[^a-z0-9]", "", key) or paper.id


def _format_entry(paper: Paper) -> str:
    """Format a single paper as a BibTeX entry."""
    key = paper.bibtex_key if paper.bibtex_key else _generate_key(paper)

    fields: list[str] = []
    fields.append(f"  title = {{{_escape_latex(paper.title)}}}")

    if paper.authors:
        authors_str = " and ".join(paper.authors)
        fields.append(f"  author = {{{_escape_latex(authors_str)}}}")

    if paper.year is not None:
        fields.append(f"  year = {{{paper.year}}}")

    if paper.journal:
        fields.append(f"  journal = {{{_escape_latex(paper.journal)}}}")

    if paper.doi:
        fields.append(f"  doi = {{{paper.doi}}}")

    if paper.abstract:
        fields.append(f"  abstract = {{{_escape_latex(paper.abstract)}}}")

    body = ",\n".join(fields)
    return f"@article{{{key},\n{body}\n}}"


class BibTeXFormat:
    """BibTeX citation export format."""

    format_name: str = "BibTeX"
    file_extension: str = ".bib"
    mime_type: str = "application/x-bibtex"

    def export_items(self, papers: list[Paper]) -> str:
        """Export multiple papers to BibTeX format."""
        return "\n\n".join(_format_entry(p) for p in papers)

    def export_single(self, paper: Paper) -> str:
        """Export a single paper to BibTeX format."""
        return _format_entry(paper)


FORMATS["bibtex"] = BibTeXFormat()
