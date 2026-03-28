"""Optional dependency handling with clear error messages.

This module provides centralized import helpers for optional dependencies,
with detailed error messages that guide users on how to install them.
"""

from typing import Any


class MissingDependencyError(ImportError):
    """Raised when an optional dependency is not installed."""

    pass


def require_pymupdf(operation: str = "PDF extraction") -> Any:
    """Import PyMuPDF (fitz) or raise a helpful error.

    PyMuPDF is required for extracting text from PDF files.

    Args:
        operation: Description of the operation requiring PyMuPDF (for error message)

    Returns:
        The fitz module

    Raises:
        MissingDependencyError: If PyMuPDF is not installed

    Example:
        >>> fitz = require_pymupdf()
        >>> doc = fitz.open("paper.pdf")
    """
    try:
        import fitz

        return fitz
    except ImportError as e:
        raise MissingDependencyError(
            f"PyMuPDF is required for {operation}.\n\n"
            "Install it with:\n"
            "  pip install pymupdf\n\n"
            "Or install incite with PDF support:\n"
            "  pip install incite[pdf]  # (if PDF extras are defined)\n\n"
            "PyMuPDF documentation: https://pymupdf.readthedocs.io/"
        ) from e


def require_pyzotero(operation: str = "Zotero API access") -> Any:
    """Import pyzotero or raise a helpful error.

    Pyzotero is required for interacting with the Zotero API to fetch
    bibliographic data and attachments.

    Args:
        operation: Description of the operation requiring pyzotero (for error message)

    Returns:
        The pyzotero module

    Raises:
        MissingDependencyError: If pyzotero is not installed

    Example:
        >>> pyzotero = require_pyzotero()
        >>> zot = pyzotero.zotero.Zotero(library_id, library_type, api_key)
    """
    try:
        import pyzotero

        return pyzotero
    except ImportError as e:
        raise MissingDependencyError(
            f"Pyzotero is required for {operation}.\n\n"
            "Install it with:\n"
            "  pip install pyzotero\n\n"
            "Or install incite with Zotero support:\n"
            "  pip install incite[zotero]  # (if Zotero extras are defined)\n\n"
            "Pyzotero documentation: https://pyzotero.readthedocs.io/"
        ) from e


def try_import_pymupdf() -> Any:
    """Try to import PyMuPDF without raising an error.

    Use this when PDF support is optional and your code can gracefully
    fall back to alternative behavior.

    Returns:
        The fitz module if available, None otherwise

    Example:
        >>> fitz = try_import_pymupdf()
        >>> if fitz:
        ...     doc = fitz.open("paper.pdf")
        ... else:
        ...     # Fall back to text-based extraction
        ...     text = extract_from_txt_file("paper.txt")
    """
    try:
        import fitz

        return fitz
    except ImportError:
        return None


def try_import_pyzotero() -> Any:
    """Try to import pyzotero without raising an error.

    Use this when Zotero API support is optional and your code can
    fall back to alternative sources.

    Returns:
        The pyzotero module if available, None otherwise

    Example:
        >>> pyzotero = try_import_pyzotero()
        >>> if pyzotero:
        ...     zot = pyzotero.zotero.Zotero(...)
        ... else:
        ...     # Fall back to BibTeX import
        ...     papers = load_from_bibtex("library.bib")
    """
    try:
        import pyzotero

        return pyzotero
    except ImportError:
        return None
