"""Tests for optional dependency error messages."""

import sys
from unittest.mock import patch

import pytest

from incite.utils.dependencies import (
    MissingDependencyError,
    require_pymupdf,
    require_pyzotero,
    try_import_pymupdf,
    try_import_pyzotero,
)


class TestRequireFunctions:
    """Test that require_* functions provide helpful error messages."""

    def test_require_pymupdf_with_missing_dependency(self):
        """require_pymupdf should raise MissingDependencyError with install instructions."""
        with patch.dict(sys.modules, {"fitz": None}):
            with pytest.raises(MissingDependencyError) as exc_info:
                require_pymupdf()

            error_message = str(exc_info.value)
            assert "PyMuPDF is required" in error_message
            assert "pip install pymupdf" in error_message
            assert "https://pymupdf.readthedocs.io/" in error_message

    def test_require_pymupdf_with_custom_operation(self):
        """Error message should include the custom operation description."""
        with patch.dict(sys.modules, {"fitz": None}):
            with pytest.raises(MissingDependencyError) as exc_info:
                require_pymupdf("extracting academic PDFs")

            error_message = str(exc_info.value)
            assert "extracting academic PDFs" in error_message

    def test_require_pyzotero_with_missing_dependency(self):
        """require_pyzotero should raise MissingDependencyError with install instructions."""
        with patch.dict(sys.modules, {"pyzotero": None}):
            with pytest.raises(MissingDependencyError) as exc_info:
                require_pyzotero()

            error_message = str(exc_info.value)
            assert "Pyzotero is required" in error_message
            assert "pip install pyzotero" in error_message
            assert "https://pyzotero.readthedocs.io/" in error_message

    def test_require_pyzotero_with_custom_operation(self):
        """Error message should include the custom operation description."""
        with patch.dict(sys.modules, {"pyzotero": None}):
            with pytest.raises(MissingDependencyError) as exc_info:
                require_pyzotero("syncing bibliography")

            error_message = str(exc_info.value)
            assert "syncing bibliography" in error_message


class TestTryImportFunctions:
    """Test that try_import_* functions handle missing dependencies gracefully."""

    def test_try_import_pymupdf_returns_none_when_missing(self):
        """try_import_pymupdf should return None instead of raising an error."""
        with patch.dict(sys.modules, {"fitz": None}):
            result = try_import_pymupdf()
            assert result is None

    def test_try_import_pyzotero_returns_none_when_missing(self):
        """try_import_pyzotero should return None instead of raising an error."""
        with patch.dict(sys.modules, {"pyzotero": None}):
            result = try_import_pyzotero()
            assert result is None


class TestErrorHierarchy:
    """Test that custom errors inherit from ImportError."""

    def test_missing_dependency_error_is_import_error(self):
        """MissingDependencyError should be a subclass of ImportError."""
        assert issubclass(MissingDependencyError, ImportError)

    def test_can_catch_as_import_error(self):
        """Code catching ImportError should still work."""
        with patch.dict(sys.modules, {"fitz": None}):
            with pytest.raises(ImportError):
                require_pymupdf()
