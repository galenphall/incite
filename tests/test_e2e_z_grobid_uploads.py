"""E2E tests for GROBID-dependent upload paths (PDF and BibTeX).

These tests require Modal GROBID to be deployed and responsive. They are
intentionally sorted LAST (z_ prefix) because uploading a PDF sets the
library to "processing", which can block other tests that need "ready".

Run:
    pytest tests/test_e2e_z_grobid_uploads.py -v -m e2e -o "addopts=" \
        --api-url https://inciteref.com --timeout=600
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.conftest import e2e_csrf

pytestmark = pytest.mark.e2e

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GROBID_PROBE_TIMEOUT = 90  # allow for Modal cold start + processing
POLL_INTERVAL = 5


def _library_is_processing(session, base_url: str) -> bool:
    """Check if the library is stuck in 'processing' state."""
    try:
        resp = session.get(f"{base_url}/api/library/status", timeout=10)
        if resp.status_code == 200:
            return "processing" in resp.text.lower() and "ready" not in resp.text.lower()
    except Exception:
        pass
    return False


def _wait_for_grobid_processing(session, base_url: str):
    """Wait for GROBID processing, skipping if Modal GROBID is unavailable.

    Waits 90s to allow for Modal cold start + GROBID processing. If library
    is still "processing" after that, Modal GROBID is likely unavailable.
    """
    deadline = time.time() + GROBID_PROBE_TIMEOUT
    while time.time() < deadline:
        resp = session.get(f"{base_url}/api/library/status")
        assert resp.status_code == 200
        if "ready" in resp.text.lower():
            return
        time.sleep(POLL_INTERVAL)

    pytest.skip(
        f"Library still processing after {GROBID_PROBE_TIMEOUT}s — "
        "Modal GROBID is likely unavailable"
    )


class TestGrobidUploads:
    """Tests PDF and BibTeX upload paths that require GROBID processing.

    Sorted last (z_ prefix) to avoid poisoning library status for other tests.
    """

    def test_01_upload_pdf(self, e2e_session, e2e_base_url):
        """Upload a PDF and verify GROBID processing completes."""
        pdf_path = FIXTURES_DIR / "test_paper.pdf"
        if not pdf_path.exists():
            pytest.skip("tests/fixtures/test_paper.pdf not found")

        if _library_is_processing(e2e_session, e2e_base_url):
            pytest.skip("Library stuck in 'processing' — GROBID may be down")

        with open(pdf_path, "rb") as f:
            resp = e2e_session.post(
                f"{e2e_base_url}/api/upload/pdfs",
                files={"files": (pdf_path.name, f, "application/pdf")},
                headers=e2e_csrf(e2e_session),
                allow_redirects=False,
            )
        assert resp.status_code in (200, 303), (
            f"PDF upload failed: {resp.status_code} {resp.text[:300]}"
        )

        _wait_for_grobid_processing(e2e_session, e2e_base_url)

    def test_02_upload_bibtex(self, e2e_session, e2e_base_url):
        """Upload a BibTeX file and verify metadata extraction."""
        bib_path = FIXTURES_DIR / "test_paper.bib"
        if not bib_path.exists():
            pytest.skip("tests/fixtures/test_paper.bib not found")

        if _library_is_processing(e2e_session, e2e_base_url):
            pytest.skip("Library stuck in 'processing' — GROBID may be down")

        with open(bib_path, "rb") as f:
            resp = e2e_session.post(
                f"{e2e_base_url}/api/upload/bibtex",
                files={"bibfile": (bib_path.name, f, "application/x-bibtex")},
                headers=e2e_csrf(e2e_session),
                allow_redirects=False,
            )
        assert resp.status_code in (200, 303), (
            f"BibTeX upload failed: {resp.status_code} {resp.text[:300]}"
        )

        _wait_for_grobid_processing(e2e_session, e2e_base_url)
