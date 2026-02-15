"""Real integration tests against the production cloud service.

These tests run against the live server at inciteref.com (or --api-url).
They exercise the full stack: login, PDF upload, GROBID processing,
recommendations, and reference manager operations.

Prerequisites:
    1. Test account: env CLOUD_TEST_EMAIL / CLOUD_TEST_PASSWORD
    2. ~/Zotero/storage with at least 1 PDF <2MB (for upload test)
    3. GROBID running on the server (for processing test)

Run:
    pytest tests/test_cloud_integration.py -v -m e2e -o "addopts=" \\
        --api-url https://inciteref.com

Configure via environment variables:
    CLOUD_TEST_EMAIL     (default: cloudtest@inciteref.com)
    CLOUD_TEST_PASSWORD  (required for e2e)
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pytest
import requests

pytestmark = pytest.mark.e2e

MAX_PDF_SIZE = 2 * 1024 * 1024  # 2 MB
PROCESSING_TIMEOUT = 300  # 5 min
POLL_INTERVAL = 5  # seconds


def _find_one_small_pdf() -> Path | None:
    """Find a single small PDF from Zotero storage."""
    zotero_storage = Path.home() / "Zotero" / "storage"
    if not zotero_storage.exists():
        return None
    for storage_dir in zotero_storage.iterdir():
        if not storage_dir.is_dir():
            continue
        for pdf in storage_dir.glob("*.pdf"):
            if pdf.stat().st_size <= MAX_PDF_SIZE:
                return pdf
    return None


@pytest.fixture(scope="module")
def base_url(request) -> str:
    url = request.config.getoption("--api-url", default="https://inciteref.com")
    return url.rstrip("/")


@pytest.fixture(scope="module")
def test_credentials() -> dict:
    email = os.environ.get("CLOUD_TEST_EMAIL", "cloudtest@inciteref.com")
    password = os.environ.get("CLOUD_TEST_PASSWORD", "")
    if not password:
        pytest.skip("CLOUD_TEST_PASSWORD not set — cannot run integration tests")
    return {"email": email, "password": password}


@pytest.fixture(scope="module")
def session(base_url, test_credentials) -> requests.Session:
    """Create an authenticated requests session."""
    s = requests.Session()
    s.headers.update({"User-Agent": "incite-integration-test/1.0"})

    # Get CSRF token from login page
    s.get(f"{base_url}/web/login")
    csrf_token = s.cookies.get("csrf_token", "")

    # Login
    resp = s.post(
        f"{base_url}/web/login",
        data={
            "email": test_credentials["email"],
            "password": test_credentials["password"],
            "csrf_token": csrf_token,
        },
        headers={"X-CSRF-Token": csrf_token},
        allow_redirects=False,
    )
    if resp.status_code != 303:
        pytest.fail(f"Login failed with status {resp.status_code}: {resp.text[:500]}")

    return s


def _csrf(session: requests.Session) -> tuple[dict, dict]:
    """Return (headers, cookies) dicts with CSRF token."""
    token = session.cookies.get("csrf_token", "")
    return {"X-CSRF-Token": token}, {"csrf_token": token}


class TestCloudIntegration:
    """Sequential integration tests against production."""

    # Shared state across tests
    _api_token: str | None = None
    _canonical_ids: list[str] = []
    _tag_ids: list[int] = []
    _note_ids: list[int] = []
    _collection_ids: list[int] = []

    def test_01_login_and_health(self, session, base_url):
        """Verify login succeeded and server is healthy."""
        assert session.cookies.get("session"), "No session cookie after login"

        resp = session.get(f"{base_url}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") in ("healthy", "degraded")

    def test_02_upload_and_process(self, session, base_url):
        """Upload a PDF and wait for GROBID processing to complete."""
        headers, cookies = _csrf(session)
        pdf_path = _find_one_small_pdf()
        if pdf_path is None:
            pytest.skip("No small PDF found in ~/Zotero/storage")

        # Upload PDF
        with open(pdf_path, "rb") as f:
            resp = session.post(
                f"{base_url}/api/upload/pdfs",
                files={"files": (pdf_path.name, f, "application/pdf")},
                headers=headers,
                cookies=cookies,
                allow_redirects=False,
            )
        # Upload may redirect to setup-status (303) or return 200
        assert resp.status_code in (200, 303), (
            f"Upload failed: {resp.status_code} {resp.text[:300]}"
        )

        # Poll for processing completion
        deadline = time.time() + PROCESSING_TIMEOUT
        while time.time() < deadline:
            resp = session.get(f"{base_url}/api/library/status")
            assert resp.status_code == 200
            # The status partial contains the library status
            if "ready" in resp.text.lower():
                return  # Processing complete
            time.sleep(POLL_INTERVAL)

        pytest.fail("Library processing did not complete within timeout")

    def test_03_recommend_html(self, session, base_url):
        """Make a recommendation request through the HTML API."""
        headers, cookies = _csrf(session)
        resp = session.post(
            f"{base_url}/api/recommend",
            data={
                "query": "The impact of rising temperatures on marine ecosystems",
                "k": "5",
                "csrf_token": cookies.get("csrf_token", ""),
            },
            headers=headers,
        )
        assert resp.status_code == 200
        assert len(resp.text) > 50, "Response too short"

    def test_04_refman_workflow(self, session, base_url):
        """Star a paper, add tags/notes, verify on detail page."""
        headers, cookies = _csrf(session)

        # Get a paper ID from the library
        resp = session.get(f"{base_url}/api/library/papers")
        assert resp.status_code == 200

        links = re.findall(r'/web/papers/([a-zA-Z0-9_%-]+)"', resp.text)
        if not links:
            pytest.skip("No papers found in library — cannot test refman")

        canonical_id = links[0]
        self.__class__._canonical_ids.append(canonical_id)

        # Star the paper
        resp = session.post(
            f"{base_url}/api/refman/papers/{canonical_id}/star",
            headers=headers,
            cookies=cookies,
        )
        assert resp.status_code == 200

        # Create a tag
        resp = session.post(
            f"{base_url}/api/refman/tags",
            json={"name": f"test-tag-{int(time.time())}"},
            headers=headers,
            cookies=cookies,
        )
        assert resp.status_code == 200
        tag_id = resp.json()["id"]
        self.__class__._tag_ids.append(tag_id)

        # Add tag to paper
        resp = session.post(
            f"{base_url}/api/refman/papers/{canonical_id}/tags/{tag_id}",
            headers=headers,
            cookies=cookies,
        )
        assert resp.status_code == 200

        # Create a note
        resp = session.post(
            f"{base_url}/api/refman/papers/{canonical_id}/notes",
            data={
                "title": "Integration test note",
                "content_md": "Created by integration test suite.",
                "csrf_token": cookies.get("csrf_token", ""),
            },
            headers=headers,
        )
        assert resp.status_code == 200
        note_id = resp.json()["id"]
        self.__class__._note_ids.append(note_id)

        # Verify paper detail page loads
        resp = session.get(f"{base_url}/web/papers/{canonical_id}")
        assert resp.status_code == 200

    def test_05_json_api_recommend(self, session, base_url):
        """Create an API token and use it for JSON recommendations."""
        headers, cookies = _csrf(session)

        # Create API token
        resp = session.post(
            f"{base_url}/api/v1/token",
            json={"name": "integration-test"},
            headers=headers,
            cookies=cookies,
        )
        assert resp.status_code == 200
        api_token = resp.json()["token"]
        assert api_token.startswith("mc_")
        self.__class__._api_token = api_token

        # Use Bearer token for JSON recommend (no CSRF needed)
        resp = requests.post(
            f"{base_url}/api/v1/recommend",
            json={"query": "ocean acidification", "k": 3},
            headers={"Authorization": f"Bearer {api_token}"},
        )
        # 200 if library has papers, 503 if empty — both are valid
        assert resp.status_code in (200, 503)
        data = resp.json()
        if resp.status_code == 200:
            assert "recommendations" in data
            assert "timing" in data
            assert isinstance(data["recommendations"], list)
        else:
            assert "error" in data

    def test_06_cleanup(self, session, base_url):
        """Clean up test artifacts created during the run."""
        headers, cookies = _csrf(session)

        # Delete notes
        for note_id in self.__class__._note_ids:
            session.delete(
                f"{base_url}/api/refman/notes/{note_id}",
                headers=headers,
                cookies=cookies,
            )

        # Remove tags from papers, then delete tags
        for canonical_id in self.__class__._canonical_ids:
            for tag_id in self.__class__._tag_ids:
                session.delete(
                    f"{base_url}/api/refman/papers/{canonical_id}/tags/{tag_id}",
                    headers=headers,
                    cookies=cookies,
                )
        for tag_id in self.__class__._tag_ids:
            session.delete(
                f"{base_url}/api/refman/tags/{tag_id}",
                headers=headers,
                cookies=cookies,
            )

        # Delete collections
        for coll_id in self.__class__._collection_ids:
            session.delete(
                f"{base_url}/api/refman/collections/{coll_id}",
                headers=headers,
                cookies=cookies,
            )

        # Revoke API token
        if self.__class__._api_token:
            session.delete(
                f"{base_url}/api/v1/token/{self.__class__._api_token}",
                headers=headers,
                cookies=cookies,
            )

        # Un-star papers
        for canonical_id in self.__class__._canonical_ids:
            session.post(
                f"{base_url}/api/refman/papers/{canonical_id}/star",
                headers=headers,
                cookies=cookies,
            )
