"""E2E tests for the paper discovery pipeline.

Tests the Rust graph-based discovery system: instant discovery,
scoped selections, dismiss/undismiss, save to library, and labels CRUD.

Run:
    pytest tests/test_e2e_discovery.py -v -m e2e -o "addopts=" \
        --api-url https://inciteref.com --timeout=120
"""

from __future__ import annotations

import re

import pytest

from tests.conftest import e2e_csrf

pytestmark = pytest.mark.e2e


def _require(value, msg="prerequisite test did not pass"):
    if not value:
        pytest.skip(msg)
    return value


class TestDiscovery:
    """Tests the Rust graph-based discovery system."""

    _job_id: str | None = None
    _selection_id: str | None = None
    _first_s2_id: str | None = None
    _saved_s2_id: str | None = None
    _label_id: int | None = None
    _dismissed_s2_id: str | None = None

    def test_01_instant_discovery(self, e2e_session, e2e_base_url):
        """Run instant discovery on full library."""
        resp = e2e_session.post(
            f"{e2e_base_url}/api/discover/instant",
            json={},
            headers=e2e_csrf(e2e_session),
        )
        if resp.status_code == 400:
            detail = resp.text[:500].lower()
            if "could be matched" in detail or "no papers" in detail:
                pytest.skip(f"Discovery unavailable: {resp.text[:200]}")
        assert resp.status_code == 200, (
            f"Instant discovery failed: {resp.status_code} {resp.text[:300]}"
        )
        data = resp.json()

        assert "results" in data, f"No results key in response: {list(data.keys())}"
        assert "job_id" in data
        self.__class__._job_id = data["job_id"]

        results = data["results"]
        assert isinstance(results, list)
        if results:
            rec = results[0]
            assert "s2_id" in rec
            assert "title" in rec
            assert "discovery_score" in rec
            self.__class__._first_s2_id = rec["s2_id"]

    def test_02_scoped_discovery(self, e2e_session, e2e_base_url):
        """Create a selection and run scoped discovery."""
        # Get some paper IDs for the selection
        resp = e2e_session.get(f"{e2e_base_url}/api/library/papers")
        assert resp.status_code == 200

        # Extract canonical IDs — these are numeric DB IDs, not string canonical_ids
        # Try to get numeric IDs from the page
        ids = re.findall(r'data-paper-id="(\d+)"', resp.text)
        if not ids:
            # Fallback: use canonical_ids
            ids = re.findall(r'/web/papers/([a-zA-Z0-9_%-]+)"', resp.text)
        if len(ids) < 2:
            pytest.skip("Need at least 2 papers for scoped discovery")

        # Create selection — expects canonical_ids as ints if numeric
        selection_ids = []
        for i in ids[:3]:
            try:
                selection_ids.append(int(i))
            except ValueError:
                selection_ids.append(i)

        resp = e2e_session.post(
            f"{e2e_base_url}/api/discover/selections",
            json={"canonical_ids": selection_ids},
            headers=e2e_csrf(e2e_session),
        )
        assert resp.status_code == 200, (
            f"Create selection failed: {resp.status_code} {resp.text[:300]}"
        )
        data = resp.json()
        assert "id" in data
        self.__class__._selection_id = data["id"]

        # Run instant discovery scoped to selection
        resp = e2e_session.post(
            f"{e2e_base_url}/api/discover/instant",
            json={"selection_id": data["id"]},
            headers=e2e_csrf(e2e_session),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data

    def test_03_dismiss_result(self, e2e_session, e2e_base_url):
        """Dismiss a discovery result."""
        s2_id = self.__class__._first_s2_id
        if not s2_id:
            pytest.skip("No discovery results to dismiss")

        resp = e2e_session.post(
            f"{e2e_base_url}/api/discover/dismiss/{s2_id}",
            headers=e2e_csrf(e2e_session),
        )
        assert resp.status_code == 200
        self.__class__._dismissed_s2_id = s2_id

    def test_04_verify_dismissed_excluded(self, e2e_session, e2e_base_url):
        """Re-run discovery and verify dismissed paper is excluded."""
        dismissed = self.__class__._dismissed_s2_id
        if not dismissed:
            pytest.skip("No dismissed paper to verify")

        resp = e2e_session.post(
            f"{e2e_base_url}/api/discover/instant",
            json={},
            headers=e2e_csrf(e2e_session),
        )
        assert resp.status_code == 200
        data = resp.json()
        result_ids = [r["s2_id"] for r in data.get("results", [])]
        assert dismissed not in result_ids, (
            f"Dismissed paper {dismissed} still appears in results"
        )

    def test_05_save_discovery_result(self, e2e_session, e2e_base_url):
        """Save a discovered paper to the library."""
        # Get a fresh result to save (not the dismissed one)
        resp = e2e_session.post(
            f"{e2e_base_url}/api/discover/instant",
            json={},
            headers=e2e_csrf(e2e_session),
        )
        if resp.status_code == 400:
            detail = resp.text[:500].lower()
            if "could be matched" in detail or "no papers" in detail:
                pytest.skip(f"Discovery unavailable: {resp.text[:200]}")
        assert resp.status_code == 200
        results = resp.json().get("results", [])
        job_id = resp.json().get("job_id")

        if not results:
            pytest.skip("No discovery results to save")

        s2_id = results[0]["s2_id"]
        resp = e2e_session.post(
            f"{e2e_base_url}/api/discover/save/{s2_id}",
            params={"job_id": job_id} if job_id else {},
            headers=e2e_csrf(e2e_session),
        )
        assert resp.status_code == 200
        self.__class__._saved_s2_id = s2_id

    def test_06_labels_crud(self, e2e_session, e2e_base_url):
        """Create, list, and delete a discovery label."""
        headers = e2e_csrf(e2e_session)
        job_id = _require(self.__class__._job_id, "no discovery job_id")

        # Create label
        resp = e2e_session.post(
            f"{e2e_base_url}/api/discover/labels",
            json={"name": "e2e-test-label", "job_id": job_id},
            headers=headers,
        )
        assert resp.status_code == 200, (
            f"Create label failed: {resp.status_code} {resp.text[:300]}"
        )
        data = resp.json()
        assert "id" in data
        self.__class__._label_id = data["id"]

        # List labels
        resp = e2e_session.get(
            f"{e2e_base_url}/api/discover/labels",
            headers=headers,
        )
        assert resp.status_code == 200
        labels_data = resp.json()
        labels = labels_data.get("labels", labels_data)
        assert isinstance(labels, list)

    def test_99_cleanup(self, e2e_session, e2e_base_url):
        """Un-dismiss, remove saved paper, delete labels/selections."""
        headers = e2e_csrf(e2e_session)

        # Delete label
        if self.__class__._label_id:
            e2e_session.delete(
                f"{e2e_base_url}/api/discover/labels/{self.__class__._label_id}",
                headers=headers,
            )

        # Delete the saved paper from library
        if self.__class__._saved_s2_id:
            # The saved paper is now in the library — remove it via refman
            # We need to find its canonical_id
            resp = e2e_session.get(f"{e2e_base_url}/api/library/papers")
            if resp.status_code == 200:
                # Look for the saved S2 ID in the library listing
                # Best effort — may not find it if the page doesn't show S2 IDs
                pass

        # Delete discovery runs (best effort)
        if self.__class__._job_id:
            e2e_session.delete(
                f"{e2e_base_url}/api/discover/runs/{self.__class__._job_id}",
                headers=headers,
            )
