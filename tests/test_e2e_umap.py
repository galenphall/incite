"""E2E tests for UMAP visualization endpoint.

Tests the UMAP projection that powers the library map view.
Read-only — no cleanup needed.

Run:
    pytest tests/test_e2e_umap.py -v -m e2e -o "addopts=" \
        --api-url https://inciteref.com --timeout=120
"""

from __future__ import annotations

import pytest

from tests.conftest import e2e_csrf

pytestmark = pytest.mark.e2e


def _get_umap_or_skip(session, base_url: str, **params) -> dict:
    """Fetch UMAP projection, skipping if library is empty/not ready."""
    resp = session.get(f"{base_url}/api/v1/umap", params=params)
    if resp.status_code == 400:
        data = resp.json()
        status = data.get("status", "")
        if status in ("empty", "not_ready"):
            pytest.skip(f"Library {status} — no UMAP data")
    assert resp.status_code == 200, (
        f"UMAP failed: {resp.status_code} {resp.text[:300]}"
    )
    return resp.json()


def _get_points(data: dict) -> list:
    """Extract points list from UMAP response.

    The response structure is {"status": ..., "projection": {"points": [...]}, ...}.
    """
    projection = data.get("projection", {})
    if isinstance(projection, dict):
        return projection.get("points", [])
    return data.get("points") or data.get("papers", [])


class TestUMAP:
    """Tests the UMAP projection endpoint."""

    def test_01_get_umap_projection(self, e2e_session, e2e_base_url):
        """Get UMAP projection and verify response structure."""
        data = _get_umap_or_skip(e2e_session, e2e_base_url)

        points = _get_points(data)
        assert isinstance(points, list), (
            f"Could not extract points from UMAP response: {list(data.keys())}"
        )
        if points:
            p = points[0]
            assert "x" in p, f"Point missing 'x': {list(p.keys())}"
            assert "y" in p, f"Point missing 'y': {list(p.keys())}"

    def test_02_point_count_matches_library(self, e2e_session, e2e_base_url):
        """Verify point count is consistent with library size."""
        resp = e2e_session.get(f"{e2e_base_url}/api/v1/health")
        assert resp.status_code == 200
        health = resp.json()
        corpus_size = health.get("corpus_size") or health.get("paper_count", 0)

        data = _get_umap_or_skip(e2e_session, e2e_base_url)
        points = _get_points(data)

        if corpus_size and points:
            assert len(points) > 0
            assert len(points) <= corpus_size * 1.1, (
                f"More UMAP points ({len(points)}) than papers ({corpus_size})"
            )

    def test_03_verify_collections_in_response(self, e2e_session, e2e_base_url):
        """Check that collections metadata is present in UMAP response."""
        data = _get_umap_or_skip(e2e_session, e2e_base_url)
        assert isinstance(data, dict)
        assert "collections" in data, (
            f"No collections key in UMAP response: {list(data.keys())}"
        )

    def test_04_force_recompute(self, e2e_session, e2e_base_url):
        """Force recompute UMAP and verify still valid."""
        data = _get_umap_or_skip(e2e_session, e2e_base_url, recompute="true")
        points = _get_points(data)
        assert isinstance(points, list)
