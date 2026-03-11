"""E2E tests for the JSON API surface used by all editor plugins.

Tests the API contract that Chrome extension, VS Code, Obsidian, Word,
and Google Docs plugins depend on. Exercises token auth, recommendations,
library operations, settings sync, and paper save/check.

Run:
    pytest tests/test_e2e_plugin_api.py -v -m e2e -o "addopts=" \
        --api-url https://inciteref.com --timeout=120
"""

from __future__ import annotations

import time

import pytest
import requests

from tests.conftest import e2e_csrf

pytestmark = pytest.mark.e2e


def _require(value, msg="prerequisite test did not pass"):
    if not value:
        pytest.skip(msg)
    return value


class TestPluginAPI:
    """Tests the JSON API surface all editor plugins depend on."""

    _api_token: str | None = None
    _saved_paper_id: str | None = None
    _original_settings: dict | None = None

    def test_01_create_api_token(self, e2e_session, e2e_base_url):
        """Create a temporary API token for Bearer auth."""
        resp = e2e_session.post(
            f"{e2e_base_url}/api/v1/token",
            json={"name": f"e2e-plugin-test-{int(time.time())}"},
            headers=e2e_csrf(e2e_session),
        )
        assert resp.status_code == 200, f"Token creation failed: {resp.text[:300]}"
        data = resp.json()
        assert "token" in data
        assert data["token"].startswith("mc_")
        self.__class__._api_token = data["token"]

    def test_02_health_check_bearer(self, e2e_base_url):
        """Health check via Bearer token (no session cookie)."""
        token = _require(self.__class__._api_token, "no API token")

        resp = requests.get(
            f"{e2e_base_url}/api/v1/health",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") in (
            "ready", "healthy", "processing", "empty", "error", "pending", "no_library",
        ), f"Unexpected health status: {data}"

    def test_03_recommend_bearer(self, e2e_base_url):
        """JSON recommend via Bearer token — verify response schema."""
        token = _require(self.__class__._api_token, "no API token")

        resp = requests.post(
            f"{e2e_base_url}/api/v1/recommend",
            json={"query": "The role of ocean acidification in coral reef decline", "k": 5},
            headers={"Authorization": f"Bearer {token}"},
        )
        # 200 if library has papers, 503 if processing/empty
        assert resp.status_code in (200, 503), f"Unexpected: {resp.status_code}"
        data = resp.json()

        if resp.status_code == 200:
            assert "recommendations" in data
            assert "timing" in data
            assert isinstance(data["recommendations"], list)
            if data["recommendations"]:
                rec = data["recommendations"][0]
                assert "title" in rec
                assert "score" in rec or "rrf_score" in rec

    def test_04_save_paper(self, e2e_base_url):
        """Save a paper from Chrome extension (POST /api/v1/library/papers)."""
        token = _require(self.__class__._api_token, "no API token")

        resp = requests.post(
            f"{e2e_base_url}/api/v1/library/papers",
            json={
                "papers": [{
                    "title": "E2E Test Paper — Plugin API Save",
                    "authors": ["Test, Author"],
                    "year": 2024,
                    "doi": "10.9999/e2e-plugin-test",
                    "abstract": "A synthetic paper created by e2e plugin API tests.",
                }],
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 201, f"Save paper failed: {resp.text[:300]}"
        data = resp.json()
        # Response has "saved", "already_existed", "errors" lists
        saved = data.get("saved", [])
        already = data.get("already_existed", [])
        all_papers = saved + already
        assert all_papers, f"No papers in response: {data}"
        paper_id = all_papers[0].get("canonical_id") or all_papers[0].get("id")
        assert paper_id, f"No paper ID in response: {all_papers[0]}"
        self.__class__._saved_paper_id = paper_id

    def test_05_check_paper_exists(self, e2e_base_url):
        """Check that saved paper is in library (POST /api/v1/library/check)."""
        token = _require(self.__class__._api_token, "no API token")

        resp = requests.post(
            f"{e2e_base_url}/api/v1/library/check",
            json={"papers": [{"doi": "10.9999/e2e-plugin-test"}]},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        results = data.get("results", [])
        assert results, f"No results in library check: {data}"
        assert any(r.get("in_library") for r in results), (
            f"Paper not found in library check: {results}"
        )

    def test_06_settings_read(self, e2e_base_url):
        """Read user settings via Bearer token."""
        token = _require(self.__class__._api_token, "no API token")

        resp = requests.get(
            f"{e2e_base_url}/api/v1/settings",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        self.__class__._original_settings = data.get("settings", data)

    def test_07_settings_write_readback(self, e2e_base_url):
        """Write a setting, read it back, then restore original."""
        token = _require(self.__class__._api_token, "no API token")

        # Write k=20
        resp = requests.put(
            f"{e2e_base_url}/api/v1/settings",
            json={"k": 20},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

        # Read back
        resp = requests.get(
            f"{e2e_base_url}/api/v1/settings",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        settings = data.get("settings", data)
        assert settings.get("k") == 20, f"Settings readback mismatch: {data}"

    def test_08_list_collections(self, e2e_base_url):
        """List collections via Bearer token."""
        token = _require(self.__class__._api_token, "no API token")

        resp = requests.get(
            f"{e2e_base_url}/api/v1/library/collections",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, (list, dict)), f"Unexpected collections format: {type(data)}"

    def test_09_tag_search(self, e2e_base_url):
        """Search tags via Bearer token."""
        token = _require(self.__class__._api_token, "no API token")

        resp = requests.get(
            f"{e2e_base_url}/api/v1/library/tags/search",
            params={"q": "test"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, (list, dict))

    def test_10_library_papers_sync(self, e2e_base_url):
        """Full paper list sync via Bearer token."""
        token = _require(self.__class__._api_token, "no API token")

        resp = requests.get(
            f"{e2e_base_url}/api/v1/library/papers",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, (list, dict))

    def test_99_cleanup(self, e2e_session, e2e_base_url):
        """Revoke API token, delete saved paper, restore settings."""
        headers = e2e_csrf(e2e_session)
        token = self.__class__._api_token

        # Restore original settings
        if self.__class__._original_settings and token:
            requests.put(
                f"{e2e_base_url}/api/v1/settings",
                json=self.__class__._original_settings,
                headers={"Authorization": f"Bearer {token}"},
            )

        # Delete saved test paper
        if self.__class__._saved_paper_id:
            e2e_session.delete(
                f"{e2e_base_url}/api/refman/papers/{self.__class__._saved_paper_id}",
                headers=headers,
            )

        # Revoke API token
        if token:
            e2e_session.delete(
                f"{e2e_base_url}/api/v1/token/{token}",
                headers=headers,
            )
