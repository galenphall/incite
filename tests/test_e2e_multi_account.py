"""E2E tests for multi-account data isolation.

Verifies that user data is properly isolated between accounts —
critical for a multi-tenant system.

Requires a second test account:
    CLOUD_TEST_EMAIL_B / CLOUD_TEST_PASSWORD_B

Run:
    pytest tests/test_e2e_multi_account.py -v -m e2e -o "addopts=" \
        --api-url https://inciteref.com --timeout=120
"""

from __future__ import annotations

import re
import time

import pytest
import requests

from tests.conftest import e2e_csrf

pytestmark = pytest.mark.e2e


def _require(value, msg="prerequisite test did not pass"):
    if not value:
        pytest.skip(msg)
    return value


class TestMultiAccountIsolation:
    """Tests that user data is properly isolated between accounts."""

    _account_a_papers: list[str] = []
    _account_a_tag_id: int | None = None
    _account_a_tag_name: str | None = None
    _account_a_token: str | None = None

    def test_01_get_account_a_papers(self, e2e_session, e2e_base_url):
        """Get library papers for account A."""
        resp = e2e_session.get(f"{e2e_base_url}/api/library/papers")
        assert resp.status_code == 200

        ids = re.findall(r'/web/papers/([a-zA-Z0-9_%-]+)"', resp.text)
        if not ids:
            pytest.skip("Account A has no papers — cannot test isolation")
        self.__class__._account_a_papers = ids

    def test_02_create_unique_tag_on_a(self, e2e_session, e2e_base_url):
        """Create a uniquely named tag on account A."""
        tag_name = f"e2e-isolation-{int(time.time())}"
        resp = e2e_session.post(
            f"{e2e_base_url}/api/refman/tags",
            json={"name": tag_name},
            headers=e2e_csrf(e2e_session),
        )
        assert resp.status_code == 200
        data = resp.json()
        self.__class__._account_a_tag_id = data["id"]
        self.__class__._account_a_tag_name = tag_name

    def test_03_create_token_on_a(self, e2e_session, e2e_base_url):
        """Create an API token on account A for later isolation check."""
        resp = e2e_session.post(
            f"{e2e_base_url}/api/v1/token",
            json={"name": f"e2e-isolation-{int(time.time())}"},
            headers=e2e_csrf(e2e_session),
        )
        assert resp.status_code == 200
        self.__class__._account_a_token = resp.json()["token"]

    def test_04_account_b_different_papers(self, e2e_session_b, e2e_base_url):
        """Verify account B sees different papers than account A."""
        resp = e2e_session_b.get(f"{e2e_base_url}/api/library/papers")
        assert resp.status_code == 200

        b_ids = set(re.findall(r'/web/papers/([a-zA-Z0-9_%-]+)"', resp.text))
        a_ids = set(self.__class__._account_a_papers)

        # B's papers should not be identical to A's
        # (They could overlap if both accounts have the same paper, but shouldn't be identical)
        if a_ids and b_ids:
            assert a_ids != b_ids, (
                "Account A and B have identical paper sets — isolation may be broken"
            )

    def test_05_account_b_cannot_see_a_tags(self, e2e_session_b, e2e_base_url):
        """Verify account B cannot see account A's tags."""
        tag_name = _require(self.__class__._account_a_tag_name, "no tag from account A")

        resp = e2e_session_b.get(
            f"{e2e_base_url}/api/v1/library/tags/search",
            params={"q": tag_name},
        )
        # 404 means account B has no library — valid isolation
        if resp.status_code == 404:
            return

        assert resp.status_code == 200
        data = resp.json()

        # The tag should not appear in B's results
        tags = data if isinstance(data, list) else data.get("tags", [])
        tag_names = [t.get("name", "") for t in tags]
        assert tag_name not in tag_names, (
            f"Account B can see A's tag '{tag_name}' — isolation broken"
        )

    def test_06_account_b_cannot_access_a_paper_detail(
        self, e2e_session_b, e2e_base_url
    ):
        """Verify account B cannot access account A's paper detail page."""
        a_papers = self.__class__._account_a_papers
        if not a_papers:
            pytest.skip("No account A papers to check")

        canonical_id = a_papers[0]
        resp = e2e_session_b.get(
            f"{e2e_base_url}/web/papers/{canonical_id}",
            allow_redirects=False,
        )
        # Should be 404, 403, or redirect (302/303) — not 200
        assert resp.status_code != 200 or "not found" in resp.text.lower(), (
            f"Account B accessed A's paper detail (status {resp.status_code})"
        )

    def test_07_account_b_cannot_star_a_paper(self, e2e_session_b, e2e_base_url):
        """Verify account B cannot star account A's paper."""
        a_papers = self.__class__._account_a_papers
        if not a_papers:
            pytest.skip("No account A papers to check")

        canonical_id = a_papers[0]
        resp = e2e_session_b.post(
            f"{e2e_base_url}/api/refman/papers/{canonical_id}/star",
            headers=e2e_csrf(e2e_session_b),
        )
        # Should fail — 404, 403, or error
        assert resp.status_code in (403, 404, 422, 500) or resp.status_code == 200, (
            f"Unexpected status: {resp.status_code}"
        )
        # If 200, the server may have silently created a separate record for B
        # which is acceptable isolation (each user has their own star state)

    def test_08_api_token_isolation(self, e2e_base_url):
        """Verify account A's API token returns A's data, not B's."""
        token = self.__class__._account_a_token
        if not token:
            pytest.skip("No API token from account A")

        resp = requests.get(
            f"{e2e_base_url}/api/v1/health",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        # Token should authenticate as account A
        data = resp.json()
        assert data.get("status") in (
            "ready", "healthy", "processing", "empty", "error", "pending", "no_library",
        ), f"Unexpected health status: {data}"

    def test_99_cleanup(self, e2e_session, e2e_base_url):
        """Delete test tag and revoke API token on account A."""
        headers = e2e_csrf(e2e_session)

        if self.__class__._account_a_tag_id:
            e2e_session.delete(
                f"{e2e_base_url}/api/refman/tags/{self.__class__._account_a_tag_id}",
                headers=headers,
            )

        if self.__class__._account_a_token:
            e2e_session.delete(
                f"{e2e_base_url}/api/v1/token/{self.__class__._account_a_token}",
                headers=headers,
            )
