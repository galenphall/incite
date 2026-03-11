"""Playwright UI test fixtures.

Uses the same test accounts as API e2e tests:
- cloudtest@inciteref.com (primary)
- cloudtest-b@inciteref.com (secondary)

SAFETY: Never create new accounts. Read-heavy, write-light.
"""
from __future__ import annotations

import os

import pytest
from playwright.sync_api import Page, expect

# Third-party scripts injected by CDN/proxy (not our code) — ignore in console checks
_IGNORE_CONSOLE_PATTERNS = [
    "cloudflareinsights.com",
    "Content Security Policy",
]


def is_real_console_error(msg_text: str) -> bool:
    """Return True if a console error is from our code, not third-party noise."""
    return not any(p in msg_text for p in _IGNORE_CONSOLE_PATTERNS)


def pytest_collection_modifyitems(items):
    """Auto-mark all tests in this directory as e2e_ui."""
    for item in items:
        if "e2e_ui" in str(item.fspath):
            item.add_marker(pytest.mark.e2e_ui)


@pytest.fixture(scope="session")
def base_url():
    """Base URL for UI tests. Defaults to production."""
    return os.environ.get("E2E_BASE_URL", "https://inciteref.com")


@pytest.fixture(scope="session")
def test_credentials():
    """Primary test account credentials."""
    email = os.environ.get("CLOUD_TEST_EMAIL", "cloudtest@inciteref.com")
    password = os.environ.get("CLOUD_TEST_PASSWORD")
    if not password:
        pytest.skip("CLOUD_TEST_PASSWORD not set")
    return {"email": email, "password": password}


@pytest.fixture(scope="session")
def test_credentials_b():
    """Secondary test account credentials."""
    email = os.environ.get("CLOUD_TEST_EMAIL_B", "cloudtest-b@inciteref.com")
    password = os.environ.get("CLOUD_TEST_PASSWORD_B")
    if not password:
        pytest.skip("CLOUD_TEST_PASSWORD_B not set")
    return {"email": email, "password": password}


@pytest.fixture
def authenticated_page(page: Page, base_url: str, test_credentials: dict):
    """A Playwright page logged in as the primary test user."""
    page.goto(f"{base_url}/web/login")
    page.fill('input[name="email"]', test_credentials["email"])
    page.fill('input[name="password"]', test_credentials["password"])
    page.click('button[type="submit"]')
    page.wait_for_url("**/web/library**", timeout=10000)
    return page
