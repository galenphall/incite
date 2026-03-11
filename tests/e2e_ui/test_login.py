"""Test login flow renders correctly and authenticates."""
from __future__ import annotations

from playwright.sync_api import Page, expect

from tests.e2e_ui.conftest import is_real_console_error


class TestLogin:
    def test_login_page_renders(self, page: Page, base_url: str):
        """Login page loads with email/password fields."""
        page.goto(f"{base_url}/web/login")
        expect(page.locator('input[name="email"]')).to_be_visible()
        expect(page.locator('input[name="password"]')).to_be_visible()
        expect(page.locator('button[type="submit"]')).to_be_visible()

    def test_login_success_redirects_to_library(
        self, page: Page, base_url: str, test_credentials: dict
    ):
        """Successful login redirects to library page."""
        page.goto(f"{base_url}/web/login")
        page.fill('input[name="email"]', test_credentials["email"])
        page.fill('input[name="password"]', test_credentials["password"])
        page.click('button[type="submit"]')
        page.wait_for_url("**/web/library**", timeout=10000)

    def test_login_failure_shows_error(self, page: Page, base_url: str):
        """Bad credentials don't redirect to library."""
        page.goto(f"{base_url}/web/login")
        page.fill('input[name="email"]', "wrong@example.com")
        page.fill('input[name="password"]', "wrongpassword")
        page.click('button[type="submit"]')
        page.wait_for_timeout(2000)
        # Should stay on login page (not redirect to library)
        assert "/web/library" not in page.url

    def test_no_console_errors_on_login(self, page: Page, base_url: str):
        """Login page has no JavaScript console errors."""
        errors = []
        page.on(
            "console",
            lambda msg: errors.append(msg.text)
            if msg.type == "error" and is_real_console_error(msg.text)
            else None,
        )
        page.goto(f"{base_url}/web/login")
        page.wait_for_load_state("networkidle")
        assert len(errors) == 0, f"Console errors on login page: {errors}"
