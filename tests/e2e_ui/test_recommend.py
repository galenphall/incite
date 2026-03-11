"""Test recommendation page accepts queries and returns results."""
from __future__ import annotations

from playwright.sync_api import Page, expect

from tests.e2e_ui.conftest import is_real_console_error


class TestRecommend:
    def test_recommend_page_loads(self, authenticated_page: Page, base_url: str):
        """Recommendation page renders with query input."""
        page = authenticated_page
        page.goto(f"{base_url}/web/recommend")
        expect(page.locator('textarea[name="query"]')).to_be_visible()
        expect(page.locator('button[type="submit"]')).to_be_visible()

    def test_recommend_returns_results(self, authenticated_page: Page, base_url: str):
        """Submitting a query returns recommendation cards."""
        page = authenticated_page
        page.goto(f"{base_url}/web/recommend")
        page.fill(
            'textarea[name="query"]',
            "Recent advances in transformer architectures have shown significant improvements",
        )
        page.click('button[type="submit"]')
        results = page.locator("#results")
        expect(results).to_be_visible(timeout=30000)

    def test_no_console_errors(self, authenticated_page: Page, base_url: str):
        """Recommendation page has no JS errors."""
        errors = []
        page = authenticated_page
        page.on(
            "console",
            lambda msg: errors.append(msg.text)
            if msg.type == "error" and is_real_console_error(msg.text)
            else None,
        )
        page.goto(f"{base_url}/web/recommend")
        page.wait_for_load_state("networkidle")
        assert len(errors) == 0, f"Console errors on recommend: {errors}"
