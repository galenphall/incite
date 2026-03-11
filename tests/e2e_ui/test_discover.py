"""Test discovery page loads and can run discovery."""
from __future__ import annotations

from playwright.sync_api import Page, expect

from tests.e2e_ui.conftest import is_real_console_error


class TestDiscover:
    def test_discover_page_loads(self, authenticated_page: Page, base_url: str):
        """Discovery page renders."""
        page = authenticated_page
        page.goto(f"{base_url}/web/discover")
        # Should have either the discover button or pre-execution view
        discover_btn = page.locator("#discover-btn")
        pre_view = page.locator("#view-pre")
        assert discover_btn.is_visible() or pre_view.is_visible(), (
            "Neither discover button nor pre-view visible on discovery page"
        )

    def test_discover_runs_without_crash(self, authenticated_page: Page, base_url: str):
        """Running discovery doesn't crash the page."""
        page = authenticated_page
        page.goto(f"{base_url}/web/discover")
        discover_btn = page.locator("#discover-btn")
        if discover_btn.is_visible():
            discover_btn.click()
            # Wait for processing (graph service may take time)
            page.wait_for_timeout(5000)
            # Page should still be functional (body visible = no crash)
            expect(page.locator("body")).to_be_visible()

    def test_no_console_errors(self, authenticated_page: Page, base_url: str):
        """Discovery page has no JS errors."""
        errors = []
        page = authenticated_page
        page.on(
            "console",
            lambda msg: errors.append(msg.text)
            if msg.type == "error" and is_real_console_error(msg.text)
            else None,
        )
        page.goto(f"{base_url}/web/discover")
        page.wait_for_load_state("networkidle")
        assert len(errors) == 0, f"Console errors on discover: {errors}"
