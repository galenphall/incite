"""Test library page renders papers and supports interaction."""
from __future__ import annotations

from playwright.sync_api import Page, expect

from tests.e2e_ui.conftest import is_real_console_error


class TestLibrary:
    def test_library_loads_with_papers(self, authenticated_page: Page):
        """Library page shows paper cards after login."""
        page = authenticated_page
        expect(page.locator("#paper-list")).to_be_visible()
        cards = page.locator(".card-interactive")
        expect(cards.first).to_be_visible(timeout=10000)

    def test_library_search_filters(self, authenticated_page: Page):
        """Typing in search bar filters papers via HTMX."""
        page = authenticated_page
        search = page.locator("#library-search-bar")
        expect(search).to_be_visible()
        search.fill("attention")
        # Wait for HTMX debounce (300ms) + response
        page.wait_for_timeout(1500)
        cards = page.locator(".card-interactive")
        expect(cards.first).to_be_visible(timeout=5000)

    def test_paper_card_links_to_detail(self, authenticated_page: Page, base_url: str):
        """Clicking a paper title navigates to paper detail."""
        page = authenticated_page
        paper_link = page.locator('a[href*="/web/papers/"]').first
        expect(paper_link).to_be_visible(timeout=10000)
        paper_link.click()
        page.wait_for_url("**/web/papers/**", timeout=10000)

    def test_no_console_errors(self, authenticated_page: Page):
        """Library page has no JavaScript console errors."""
        errors = []
        page = authenticated_page
        page.on(
            "console",
            lambda msg: errors.append(msg.text)
            if msg.type == "error" and is_real_console_error(msg.text)
            else None,
        )
        page.wait_for_load_state("networkidle")
        assert len(errors) == 0, f"Console errors on library: {errors}"
