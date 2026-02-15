"""Playwright browser session management for proxy authentication.

Handles the login flow for university proxy services (Shibboleth, CAS,
Duo MFA, etc.) by opening a visible browser for the user to complete
authentication, then saving the session for headless reuse.

Uses sync Playwright (the entire codebase is sync, no asyncio).
"""

import logging
import re
import time
from pathlib import Path
from typing import Callable, Optional

from incite.acquire.config import ProxyConfig

logger = logging.getLogger(__name__)

# Domains / URL fragments that indicate we're still on an auth page.
# These match the IdP / SSO / MFA intermediaries, NOT the proxy itself.
_AUTH_DOMAINS = [
    "weblogin.",
    "shibboleth.",
    "idp.",
    "duosecurity.com",
    "duo.com",
    "cas/login",
    "saml",
    "simplesaml",
    "adfs.",
    "login.microsoftonline.com",
    "accounts.google.com",
]

# Domains that are part of the proxy or auth infrastructure, not the target
_PROXY_DOMAINS = [
    "proxy.lib.",
    "ezproxy.",
    "libproxy.",
]


def _extract_citation_pdf_url(html: str) -> Optional[str]:
    """Extract citation_pdf_url from HTML meta tags."""
    match = re.search(
        r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
        html,
        re.IGNORECASE,
    )
    if match:
        return match.group(1)
    match = re.search(
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url["\']',
        html,
        re.IGNORECASE,
    )
    if match:
        return match.group(1)
    return None


def _check_playwright_available() -> bool:
    """Check if Playwright is installed and usable."""
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401

        return True
    except ImportError:
        return False


def _is_still_authenticating(url: str) -> bool:
    """Check if the browser is still on an auth/login page.

    Returns True when the URL is on an IdP (Shibboleth, Duo, CAS)
    or on the proxy's own login path. Returns False once the browser
    has reached any content page — including proxied content like
    ``www-nature-com.proxy.lib.umich.edu``.
    """
    from urllib.parse import urlparse

    url_lower = url.lower()

    # On an IdP / SSO / MFA page?
    if any(indicator in url_lower for indicator in _AUTH_DOMAINS):
        return True

    # On the proxy's own login page (not proxied content)?
    # e.g. proxy.lib.umich.edu/login?url=... — host is exactly the
    # proxy domain AND path starts with /login
    parsed = urlparse(url_lower)
    host = parsed.hostname or ""
    path = parsed.path or ""

    for proxy_domain in _PROXY_DOMAINS:
        if proxy_domain in host and "/login" in path:
            return True

    # about:blank or empty
    if not host or host == "about":
        return True

    return False


class PlaywrightSession:
    """Manages Playwright browser sessions for proxy authentication.

    Session state (cookies, localStorage) is persisted to disk so that
    subsequent headless requests can reuse the authenticated session.

    Storage layout:
        ~/.incite/proxy_session/
            storage_state.json   # Playwright browser context state
    """

    def __init__(self, config: ProxyConfig):
        self.config = config
        self.session_dir = config.session_dir
        self._storage_state_path = self.session_dir / "storage_state.json"
        self._playwright = None
        self._browser = None

    def has_saved_session(self) -> bool:
        """Check if a saved session exists on disk."""
        return self._storage_state_path.exists()

    def ensure_authenticated(self, interactive: bool = True) -> bool:
        """Ensure we have a valid authenticated session.

        1. Try saved session first (headless, navigate to test URL,
           check if we land on a login page).
        2. If expired/missing and interactive=True: open visible Chromium
           for the user to complete login, then save the session.

        Args:
            interactive: If True, open a visible browser for login when needed.

        Returns:
            True if authenticated, False if auth failed or was declined.
        """
        if not _check_playwright_available():
            logger.error(
                "Playwright is not installed. "
                "Install with: pip install 'incite[acquire]' && playwright install chromium"
            )
            return False

        # Try saved session first
        if self.has_saved_session():
            if self._test_saved_session():
                logger.info("Saved proxy session is still valid")
                return True
            logger.info("Saved proxy session has expired")

        # Need fresh authentication
        if not interactive:
            return False

        return self._interactive_login()

    def _test_saved_session(self) -> bool:
        """Test if the saved session is still valid by navigating to the test URL."""
        from playwright.sync_api import sync_playwright

        test_url = self.config.test_url

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(storage_state=str(self._storage_state_path))
                page = context.new_page()

                page.goto(test_url, wait_until="domcontentloaded", timeout=15000)

                # If we're still on auth/proxy pages, session has expired
                is_valid = not _is_still_authenticating(page.url)

                context.close()
                browser.close()
                return is_valid
        except Exception as e:
            logger.warning(f"Session test failed: {e}")
            return False

    def _interactive_login(self) -> bool:
        """Open a visible browser for the user to complete proxy login.

        Uses Playwright's navigation event listener to detect when the
        browser leaves auth pages (Shibboleth, Duo, CAS, etc.) and
        reaches content. Times out after 5 minutes.

        Returns:
            True if login succeeded, False otherwise.
        """
        from playwright.sync_api import sync_playwright

        test_url = self.config.test_url

        print("\nOpening browser for proxy authentication...", flush=True)
        print(
            f"Please log in to: {self.config.institution_name or 'your library proxy'}",
            flush=True,
        )
        print(
            "The browser will close automatically when login is complete.\n",
            flush=True,
        )

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=False)

                # Load saved state if it exists (may have partial cookies)
                context_opts = {}
                if self.has_saved_session():
                    context_opts["storage_state"] = str(self._storage_state_path)

                context = browser.new_context(**context_opts)
                page = context.new_page()

                # Track auth completion via navigation events.
                # page.url polling misses SAML POST redirects; event
                # listeners catch every frame navigation reliably.
                auth_done_url: list[str] = []  # mutable container for closure

                def _on_navigated(frame):
                    if frame != page.main_frame:
                        return
                    url = frame.url
                    if not _is_still_authenticating(url):
                        auth_done_url.append(url)

                page.on("framenavigated", _on_navigated)

                page.goto(test_url, wait_until="domcontentloaded", timeout=30000)

                # Wait for auth to complete (5 minute timeout)
                timeout_secs = 300
                start = time.time()

                while time.time() - start < timeout_secs:
                    if auth_done_url:
                        # Auth complete -- save session
                        self.session_dir.mkdir(parents=True, exist_ok=True)
                        context.storage_state(path=str(self._storage_state_path))

                        print(
                            "\nAuthentication successful! Session saved.",
                            flush=True,
                        )
                        context.close()
                        browser.close()
                        return True

                    # Pump Playwright's event loop briefly
                    page.wait_for_timeout(1000)

                # Timeout
                print(
                    "Authentication timed out (5 minutes). Please try again.",
                    flush=True,
                )
                context.close()
                browser.close()
                return False

        except Exception as e:
            logger.error(f"Interactive login failed: {e}")
            return False

    def download_pdf(
        self,
        url: str,
        dest: Path,
        url_rewriter: Optional[Callable[[str], str]] = None,
    ) -> Optional[Path]:
        """Download a PDF using the saved proxy session.

        Three-tier cascade using a single HTTP request for tiers 1+2:

        1. Direct PDF: Response body starts with PDF magic bytes -> save it.
        2. Meta tag: Response is HTML -> extract ``citation_pdf_url`` meta
           tag -> download the PDF URL (rewritten through proxy if needed).
        3. Page fallback: Full browser navigation for JS-rendered pages.

        Args:
            url: The proxied URL to download from.
            dest: Destination file path for the PDF.
            url_rewriter: Optional callback to rewrite raw publisher URLs
                through the proxy (e.g., ``EZProxyPrefix.rewrite_url``).

        Returns:
            Path to downloaded PDF, or None if download failed.
        """
        if not _check_playwright_available():
            return None

        if not self.has_saved_session():
            logger.warning("No saved session; cannot download via proxy")
            return None

        from playwright.sync_api import sync_playwright

        from incite.acquire.proxy import _is_pdf

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    storage_state=str(self._storage_state_path),
                    accept_downloads=True,
                )

                # Single API request — used for tiers 1 and 2
                try:
                    resp = context.request.get(url, max_redirects=10, timeout=30000)
                except Exception as e:
                    logger.debug(f"API request failed for {url}: {e}")
                    resp = None

                if resp and resp.ok:
                    body = resp.body()

                    # Tier 1: Response is already a PDF
                    if _is_pdf(body):
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(body)
                        context.close()
                        browser.close()
                        return dest

                    # Tier 2: HTML with citation_pdf_url meta tag
                    html = body.decode("utf-8", errors="replace")
                    pdf_url = _extract_citation_pdf_url(html)
                    if pdf_url:
                        if url_rewriter:
                            pdf_url = url_rewriter(pdf_url)
                        result = self._api_download(context, pdf_url, dest)
                        if result:
                            context.close()
                            browser.close()
                            return result

                # Tier 3: Full page navigation fallback (JS-rendered pages)
                page = context.new_page()
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                except Exception:
                    pass

                pdf_url = self._find_pdf_link(page)
                if pdf_url:
                    if url_rewriter:
                        pdf_url = url_rewriter(pdf_url)
                    result = self._api_download(context, pdf_url, dest)
                    if result:
                        page.close()
                        context.close()
                        browser.close()
                        return result

                page.close()
                context.close()
                browser.close()
                return None

        except Exception as e:
            logger.warning(f"Playwright download failed for {url}: {e}")
            return None

    def _api_download(self, context, url: str, dest: Path) -> Optional[Path]:
        """Download a PDF using Playwright's API request context.

        Uses the browser context's cookies but bypasses rendering,
        avoiding issues with headless Chromium discarding inline PDFs.
        """
        from incite.acquire.proxy import _is_pdf

        try:
            resp = context.request.get(url, max_redirects=10, timeout=30000)
            if resp.ok:
                body = resp.body()
                if _is_pdf(body):
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(body)
                    return dest
        except Exception as e:
            logger.debug(f"API download failed for {url}: {e}")

        return None

    def _find_pdf_link(self, page) -> Optional[str]:
        """Search the current page for a PDF download link.

        Checks, in order:
        1. Meta tag: <meta name="citation_pdf_url" content="...">
        2. Link with href containing ".pdf"
        3. "Download PDF" buttons/links
        """
        # 1. Citation meta tag (most reliable)
        meta = page.query_selector('meta[name="citation_pdf_url"]')
        if meta:
            pdf_url = meta.get_attribute("content")
            if pdf_url:
                return pdf_url

        # 2. Direct PDF links
        pdf_link = page.query_selector('a[href*=".pdf"]')
        if pdf_link:
            href = pdf_link.get_attribute("href")
            if href:
                return href

        # 3. Download PDF buttons (common on publisher sites)
        for selector in [
            'a:has-text("Download PDF")',
            'a:has-text("Download paper")',
            'a:has-text("Full Text PDF")',
            'button:has-text("Download PDF")',
        ]:
            try:
                el = page.query_selector(selector)
                if el:
                    href = el.get_attribute("href")
                    if href:
                        return href
            except Exception:
                continue

        return None

    def close(self):
        """Clean up browser resources."""
        # Resources are cleaned up within context managers in each method
        pass
