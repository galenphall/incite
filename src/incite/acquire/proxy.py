"""Library proxy abstraction and implementations.

Provides a generic interface for routing PDF downloads through
institutional library proxies. Supports EZproxy (prefix and suffix
modes) and VPN (no-op rewriting).
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

from incite.acquire.config import ProxyConfig

logger = logging.getLogger(__name__)


def _is_pdf(data: bytes) -> bool:
    """Validate that data starts with PDF magic bytes."""
    return data[:5] == b"%PDF-"


class LibraryProxy(ABC):
    """Interface for institution-specific library proxy access."""

    def __init__(self, config: ProxyConfig):
        self.config = config

    @abstractmethod
    def rewrite_url(self, url: str) -> str:
        """Transform a publisher URL to route through the proxy."""
        ...

    @abstractmethod
    def ensure_authenticated(self, interactive: bool = True) -> bool:
        """Ensure the browser session is authenticated with the proxy.

        If interactive=True, may open a visible browser for the user to
        complete login (handles Shibboleth, Duo MFA, CAS, etc.).

        Returns:
            True if authenticated, False if authentication failed/was declined.
        """
        ...

    @abstractmethod
    def download_pdf(self, url: str, dest: Path) -> Optional[Path]:
        """Download a PDF from the given URL through the proxy.

        Args:
            url: The publisher PDF URL (will be rewritten through proxy).
            dest: Destination path for the downloaded PDF.

        Returns:
            Path to downloaded PDF, or None if download failed.
        """
        ...

    def close(self):
        """Clean up resources (e.g., browser contexts)."""
        pass


class EZProxyPrefix(LibraryProxy):
    """EZproxy in prefix mode -- the most common setup.

    Prepends the proxy URL to publisher URLs:
        https://www.nature.com/articles/nature12373.pdf
     -> https://proxy.lib.umich.edu/login?url=https://www.nature.com/articles/nature12373.pdf
    """

    def rewrite_url(self, url: str) -> str:
        return self.config.proxy_url + url

    def ensure_authenticated(self, interactive: bool = True) -> bool:
        # Delegate to PlaywrightSession (lazy import)
        session = self._get_session()
        if session is None:
            logger.warning("Playwright not available; cannot authenticate proxy")
            return False
        return session.ensure_authenticated(interactive=interactive)

    def download_pdf(self, url: str, dest: Path) -> Optional[Path]:
        proxied_url = self.rewrite_url(url)

        # Try Playwright session first (has proxy cookies)
        session = self._get_session()
        if session is not None and session.has_saved_session():
            result = session.download_pdf(proxied_url, dest, url_rewriter=self.rewrite_url)
            if result:
                return result

        # Fallback: direct requests (works if on VPN or proxy doesn't need auth)
        return _download_direct(proxied_url, dest)

    def _get_session(self):
        """Lazy import PlaywrightSession to avoid hard dependency."""
        if not hasattr(self, "_session"):
            try:
                from incite.acquire.session import PlaywrightSession

                self._session = PlaywrightSession(self.config)
            except ImportError:
                self._session = None
        return self._session

    def close(self):
        if hasattr(self, "_session") and self._session is not None:
            self._session.close()


class EZProxySuffix(LibraryProxy):
    """EZproxy in suffix/hostname rewriting mode.

    Rewrites the hostname by replacing dots with dashes and appending
    the proxy suffix:
        https://www.nature.com/articles/nature12373.pdf
     -> https://www-nature-com.proxy.lib.umich.edu/articles/nature12373.pdf
    """

    def rewrite_url(self, url: str) -> str:
        parsed = urlparse(url)
        # Replace dots in hostname with dashes, append proxy suffix
        proxied_host = parsed.hostname.replace(".", "-") + self.config.proxy_suffix
        return f"{parsed.scheme}://{proxied_host}{parsed.path}"

    def ensure_authenticated(self, interactive: bool = True) -> bool:
        session = self._get_session()
        if session is None:
            return False
        return session.ensure_authenticated(interactive=interactive)

    def download_pdf(self, url: str, dest: Path) -> Optional[Path]:
        proxied_url = self.rewrite_url(url)
        session = self._get_session()
        if session is not None and session.has_saved_session():
            result = session.download_pdf(proxied_url, dest, url_rewriter=self.rewrite_url)
            if result:
                return result
        return _download_direct(proxied_url, dest)

    def _get_session(self):
        if not hasattr(self, "_session"):
            try:
                from incite.acquire.session import PlaywrightSession

                self._session = PlaywrightSession(self.config)
            except ImportError:
                self._session = None
        return self._session

    def close(self):
        if hasattr(self, "_session") and self._session is not None:
            self._session.close()


class VPNProxy(LibraryProxy):
    """VPN mode -- no URL rewriting needed.

    The user is on their university VPN, so publisher URLs work directly.
    Auth check verifies that a known paywalled URL is accessible.
    """

    def rewrite_url(self, url: str) -> str:
        return url  # No-op

    def ensure_authenticated(self, interactive: bool = True) -> bool:
        """Check VPN connectivity by trying to access the test DOI."""
        test_url = f"https://doi.org/{self.config.test_doi}"
        try:
            resp = requests.head(test_url, allow_redirects=True, timeout=10)
            # If we can reach the publisher and aren't redirected to a paywall
            # interstitial, the VPN is working
            return resp.ok
        except requests.RequestException:
            return False

    def download_pdf(self, url: str, dest: Path) -> Optional[Path]:
        return _download_direct(url, dest)


def _download_direct(url: str, dest: Path) -> Optional[Path]:
    """Download a PDF via direct HTTP request.

    Args:
        url: URL to download from.
        dest: Destination file path.

    Returns:
        Path to downloaded file, or None if download failed or
        response wasn't a valid PDF.
    """
    try:
        resp = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": "inCite/0.1 (mailto:research@umich.edu)"},
            stream=True,
        )
        resp.raise_for_status()

        # Read full content
        content = resp.content

        # Validate PDF magic bytes
        if not _is_pdf(content):
            logger.warning(f"Response from {url} is not a valid PDF")
            return None

        # Write to destination
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        return dest

    except requests.RequestException as e:
        logger.warning(f"Download failed for {url}: {e}")
        return None


def create_proxy(config: ProxyConfig) -> LibraryProxy:
    """Factory: create a LibraryProxy from configuration.

    Args:
        config: Proxy configuration.

    Returns:
        Appropriate LibraryProxy implementation.

    Raises:
        ValueError: If proxy_type is not recognized.
    """
    if config.proxy_type == "ezproxy_prefix":
        return EZProxyPrefix(config)
    elif config.proxy_type == "ezproxy_suffix":
        return EZProxySuffix(config)
    elif config.proxy_type == "vpn":
        return VPNProxy(config)
    else:
        raise ValueError(f"Unknown proxy type: {config.proxy_type}")
