"""Unpaywall API client for finding open-access PDFs.

Unpaywall is free and covers ~30-40% of papers. Even for closed-access
papers, it often returns the publisher PDF URL, which can be routed
through a library proxy.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class UnpaywallResult:
    """Result from an Unpaywall DOI lookup."""

    doi: str
    is_oa: bool
    best_oa_url: Optional[str]  # Direct OA PDF link (green/gold OA)
    publisher_pdf_url: Optional[str]  # Publisher's PDF URL (may be paywalled)
    oa_status: str  # "gold", "green", "hybrid", "bronze", "closed"


class UnpaywallClient:
    """Unpaywall API client. Free, rate-limited to 100K/day per email.

    Mirrors the SemanticScholarClient pattern: rate limiting, Optional returns,
    RequestException handling.
    """

    BASE_URL = "https://api.unpaywall.org/v2"

    def __init__(self, email: str, delay: float = 0.1):
        """Initialize client.

        Args:
            email: Required by Unpaywall TOS (identifies your app).
                   Uses OPENALEX_EMAIL env var convention.
            delay: Minimum delay between requests in seconds.
        """
        self.email = email
        self.delay = delay
        self._last_request = 0.0

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.time()

    def lookup(self, doi: str) -> Optional[UnpaywallResult]:
        """Look up a DOI on Unpaywall.

        Args:
            doi: The DOI to look up (e.g., "10.1038/nature12373")

        Returns:
            UnpaywallResult with OA status and PDF URLs, or None if not found.
        """
        self._rate_limit()

        url = f"{self.BASE_URL}/{doi}"
        params = {"email": self.email}

        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 404:
                return None

            response.raise_for_status()
            data = response.json()

            # Extract best OA URL
            best_oa_url = None
            best_oa_location = data.get("best_oa_location")
            if best_oa_location:
                best_oa_url = best_oa_location.get("url_for_pdf") or best_oa_location.get("url")

            # Extract publisher PDF URL (useful even for closed papers)
            publisher_pdf_url = None
            for location in data.get("oa_locations", []):
                if location.get("host_type") == "publisher":
                    publisher_pdf_url = location.get("url_for_pdf") or location.get("url")
                    break

            # If no publisher URL from OA locations, check first_oa_location
            if not publisher_pdf_url:
                first_loc = data.get("first_oa_location")
                if first_loc and first_loc.get("host_type") == "publisher":
                    publisher_pdf_url = first_loc.get("url_for_pdf") or first_loc.get("url")

            return UnpaywallResult(
                doi=doi,
                is_oa=data.get("is_oa", False),
                best_oa_url=best_oa_url,
                publisher_pdf_url=publisher_pdf_url,
                oa_status=data.get("oa_status", "closed"),
            )
        except requests.RequestException as e:
            logger.warning(f"Unpaywall lookup failed for {doi}: {e}")
            return None

    def batch_lookup(self, dois: list[str]) -> dict[str, UnpaywallResult]:
        """Look up multiple DOIs sequentially (Unpaywall has no batch endpoint).

        Args:
            dois: List of DOIs to look up.

        Returns:
            Dict mapping DOI to UnpaywallResult (only successful lookups).
        """
        results: dict[str, UnpaywallResult] = {}

        for doi in tqdm(dois, desc="Unpaywall lookup", unit="doi"):
            result = self.lookup(doi)
            if result:
                results[doi] = result

        return results
