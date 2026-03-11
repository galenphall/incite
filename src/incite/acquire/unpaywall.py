"""Unpaywall API client for open-access PDF URL lookup."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.unpaywall.org/v2"


@dataclass(frozen=True)
class UnpaywallResult:
    """Result from an Unpaywall DOI lookup."""

    is_oa: bool
    best_oa_url: str | None


class UnpaywallClient:
    """Client for the Unpaywall REST API (polite pool via email)."""

    def __init__(self, email: str, timeout: int = 10) -> None:
        self.email = email
        self.timeout = timeout

    def lookup(self, doi: str) -> UnpaywallResult | None:
        """Look up open-access info for a DOI.

        Returns None on 404, timeout, or network errors.
        """
        url = f"{BASE_URL}/{doi}"
        try:
            resp = requests.get(url, params={"email": self.email}, timeout=self.timeout)
        except requests.Timeout:
            logger.warning("Unpaywall timeout for DOI %s", doi)
            return None
        except requests.RequestException:
            logger.warning("Unpaywall request failed for DOI %s", doi, exc_info=True)
            return None

        if resp.status_code == 404:
            return None
        resp.raise_for_status()

        data = resp.json()
        best_loc = data.get("best_oa_location")
        best_oa_url: str | None = None
        if best_loc is not None:
            best_oa_url = best_loc.get("url_for_pdf") or best_loc.get("url")

        return UnpaywallResult(is_oa=data.get("is_oa", False), best_oa_url=best_oa_url)
