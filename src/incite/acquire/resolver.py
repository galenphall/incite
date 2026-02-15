"""DOI resolution utilities.

Resolves DOIs to publisher landing page URLs. PDF discovery from landing
pages is handled by the session layer (citation_pdf_url meta tag extraction
and page navigation fallback).
"""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def resolve_doi(doi: str) -> Optional[str]:
    """Resolve a DOI to its landing page URL.

    Follows redirects from https://doi.org/{doi} and returns the final URL.
    Does NOT download the page -- just follows the redirect chain.

    Args:
        doi: The DOI to resolve (e.g., "10.1038/nature12373")

    Returns:
        Final landing page URL, or None if resolution fails.
    """
    try:
        resp = requests.head(
            f"https://doi.org/{doi}",
            allow_redirects=True,
            timeout=20,
            headers={"User-Agent": "inCite/0.1 (mailto:research@umich.edu)"},
        )
        # Return the final URL even on non-200 status (e.g., 403).
        # Many publishers block HEAD requests but the redirect chain
        # still reveals the landing page URL, which is all we need.
        if resp.url and resp.url != f"https://doi.org/{doi}":
            return resp.url
        if resp.ok:
            return resp.url
        return None
    except requests.RequestException as e:
        logger.warning(f"DOI resolution failed for {doi}: {e}")
        return None


def doi_slug(doi: str) -> str:
    """Convert a DOI to a filesystem-safe filename slug.

    Args:
        doi: The DOI (e.g., "10.1038/nature12373")

    Returns:
        Filesystem-safe string (e.g., "10.1038_nature12373")
    """
    return doi.replace("/", "_")
