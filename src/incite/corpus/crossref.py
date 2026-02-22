"""CrossRef API client for resolving DOIs from paper titles.

Uses the CrossRef works endpoint to search for papers by bibliographic
metadata (title, author, year) and validates matches using fuzzy string
matching before returning a DOI.
"""

from __future__ import annotations

import logging
import os
import re
import time
import unicodedata

import requests

logger = logging.getLogger(__name__)

CROSSREF_API_URL = "https://api.crossref.org/works"
REQUEST_DELAY = 0.1  # seconds between requests (polite pool allows 50 req/s)


def normalize_title(title: str) -> str:
    """Normalize a title for fuzzy comparison.

    Applies NFKD unicode normalization, strips combining marks, removes
    LaTeX markup, lowercases, removes punctuation (except hyphens), and
    collapses whitespace.
    """
    if not title:
        return ""
    # NFKD normalization and strip combining marks
    text = unicodedata.normalize("NFKD", title)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Strip LaTeX commands like \textit{...}, \emph{...}, \textbf{...}
    text = re.sub(r"\\(?:textit|emph|textbf|textrm|mathrm)\{([^}]*)\}", r"\1", text)
    # Remove remaining braces (used for capitalization in BibTeX)
    text = text.replace("{", "").replace("}", "")
    # Remove backslash-escaped characters
    text = re.sub(r"\\.", "", text)
    # Lowercase
    text = text.lower()
    # Remove punctuation except hyphens
    text = re.sub(r"[^\w\s-]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_author_lastnames(authors: list[dict]) -> list[str]:
    """Extract last names from CrossRef author records."""
    lastnames = []
    for author in authors:
        family = author.get("family", "")
        if family:
            lastnames.append(family)
    return lastnames


def search_by_bibliographic(
    title: str,
    first_author_last: str | None = None,
    year: int | None = None,
) -> dict | None:
    """Search CrossRef for a paper by bibliographic metadata.

    Uses the ``query.bibliographic`` parameter to search title, author,
    and year simultaneously. The top result is validated using fuzzy title
    matching before returning.

    Args:
        title: Paper title (required)
        first_author_last: First author's last name for validation
        year: Publication year for validation

    Returns:
        Dict with ``doi``, ``title``, ``authors`` (list of last names),
        and ``year`` if a valid match is found, None otherwise.
    """
    try:
        from rapidfuzz.fuzz import token_sort_ratio
    except ImportError:
        logger.debug("rapidfuzz not installed, skipping CrossRef lookup")
        return None

    if not title or not title.strip():
        return None

    # Build query string from bibliographic fields
    query_parts = [title]
    if first_author_last:
        query_parts.append(first_author_last)
    if year:
        query_parts.append(str(year))
    query_str = " ".join(query_parts)

    mailto = os.environ.get("OPENALEX_EMAIL", "")
    params: dict = {
        "query.bibliographic": query_str,
        "rows": 1,
        "select": "DOI,title,author,published-print,published-online",
    }
    if mailto:
        params["mailto"] = mailto

    try:
        resp = requests.get(
            CROSSREF_API_URL,
            params=params,
            timeout=15,
            headers={"User-Agent": f"incite/0.1 (mailto:{mailto})"},
        )
        resp.raise_for_status()
    except requests.RequestException:
        logger.debug("CrossRef request failed for '%s'", title[:80], exc_info=True)
        return None

    data = resp.json()
    items = data.get("message", {}).get("items", [])
    if not items:
        return None

    item = items[0]
    candidate_titles = item.get("title", [])
    if not candidate_titles:
        return None

    candidate_title = candidate_titles[0]
    candidate_doi = item.get("DOI")
    if not candidate_doi:
        return None

    # Extract year from published-print or published-online
    candidate_year = None
    for date_field in ("published-print", "published-online"):
        date_parts = item.get(date_field, {}).get("date-parts", [[]])
        if date_parts and date_parts[0] and date_parts[0][0]:
            candidate_year = date_parts[0][0]
            break

    candidate_authors = _extract_author_lastnames(item.get("author", []))

    # Validate match using fuzzy title comparison
    norm_query = normalize_title(title)
    norm_candidate = normalize_title(candidate_title)
    score = token_sort_ratio(norm_query, norm_candidate)

    if score >= 90:
        # High-confidence match
        pass
    elif score >= 80:
        # Medium confidence — require year and author confirmation
        year_ok = (
            year is not None and candidate_year is not None and abs(year - candidate_year) <= 1
        )
        author_ok = False
        if first_author_last and candidate_authors:
            author_score = token_sort_ratio(first_author_last.lower(), candidate_authors[0].lower())
            author_ok = author_score >= 80
        if not (year_ok and author_ok):
            return None
    else:
        # Low confidence — reject
        return None

    return {
        "doi": candidate_doi,
        "title": candidate_title,
        "authors": candidate_authors,
        "year": candidate_year,
    }


def resolve_dois_batch(
    papers: list,
    max_papers: int = 200,
) -> int:
    """Resolve DOIs for papers missing them via CrossRef title search.

    Modifies papers in-place, setting ``paper.doi`` when a match is found.

    Args:
        papers: List of Paper objects. Only papers with a title and no DOI
            are processed.
        max_papers: Maximum number of papers to look up (default 200).

    Returns:
        Number of papers that received a DOI.
    """
    try:
        import rapidfuzz  # noqa: F401
    except ImportError:
        logger.info("rapidfuzz not installed, skipping CrossRef DOI resolution")
        return 0

    candidates = [p for p in papers if not p.doi and p.title and p.title.strip()]
    if not candidates:
        return 0

    candidates = candidates[:max_papers]
    resolved = 0

    for paper in candidates:
        first_author_last = None
        if paper.authors:
            # Extract last name from first author (format: "First Last")
            parts = paper.authors[0].rsplit(" ", 1)
            first_author_last = parts[-1] if parts else None

        result = search_by_bibliographic(
            title=paper.title,
            first_author_last=first_author_last,
            year=paper.year,
        )
        if result:
            paper.doi = result["doi"]
            resolved += 1
            logger.debug("Resolved DOI for '%s': %s", paper.title[:60], result["doi"])

        time.sleep(REQUEST_DELAY)

    logger.info("Resolved %d/%d DOIs via CrossRef title search", resolved, len(candidates))
    return resolved
