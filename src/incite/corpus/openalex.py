"""OpenAlex API client for fetching paper metadata."""

import logging
import time
from typing import Optional

import requests

from incite.models import Paper

logger = logging.getLogger(__name__)


class OpenAlexClient:
    """Client for OpenAlex API.

    OpenAlex provides free access to academic paper metadata with high coverage.
    Rate limits: 10 req/s without email, 100 req/s with polite pool.
    """

    BASE_URL = "https://api.openalex.org"

    def __init__(self, email: Optional[str] = None, delay: float = 0.1):
        """Initialize client.

        Args:
            email: Email for polite pool (higher rate limits)
            delay: Delay between requests in seconds
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

    def _params(self) -> dict:
        """Get request parameters."""
        params = {}
        if self.email:
            params["mailto"] = self.email
        return params

    @staticmethod
    def reconstruct_abstract(inverted_index: dict) -> str:
        """Reconstruct abstract from OpenAlex inverted index format.

        OpenAlex stores abstracts as {word: [positions]} to save space.
        """
        if not inverted_index:
            return ""
        words = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word
        return " ".join(words[pos] for pos in sorted(words.keys()))

    def get_work(self, openalex_id: str) -> Optional[Paper]:
        """Fetch a single work by OpenAlex ID.

        Args:
            openalex_id: OpenAlex work ID (e.g., "W1234567890" or full URL)

        Returns:
            Paper object or None if not found
        """
        self._rate_limit()

        # Extract ID from URL if needed
        if openalex_id.startswith("https://"):
            openalex_id = openalex_id.split("/")[-1]

        url = f"{self.BASE_URL}/works/{openalex_id}"

        try:
            response = requests.get(url, params=self._params())
            response.raise_for_status()
            data = response.json()

            # Reconstruct abstract from inverted index
            abstract = self.reconstruct_abstract(data.get("abstract_inverted_index", {}))

            # Extract authors
            authors = []
            for authorship in data.get("authorships", []):
                author = authorship.get("author", {})
                if author.get("display_name"):
                    authors.append(author["display_name"])

            return Paper(
                id=openalex_id,
                title=data.get("title", ""),
                abstract=abstract,
                authors=authors,
                year=data.get("publication_year"),
                doi=(
                    data.get("doi", "").replace("https://doi.org/", "") if data.get("doi") else None
                ),
            )
        except requests.RequestException as e:
            logger.warning("Error fetching %s: %s", openalex_id, e)
            return None

    def get_works_batch(self, openalex_ids: list[str]) -> list[Paper]:
        """Fetch multiple works by OpenAlex IDs.

        Uses filter API to fetch up to 50 works at once.

        Args:
            openalex_ids: List of OpenAlex work IDs

        Returns:
            List of Paper objects
        """
        papers = []

        # Process in batches of 50 (OpenAlex limit)
        batch_size = 50
        for i in range(0, len(openalex_ids), batch_size):
            batch = openalex_ids[i : i + batch_size]
            self._rate_limit()

            # Clean IDs
            clean_ids = []
            for oid in batch:
                if oid.startswith("https://"):
                    oid = oid.split("/")[-1]
                clean_ids.append(oid)

            # Build filter query
            filter_query = "|".join(clean_ids)
            url = f"{self.BASE_URL}/works"
            params = {
                **self._params(),
                "filter": f"openalex_id:{filter_query}",
                "per-page": batch_size,
            }

            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                for item in data.get("results", []):
                    # Skip papers without title
                    title = item.get("title")
                    if not title:
                        continue

                    abstract = self.reconstruct_abstract(item.get("abstract_inverted_index", {}))

                    authors = []
                    for authorship in item.get("authorships", []):
                        author = authorship.get("author", {})
                        if author.get("display_name"):
                            authors.append(author["display_name"])

                    openalex_id = item.get("id", "").split("/")[-1]

                    papers.append(
                        Paper(
                            id=openalex_id,
                            title=title,
                            abstract=abstract,
                            authors=authors,
                            year=item.get("publication_year"),
                            doi=(
                                item.get("doi", "").replace("https://doi.org/", "")
                                if item.get("doi")
                                else None
                            ),
                        )
                    )
            except requests.RequestException as e:
                logger.warning("Error in batch fetch: %s", e)

        return papers
