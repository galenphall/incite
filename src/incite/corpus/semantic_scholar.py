"""Semantic Scholar API client for fetching papers."""

import logging
import time
from typing import Optional

import requests

from incite.models import Paper

logger = logging.getLogger(__name__)


class SemanticScholarClient:
    """Client for Semantic Scholar API.

    Rate limits: 100 requests/5 min without API key, 1 request/sec with key.
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: Optional[str] = None, delay: float = 1.0):
        """Initialize client.

        Args:
            api_key: Optional API key for higher rate limits
            delay: Delay between requests in seconds
        """
        self.api_key = api_key
        self.delay = delay
        self._last_request = 0.0

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.time()

    def _headers(self) -> dict:
        """Get request headers."""
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Fetch a single paper by ID.

        Args:
            paper_id: Semantic Scholar paper ID, DOI, or arXiv ID

        Returns:
            Paper object or None if not found
        """
        self._rate_limit()

        fields = "paperId,title,abstract,authors,year,externalIds"
        url = f"{self.BASE_URL}/paper/{paper_id}?fields={fields}"

        try:
            response = requests.get(url, headers=self._headers())
            response.raise_for_status()
            data = response.json()

            return Paper(
                id=data["paperId"],
                title=data.get("title", ""),
                abstract=data.get("abstract", ""),
                authors=[a.get("name", "") for a in data.get("authors", [])],
                year=data.get("year"),
                doi=data.get("externalIds", {}).get("DOI"),
            )
        except requests.RequestException:
            return None

    def search_papers(
        self,
        query: str,
        limit: int = 100,
        fields_of_study: Optional[list[str]] = None,
        year_range: Optional[tuple[int, int]] = None,
    ) -> list[Paper]:
        """Search for papers.

        Args:
            query: Search query
            limit: Maximum number of results (max 100 per request)
            fields_of_study: Filter by field (e.g., ["Computer Science"])
            year_range: Filter by year range (start, end)

        Returns:
            List of Paper objects
        """
        papers = []
        offset = 0
        batch_size = min(100, limit)

        while len(papers) < limit:
            self._rate_limit()

            fields = "paperId,title,abstract,authors,year,externalIds"
            url = (
                f"{self.BASE_URL}/paper/search?query={query}&fields={fields}"
                f"&offset={offset}&limit={batch_size}"
            )

            if fields_of_study:
                url += f"&fieldsOfStudy={','.join(fields_of_study)}"
            if year_range:
                url += f"&year={year_range[0]}-{year_range[1]}"

            try:
                response = requests.get(url, headers=self._headers())
                response.raise_for_status()
                data = response.json()

                for item in data.get("data", []):
                    if item.get("title") and item.get("abstract"):
                        papers.append(
                            Paper(
                                id=item["paperId"],
                                title=item["title"],
                                abstract=item.get("abstract", ""),
                                authors=[a.get("name", "") for a in item.get("authors", [])],
                                year=item.get("year"),
                                doi=item.get("externalIds", {}).get("DOI"),
                            )
                        )

                if len(data.get("data", [])) < batch_size:
                    break  # No more results

                offset += batch_size

            except requests.RequestException as e:
                logger.warning("Error fetching papers: %s", e)
                break

        return papers[:limit]

    def get_paper_references(self, paper_id: str, limit: int = 100) -> list[Paper]:
        """Get papers cited by a paper.

        Args:
            paper_id: Semantic Scholar paper ID
            limit: Maximum number of references

        Returns:
            List of referenced Paper objects
        """
        self._rate_limit()

        fields = "paperId,title,abstract,authors,year,externalIds"
        url = f"{self.BASE_URL}/paper/{paper_id}/references?fields={fields}&limit={limit}"

        papers = []
        try:
            response = requests.get(url, headers=self._headers())
            response.raise_for_status()
            data = response.json()

            for item in data.get("data", []):
                cited = item.get("citedPaper", {})
                if cited.get("paperId") and cited.get("title"):
                    papers.append(
                        Paper(
                            id=cited["paperId"],
                            title=cited["title"],
                            abstract=cited.get("abstract", ""),
                            authors=[a.get("name", "") for a in cited.get("authors", [])],
                            year=cited.get("year"),
                            doi=cited.get("externalIds", {}).get("DOI"),
                        )
                    )
        except requests.RequestException as e:
            logger.warning("Error fetching references: %s", e)

        return papers

    def get_paper_citations(self, paper_id: str, limit: int = 100) -> list[Paper]:
        """Get papers that cite a paper.

        Args:
            paper_id: Semantic Scholar paper ID
            limit: Maximum number of citations

        Returns:
            List of citing Paper objects
        """
        self._rate_limit()

        fields = "paperId,title,abstract,authors,year,externalIds"
        url = f"{self.BASE_URL}/paper/{paper_id}/citations?fields={fields}&limit={limit}"

        papers = []
        try:
            response = requests.get(url, headers=self._headers())
            response.raise_for_status()
            data = response.json()

            for item in data.get("data", []):
                citing = item.get("citingPaper", {})
                if citing.get("paperId") and citing.get("title"):
                    papers.append(
                        Paper(
                            id=citing["paperId"],
                            title=citing["title"],
                            abstract=citing.get("abstract", ""),
                            authors=[a.get("name", "") for a in citing.get("authors", [])],
                            year=citing.get("year"),
                            doi=citing.get("externalIds", {}).get("DOI"),
                        )
                    )
        except requests.RequestException as e:
            logger.warning("Error fetching citations: %s", e)

        return papers
