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

        fields = "paperId,title,abstract,authors,year,externalIds,openAccessPdf,venue"
        url = f"{self.BASE_URL}/paper/{paper_id}/references?fields={fields}&limit={limit}"

        papers = []
        try:
            response = requests.get(url, headers=self._headers())
            response.raise_for_status()
            data = response.json()
            if not data:
                return papers

            for item in data.get("data") or []:
                cited = item.get("citedPaper", {})
                if cited.get("paperId") and cited.get("title"):
                    oa_pdf = cited.get("openAccessPdf") or {}
                    papers.append(
                        Paper(
                            id=cited["paperId"],
                            title=cited["title"],
                            abstract=cited.get("abstract", ""),
                            authors=[a.get("name", "") for a in cited.get("authors", [])],
                            year=cited.get("year"),
                            doi=cited.get("externalIds", {}).get("DOI"),
                            pdf_url=oa_pdf.get("url"),
                            journal=cited.get("venue") or None,
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

        fields = "paperId,title,abstract,authors,year,externalIds,openAccessPdf,venue"
        url = f"{self.BASE_URL}/paper/{paper_id}/citations?fields={fields}&limit={limit}"

        papers = []
        try:
            response = requests.get(url, headers=self._headers())
            response.raise_for_status()
            data = response.json()
            if not data:
                return papers

            for item in data.get("data") or []:
                citing = item.get("citingPaper", {})
                if citing.get("paperId") and citing.get("title"):
                    oa_pdf = citing.get("openAccessPdf") or {}
                    papers.append(
                        Paper(
                            id=citing["paperId"],
                            title=citing["title"],
                            abstract=citing.get("abstract", ""),
                            authors=[a.get("name", "") for a in citing.get("authors", [])],
                            year=citing.get("year"),
                            doi=citing.get("externalIds", {}).get("DOI"),
                            pdf_url=oa_pdf.get("url"),
                            journal=citing.get("venue") or None,
                        )
                    )
        except requests.RequestException as e:
            logger.warning("Error fetching citations: %s", e)

        return papers

    def get_recommendations(self, paper_id: str, limit: int = 100) -> list[Paper]:
        """Get semantically similar papers via the S2 Recommendations API.

        Args:
            paper_id: Semantic Scholar paper ID
            limit: Maximum number of recommendations

        Returns:
            List of recommended Paper objects
        """
        self._rate_limit()

        url = f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper_id}"
        fields = "paperId,title,authors,year,externalIds,abstract,citationCount,openAccessPdf,venue"

        papers = []
        try:
            response = requests.get(
                url,
                params={"fields": fields, "limit": limit},
                headers=self._headers(),
            )
            response.raise_for_status()
            data = response.json()

            for item in data.get("recommendedPapers") or []:
                pid = item.get("paperId")
                title = item.get("title")
                if not pid or not title:
                    continue
                oa_pdf = item.get("openAccessPdf") or {}
                papers.append(
                    Paper(
                        id=pid,
                        title=title,
                        abstract=item.get("abstract", "") or "",
                        authors=[a.get("name", "") for a in item.get("authors", [])],
                        year=item.get("year"),
                        doi=item.get("externalIds", {}).get("DOI"),
                        pdf_url=oa_pdf.get("url"),
                        journal=item.get("venue") or None,
                    )
                )
        except requests.RequestException as e:
            logger.warning("Error fetching S2 recommendations for %s: %s", paper_id, e)

        return papers[:limit]

    def get_recommendations_batch(
        self,
        positive_ids: list[str],
        negative_ids: list[str] | None = None,
        limit: int = 500,
    ) -> list[Paper]:
        """Multi-paper recommendations via the S2 Recommendations API.

        Uses the POST endpoint which considers all positive papers together
        for better relevance than per-paper calls.

        Args:
            positive_ids: S2 paper IDs to recommend from (max 100)
            negative_ids: Optional S2 paper IDs to avoid recommending
            limit: Maximum number of recommendations

        Returns:
            List of recommended Paper objects, ranked by relevance
        """
        self._rate_limit()

        url = "https://api.semanticscholar.org/recommendations/v1/papers/"
        fields = "paperId,title,authors,year,externalIds,abstract,citationCount,openAccessPdf,venue"

        body: dict = {"positivePaperIds": positive_ids[:100]}
        if negative_ids:
            body["negativePaperIds"] = negative_ids[:100]

        papers = []
        try:
            response = requests.post(
                url,
                params={"fields": fields, "limit": limit},
                headers=self._headers(),
                json=body,
            )
            response.raise_for_status()
            data = response.json()

            for item in data.get("recommendedPapers") or []:
                pid = item.get("paperId")
                title = item.get("title")
                if not pid or not title:
                    continue
                oa_pdf = item.get("openAccessPdf") or {}
                papers.append(
                    Paper(
                        id=pid,
                        title=title,
                        abstract=item.get("abstract", "") or "",
                        authors=[a.get("name", "") for a in item.get("authors", [])],
                        year=item.get("year"),
                        doi=item.get("externalIds", {}).get("DOI"),
                        pdf_url=oa_pdf.get("url"),
                        journal=item.get("venue") or None,
                    )
                )
        except requests.RequestException as e:
            logger.warning("Error fetching S2 batch recommendations: %s", e)

        return papers[:limit]

    def get_papers_batch(self, paper_ids: list[str], batch_size: int = 500) -> dict[str, Paper]:
        """Fetch multiple papers by ID using the batch endpoint.

        Uses POST /paper/batch for efficient bulk lookups (up to 500 per request).

        Args:
            paper_ids: List of paper IDs (e.g., "DOI:10.1234/xxx", S2 IDs, etc.)
            batch_size: Max IDs per request (S2 limit is 500)

        Returns:
            Dict mapping input ID to Paper object (only successful lookups included)
        """
        results: dict[str, Paper] = {}

        for i in range(0, len(paper_ids), batch_size):
            batch = paper_ids[i : i + batch_size]
            self._rate_limit()

            url = f"{self.BASE_URL}/paper/batch"
            fields = "paperId,title,abstract,authors,year,externalIds,openAccessPdf,venue"

            try:
                response = requests.post(
                    url,
                    headers=self._headers(),
                    params={"fields": fields},
                    json={"ids": batch},
                )
                response.raise_for_status()
                data = response.json()

                for item, input_id in zip(data, batch):
                    if item is None:
                        continue
                    if not item.get("title"):
                        continue
                    abstract = item.get("abstract") or ""
                    if not abstract:
                        continue
                    oa_pdf = item.get("openAccessPdf") or {}
                    results[input_id] = Paper(
                        id=item["paperId"],
                        title=item["title"],
                        abstract=abstract,
                        authors=[a.get("name", "") for a in item.get("authors", [])],
                        year=item.get("year"),
                        doi=item.get("externalIds", {}).get("DOI"),
                        pdf_url=oa_pdf.get("url"),
                        journal=item.get("venue") or None,
                    )
            except requests.RequestException as e:
                logger.warning("Error in S2 batch fetch: %s", e)

        return results
