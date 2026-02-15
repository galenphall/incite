"""Abstract enrichment pipeline for training data.

Fetches missing or short abstracts for training examples via cascading
API lookup: Semantic Scholar -> OpenAlex, with disk-backed JSONL caching.

Usage:
    from incite.finetuning.enrichment import AbstractEnricher, enrich_examples
    from incite.finetuning.types import load_training_data

    enricher = AbstractEnricher()
    examples = load_training_data(Path("data/finetuning/master_train.jsonl"))
    enriched, stats = enrich_examples(examples, enricher, min_abstract_length=100)
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from incite.corpus.openalex import OpenAlexClient
from incite.corpus.semantic_scholar import SemanticScholarClient
from incite.finetuning.types import TrainingExample
from incite.models import (
    strip_metadata_prefix,
)

logger = logging.getLogger(__name__)

# Default cache location relative to project root
DEFAULT_CACHE_PATH = Path("data/finetuning/enrichment_cache.jsonl")

# Max retries for transient API errors
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds, doubles each retry


@dataclass
class CacheEntry:
    """A cached abstract lookup result."""

    doi: str
    title: str
    abstract: str
    source: str  # "semantic_scholar", "openalex", or "not_found"

    def to_dict(self) -> dict:
        return {
            "doi": self.doi,
            "title": self.title,
            "abstract": self.abstract,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CacheEntry":
        return cls(
            doi=data.get("doi", ""),
            title=data.get("title", ""),
            abstract=data.get("abstract", ""),
            source=data.get("source", ""),
        )


@dataclass
class EnrichmentStats:
    """Statistics from an enrichment run."""

    total_examples: int = 0
    short_abstracts_found: int = 0
    enriched_from_s2: int = 0
    enriched_from_openalex: int = 0
    cache_hits: int = 0
    not_found: int = 0
    errors: int = 0
    skipped_no_identifier: int = 0

    @property
    def total_enriched(self) -> int:
        return self.enriched_from_s2 + self.enriched_from_openalex + self.cache_hits

    def __str__(self) -> str:
        lines = [
            "Enrichment Stats:",
            f"  Total examples:       {self.total_examples}",
            f"  Short/missing:        {self.short_abstracts_found}",
            f"  Enriched (total):     {self.total_enriched}",
            f"    From S2:            {self.enriched_from_s2}",
            f"    From OpenAlex:      {self.enriched_from_openalex}",
            f"    From cache:         {self.cache_hits}",
            f"  Not found:            {self.not_found}",
            f"  No identifier:        {self.skipped_no_identifier}",
            f"  Errors:               {self.errors}",
        ]
        return "\n".join(lines)


def _normalize_title(title: str) -> str:
    """Normalize a title for cache lookup (lowercase, strip punctuation)."""
    return re.sub(r"[^a-z0-9\s]", "", title.lower()).strip()


def _extract_title_from_positive(positive: str) -> str:
    """Extract the title from a formatted positive string.

    The positive is formatted as "Title. Authors. Year. Journal. Abstract"
    via format_paper_embedding_text(). The title is the first segment before
    the first ". " separator.
    """
    # Split on ". " and take the first part as the title
    parts = positive.split(". ", 1)
    return parts[0].strip() if parts else ""


class AbstractEnricher:
    """Cascading abstract lookup with disk cache.

    Lookup order:
    1. Disk cache (JSONL file)
    2. Semantic Scholar API (by DOI or title search)
    3. OpenAlex API (by DOI or title search)

    Results (including not-found) are cached to avoid repeated API calls.
    """

    def __init__(
        self,
        cache_path: Path = DEFAULT_CACHE_PATH,
        s2_api_key: Optional[str] = None,
        openalex_email: Optional[str] = None,
    ):
        """Initialize the enricher with API clients and cache.

        Args:
            cache_path: Path to JSONL cache file.
            s2_api_key: Semantic Scholar API key (falls back to
                SEMANTIC_SCHOLAR_API_KEY env var).
            openalex_email: Email for OpenAlex polite pool (falls back to
                OPENALEX_EMAIL env var).
        """
        self.cache_path = cache_path

        # Initialize API clients
        s2_key = s2_api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        oa_email = openalex_email or os.environ.get("OPENALEX_EMAIL")

        self._s2 = SemanticScholarClient(api_key=s2_key, delay=1.0)
        self._openalex = OpenAlexClient(email=oa_email, delay=0.1)

        # In-memory cache: normalized_title -> CacheEntry
        self._cache: dict[str, CacheEntry] = {}
        # Also index by DOI for faster lookup
        self._doi_cache: dict[str, CacheEntry] = {}

        self._load_cache()

    def _load_cache(self) -> None:
        """Load existing cache entries from disk."""
        if not self.cache_path.exists():
            logger.info("No existing cache at %s", self.cache_path)
            return

        count = 0
        with open(self.cache_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = CacheEntry.from_dict(json.loads(line))
                    norm_title = _normalize_title(entry.title)
                    if norm_title:
                        self._cache[norm_title] = entry
                    if entry.doi:
                        self._doi_cache[entry.doi.lower()] = entry
                    count += 1
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning("Skipping malformed cache line: %s", e)

        logger.info("Loaded %d cache entries from %s", count, self.cache_path)

    def _save_entry(self, entry: CacheEntry) -> None:
        """Append a single cache entry to disk and update in-memory cache."""
        norm_title = _normalize_title(entry.title)
        if norm_title:
            self._cache[norm_title] = entry
        if entry.doi:
            self._doi_cache[entry.doi.lower()] = entry

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def _lookup_cache(self, title: str, doi: str = "") -> Optional[CacheEntry]:
        """Check in-memory cache by DOI or normalized title."""
        if doi:
            entry = self._doi_cache.get(doi.lower())
            if entry is not None:
                return entry

        norm = _normalize_title(title)
        if norm:
            return self._cache.get(norm)

        return None

    def _fetch_with_retry(self, fetch_fn, description: str) -> Optional[object]:
        """Call a fetch function with exponential backoff on transient errors."""
        for attempt in range(MAX_RETRIES):
            try:
                return fetch_fn()
            except Exception as e:
                wait = RETRY_BACKOFF * (2**attempt)
                logger.warning(
                    "Attempt %d/%d for %s failed: %s. Retrying in %.1fs",
                    attempt + 1,
                    MAX_RETRIES,
                    description,
                    e,
                    wait,
                )
                time.sleep(wait)
        return None

    def fetch_abstract(self, title: str, doi: str = "") -> Optional[CacheEntry]:
        """Fetch an abstract via cascading lookup.

        Args:
            title: Paper title for search-based lookup.
            doi: DOI for direct lookup (preferred when available).

        Returns:
            CacheEntry with the abstract, or None if all sources fail.
            The entry is cached to disk regardless of outcome.
        """
        # 1. Check cache
        cached = self._lookup_cache(title, doi)
        if cached is not None:
            return cached

        # 2. Try Semantic Scholar
        paper = self._try_semantic_scholar(title, doi)
        if paper is not None and paper.abstract and len(paper.abstract) >= 50:
            entry = CacheEntry(
                doi=paper.doi or doi,
                title=title,
                abstract=paper.abstract,
                source="semantic_scholar",
            )
            self._save_entry(entry)
            return entry

        # 3. Try OpenAlex
        paper = self._try_openalex(title, doi)
        if paper is not None and paper.abstract and len(paper.abstract) >= 50:
            entry = CacheEntry(
                doi=paper.doi or doi,
                title=title,
                abstract=paper.abstract,
                source="openalex",
            )
            self._save_entry(entry)
            return entry

        # 4. Cache the miss to avoid repeated lookups
        entry = CacheEntry(
            doi=doi,
            title=title,
            abstract="",
            source="not_found",
        )
        self._save_entry(entry)
        return entry

    def _try_semantic_scholar(self, title: str, doi: str) -> Optional[object]:
        """Try Semantic Scholar by DOI first, then title search."""
        if doi:
            paper = self._fetch_with_retry(
                lambda: self._s2.get_paper(doi),
                f"S2 DOI lookup: {doi}",
            )
            if paper is not None:
                return paper

        # Title-based search as fallback
        if title:
            paper = self._fetch_with_retry(
                lambda: self._s2_title_search(title),
                f"S2 title search: {title[:60]}",
            )
            if paper is not None:
                return paper

        return None

    def _s2_title_search(self, title: str):
        """Search S2 by title and return best match."""
        results = self._s2.search_papers(title, limit=3)
        if not results:
            return None
        # Find best title match
        norm_query = _normalize_title(title)
        for paper in results:
            if _normalize_title(paper.title) == norm_query:
                return paper
        return None

    def _try_openalex(self, title: str, doi: str) -> Optional[object]:
        """Try OpenAlex by DOI first, then title search."""
        if doi:
            paper = self._fetch_with_retry(
                lambda: self._openalex_doi_lookup(doi),
                f"OpenAlex DOI lookup: {doi}",
            )
            if paper is not None:
                return paper

        # Title search as fallback
        if title:
            paper = self._fetch_with_retry(
                lambda: self._openalex_title_search(title),
                f"OpenAlex title search: {title[:60]}",
            )
            if paper is not None:
                return paper

        return None

    def _openalex_doi_lookup(self, doi: str):
        """Look up a paper in OpenAlex by DOI."""
        import requests

        self._openalex._rate_limit()
        url = f"{OpenAlexClient.BASE_URL}/works/doi:{doi}"
        params = self._openalex._params()

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        abstract = OpenAlexClient.reconstruct_abstract(data.get("abstract_inverted_index", {}))
        if not abstract:
            return None

        doi_val = data.get("doi", "")
        if doi_val:
            doi_val = doi_val.replace("https://doi.org/", "")

        from incite.models import Paper

        return Paper(
            id=data.get("id", "").split("/")[-1],
            title=data.get("title", ""),
            abstract=abstract,
            authors=[
                a.get("author", {}).get("display_name", "") for a in data.get("authorships", [])
            ],
            year=data.get("publication_year"),
            doi=doi_val or None,
        )

    def _openalex_title_search(self, title: str):
        """Search OpenAlex by title and return best match."""
        import requests

        self._openalex._rate_limit()
        url = f"{OpenAlexClient.BASE_URL}/works"
        params = {
            **self._openalex._params(),
            "search": title,
            "per-page": 3,
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        norm_query = _normalize_title(title)
        for item in data.get("results", []):
            item_title = item.get("title", "")
            if _normalize_title(item_title) == norm_query:
                abstract = OpenAlexClient.reconstruct_abstract(
                    item.get("abstract_inverted_index", {})
                )
                if abstract:
                    doi_val = item.get("doi", "")
                    if doi_val:
                        doi_val = doi_val.replace("https://doi.org/", "")

                    from incite.models import Paper

                    return Paper(
                        id=item.get("id", "").split("/")[-1],
                        title=item_title,
                        abstract=abstract,
                        authors=[
                            a.get("author", {}).get("display_name", "")
                            for a in item.get("authorships", [])
                        ],
                        year=item.get("publication_year"),
                        doi=doi_val or None,
                    )
        return None


def _extract_doi_from_example(example: TrainingExample) -> str:
    """Try to extract a DOI from the example's cited_paper_id field.

    Many training examples use DOI as cited_paper_id (e.g. "10.1038/s41586-...").
    """
    paper_id = example.cited_paper_id
    if not paper_id:
        return ""
    # DOI pattern: starts with "10." followed by registrant/suffix
    if re.match(r"^10\.\d{4,}/.+", paper_id):
        return paper_id
    return ""


def enrich_examples(
    examples: list[TrainingExample],
    enricher: AbstractEnricher,
    min_abstract_length: int = 100,
    show_progress: bool = True,
) -> tuple[list[TrainingExample], EnrichmentStats]:
    """Scan training examples and enrich short/missing abstracts.

    Identifies examples whose positive field has a short abstract (after
    stripping the metadata prefix), fetches replacements via the enricher,
    and rebuilds the positive with the new abstract.

    Args:
        examples: List of training examples (not modified in place).
        enricher: AbstractEnricher instance with API clients and cache.
        min_abstract_length: Minimum abstract character count. Positives
            with shorter abstracts are candidates for enrichment.
        show_progress: Show a progress bar via tqdm if available.

    Returns:
        Tuple of (enriched examples list, enrichment stats).
    """
    stats = EnrichmentStats(total_examples=len(examples))
    enriched = []

    # Set up progress bar
    iterator = examples
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(examples, desc="Enriching abstracts", unit="ex")
        except ImportError:
            pass

    for example in iterator:
        positive = example.positive
        abstract_text = strip_metadata_prefix(positive)

        # Check if abstract is short/missing
        if len(abstract_text) >= min_abstract_length:
            enriched.append(example)
            continue

        stats.short_abstracts_found += 1

        # Extract identifiers for lookup
        title = _extract_title_from_positive(positive)
        doi = _extract_doi_from_example(example)

        if not title and not doi:
            stats.skipped_no_identifier += 1
            enriched.append(example)
            continue

        # Attempt enrichment
        try:
            result = enricher.fetch_abstract(title, doi)
        except Exception as e:
            logger.error("Enrichment error for '%s': %s", title[:60], e)
            stats.errors += 1
            enriched.append(example)
            continue

        if result is None or result.source == "not_found" or not result.abstract:
            stats.not_found += 1
            enriched.append(example)
            continue

        # Track the source
        if result.source == "semantic_scholar":
            stats.enriched_from_s2 += 1
        elif result.source == "openalex":
            stats.enriched_from_openalex += 1
        else:
            stats.cache_hits += 1

        # Rebuild the positive with the new abstract.
        # Parse existing metadata from the positive prefix to preserve it.
        new_positive = _rebuild_positive_with_abstract(positive, result.abstract)

        new_example = TrainingExample(
            query=example.query,
            positive=new_positive,
            hard_negatives=example.hard_negatives,
            source_paper_id=example.source_paper_id,
            cited_paper_id=example.cited_paper_id,
            source=example.source,
            scale=example.scale,
            passage_positive=example.passage_positive,
            passage_score=example.passage_score,
            passage_validation=example.passage_validation,
            intent=example.intent,
            passage_section=example.passage_section,
            passage_hard_negatives=example.passage_hard_negatives,
        )
        enriched.append(new_example)

    return enriched, stats


def _rebuild_positive_with_abstract(
    original_positive: str,
    new_abstract: str,
) -> str:
    """Replace the abstract portion of a formatted positive string.

    The positive is formatted as "Title. Authors. Year. Journal. Abstract"
    via format_paper_embedding_text(). We preserve the metadata prefix
    and replace only the abstract/content portion.

    Args:
        original_positive: Original positive string with metadata prefix.
        new_abstract: New abstract text to substitute.

    Returns:
        Rebuilt positive string with metadata prefix + new abstract.
    """
    from incite.models import _METADATA_PREFIX_PATTERN

    match = _METADATA_PREFIX_PATTERN.match(original_positive)
    if match and len(original_positive) - match.end() < len(new_abstract):
        # Prefix matched and new abstract is longer -> replace
        prefix = original_positive[: match.end()].rstrip()
        # Ensure proper ". " separator between prefix and abstract
        if not prefix.endswith("."):
            prefix += "."
        return f"{prefix} {new_abstract}"

    # No prefix match or original abstract is already longer.
    # Fall back to replacing everything after the title.
    title = _extract_title_from_positive(original_positive)
    if title:
        # Check for author/year/journal metadata segments before the abstract
        # by looking for the short abstract that needs replacement
        old_abstract = strip_metadata_prefix(original_positive)
        if old_abstract and old_abstract in original_positive:
            return original_positive.replace(old_abstract, new_abstract, 1)

    # Last resort: just replace content after stripping
    stripped = strip_metadata_prefix(original_positive)
    if stripped and stripped in original_positive:
        return original_positive.replace(stripped, new_abstract, 1)

    # Cannot parse structure, return original unchanged
    logger.warning(
        "Could not parse positive structure for enrichment: %s",
        original_positive[:80],
    )
    return original_positive
