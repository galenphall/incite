"""BibTeX parsing and metadata enrichment pipeline."""

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from tqdm import tqdm

from incite.corpus.loader import load_corpus, save_corpus
from incite.corpus.openalex import OpenAlexClient
from incite.corpus.semantic_scholar import SemanticScholarClient
from incite.models import Paper

logger = logging.getLogger(__name__)


class BibTeXParser:
    """Parser for BibTeX files."""

    @staticmethod
    def parse_file(path: str | Path) -> list[dict]:
        """Parse a .bib file and return list of entry dicts.

        Args:
            path: Path to .bib file

        Returns:
            List of entry dictionaries with extracted fields
        """

        with open(path, encoding="utf-8") as f:
            content = f.read()
        return BibTeXParser.parse_string(content)

    @staticmethod
    def parse_string(bibtex: str) -> list[dict]:
        """Parse a BibTeX string and return list of entry dicts.

        Args:
            bibtex: BibTeX string content

        Returns:
            List of entry dictionaries with extracted fields
        """
        import bibtexparser

        library = bibtexparser.parse_string(bibtex)

        entries = []
        for entry in library.entries:
            parsed = BibTeXParser._extract_entry(entry)
            if parsed:
                entries.append(parsed)

        return entries

    @staticmethod
    def _extract_entry(entry) -> Optional[dict]:
        """Extract relevant fields from a bibtexparser entry.

        Args:
            entry: bibtexparser Entry object

        Returns:
            Dict with extracted fields or None if entry is invalid
        """
        # Skip entries without a title
        title = entry.fields_dict.get("title")
        if not title:
            return None

        # Clean title (remove braces used for capitalization)
        title_val = BibTeXParser._clean_latex(title.value)

        # Extract authors
        authors = []
        author_field = entry.fields_dict.get("author")
        if author_field:
            authors = BibTeXParser._parse_authors(author_field.value)

        # Extract year
        year = None
        year_field = entry.fields_dict.get("year")
        if year_field:
            try:
                year = int(re.sub(r"[^\d]", "", year_field.value))
            except (ValueError, TypeError):
                pass

        # Extract DOI
        doi = None
        doi_field = entry.fields_dict.get("doi")
        if doi_field:
            doi = BibTeXParser._clean_doi(doi_field.value)

        # Extract abstract (if present)
        abstract = None
        abstract_field = entry.fields_dict.get("abstract")
        if abstract_field:
            abstract = BibTeXParser._clean_latex(abstract_field.value)

        # Extract journal/venue (try journal first, then booktitle for conferences)
        journal = None
        journal_field = entry.fields_dict.get("journal")
        if journal_field:
            journal = BibTeXParser._clean_latex(journal_field.value)
        elif entry.fields_dict.get("booktitle"):
            journal = BibTeXParser._clean_latex(entry.fields_dict["booktitle"].value)

        return {
            "key": entry.key,
            "title": title_val,
            "authors": authors,
            "year": year,
            "doi": doi,
            "abstract": abstract,
            "journal": journal,
            "entry_type": entry.entry_type,
        }

    @staticmethod
    def _clean_latex(text: str) -> str:
        """Remove LaTeX formatting from text."""
        if not text:
            return ""
        # Remove braces (used for capitalization protection)
        text = re.sub(r"[{}]", "", text)
        # Convert common LaTeX commands
        text = text.replace(r"\'", "'")
        text = text.replace(r"\"", '"')
        text = text.replace(r"\&", "&")
        text = re.sub(r"\\textit\{([^}]+)\}", r"\1", text)
        text = re.sub(r"\\textbf\{([^}]+)\}", r"\1", text)
        text = re.sub(r"\\emph\{([^}]+)\}", r"\1", text)
        # Clean up whitespace
        text = " ".join(text.split())
        return text

    @staticmethod
    def _parse_authors(author_str: str) -> list[str]:
        """Parse BibTeX author string into list of names."""
        if not author_str:
            return []

        # Split on " and " (BibTeX author separator)
        author_str = BibTeXParser._clean_latex(author_str)
        authors = re.split(r"\s+and\s+", author_str, flags=re.IGNORECASE)

        result = []
        for author in authors:
            author = author.strip()
            if not author:
                continue
            # Handle "Last, First" format -> "First Last"
            if "," in author:
                parts = [p.strip() for p in author.split(",", 1)]
                if len(parts) == 2:
                    author = f"{parts[1]} {parts[0]}"
            result.append(author)

        return result

    @staticmethod
    def _clean_doi(doi: str) -> Optional[str]:
        """Clean and normalize a DOI."""
        if not doi:
            return None
        # Remove URL prefix if present
        doi = re.sub(r"https?://doi\.org/", "", doi)
        doi = re.sub(r"https?://dx\.doi\.org/", "", doi)
        doi = doi.strip()
        return doi if doi else None


def bibtex_entries_to_papers(entries: list[dict]) -> list[Paper]:
    """Convert parsed BibTeX entry dicts to Paper objects.

    Args:
        entries: List of dicts from BibTeXParser.parse_string()

    Returns:
        List of Paper objects (entries without titles are skipped)
    """
    papers = []
    for entry in entries:
        title = entry.get("title", "").strip()
        if not title:
            continue

        # Deterministic ID from bibtex key
        key = entry.get("key", "")
        id_hash = hashlib.md5(key.encode()).hexdigest()[:12]
        paper_id = f"bib_{id_hash}"

        papers.append(
            Paper(
                id=paper_id,
                title=title,
                abstract=entry.get("abstract", ""),
                authors=entry.get("authors", []),
                year=entry.get("year"),
                doi=entry.get("doi"),
                journal=entry.get("journal"),
                bibtex_key=key,
                source_file="paperpile",
            )
        )
    return papers


def _word_set(s: str) -> set[str]:
    """Tokenize into lowercase alphanumeric words >= 3 chars."""
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) >= 3}


def match_pdfs_to_papers(pdf_filenames: list[str], papers: list[Paper]) -> dict[str, str]:
    """Match uploaded PDF filenames to papers using heuristics.

    Tries five strategies in order:
    1. Exact bibtex_key match (key.pdf)
    2. DOI-in-filename match (e.g. "10.1038_s41558-023-01234-5.pdf")
    3. Normalized title substring in filename
    4. Fuzzy Jaccard title similarity (word-level, threshold >= 0.5)
    5. Author last name + year in filename

    Args:
        pdf_filenames: List of uploaded PDF filenames (basename only)
        papers: List of Paper objects to match against

    Returns:
        Dict mapping paper_id to matched PDF filename
    """
    matches: dict[str, str] = {}
    unmatched_files = set(pdf_filenames)

    def _normalize(s: str) -> str:
        """Lowercase, strip non-alphanumeric."""
        return re.sub(r"[^a-z0-9]", "", s.lower())

    def _normalize_doi(doi: str) -> str:
        """Normalize a DOI for filename comparison."""
        return doi.lower().replace("/", "_").replace(":", "_").replace("-", "_")

    # Pass 1: exact bibtex key match (e.g. "Smith2020.pdf" matches key "Smith2020")
    for paper in papers:
        if paper.id in matches:
            continue
        if not paper.bibtex_key:
            continue
        norm_key = _normalize(paper.bibtex_key)
        for fname in list(unmatched_files):
            stem = _normalize(Path(fname).stem)
            if stem == norm_key:
                matches[paper.id] = fname
                unmatched_files.discard(fname)
                break

    # Pass 2: DOI-in-filename match
    for paper in papers:
        if paper.id in matches:
            continue
        if not paper.doi:
            continue
        norm_doi = _normalize_doi(paper.doi)
        if len(norm_doi) < 5:
            continue
        for fname in list(unmatched_files):
            norm_fname = Path(fname).stem.lower().replace("/", "_").replace("-", "_")
            if norm_doi in norm_fname:
                matches[paper.id] = fname
                unmatched_files.discard(fname)
                break

    # Pass 3: title substring in filename
    for paper in papers:
        if paper.id in matches:
            continue
        norm_title = _normalize(paper.title)
        if len(norm_title) < 10:
            continue
        for fname in list(unmatched_files):
            norm_fname = _normalize(Path(fname).stem)
            if norm_title[:40] in norm_fname or norm_fname in norm_title:
                matches[paper.id] = fname
                unmatched_files.discard(fname)
                break

    # Pass 4: fuzzy Jaccard title similarity
    for paper in papers:
        if paper.id in matches:
            continue
        title_words = _word_set(paper.title)
        if len(title_words) < 3:
            continue
        for fname in list(unmatched_files):
            fname_words = _word_set(Path(fname).stem)
            if not fname_words:
                continue
            overlap = title_words & fname_words
            union = title_words | fname_words
            jaccard = len(overlap) / len(union)
            if jaccard >= 0.5 and len(overlap) >= 3:
                matches[paper.id] = fname
                unmatched_files.discard(fname)
                break

    # Pass 5: author last name + year
    for paper in papers:
        if paper.id in matches:
            continue
        if not paper.authors or not paper.year:
            continue
        first_author_last = _normalize(paper.authors[0].split()[-1])
        year_str = str(paper.year)
        if len(first_author_last) < 3:
            continue
        for fname in list(unmatched_files):
            norm_fname = _normalize(Path(fname).stem)
            if first_author_last in norm_fname and year_str in fname:
                matches[paper.id] = fname
                unmatched_files.discard(fname)
                break

    return matches


class MetadataEnricher:
    """Enriches paper metadata from external APIs."""

    def __init__(
        self,
        s2_client: Optional[SemanticScholarClient] = None,
        openalex_client: Optional[OpenAlexClient] = None,
    ):
        """Initialize enricher with API clients.

        Args:
            s2_client: Semantic Scholar client (preferred for abstracts)
            openalex_client: OpenAlex client (fallback)
        """
        self.s2_client = s2_client
        self.openalex_client = openalex_client

    def enrich_from_doi(self, doi: str) -> Optional[Paper]:
        """Attempt to enrich metadata using DOI.

        Tries Semantic Scholar first (better abstract coverage),
        then falls back to OpenAlex.

        Args:
            doi: DOI string (without URL prefix)

        Returns:
            Paper object or None if both sources fail
        """
        # Try Semantic Scholar first (better abstract coverage)
        if self.s2_client:
            paper = self.s2_client.get_paper(f"DOI:{doi}")
            if paper and paper.title:
                paper.doi = doi
                return paper

        # Fall back to OpenAlex
        if self.openalex_client:
            paper = self._lookup_openalex_by_doi(doi)
            if paper:
                return paper

        return None

    def _lookup_openalex_by_doi(self, doi: str) -> Optional[Paper]:
        """Look up a paper in OpenAlex by DOI.

        Args:
            doi: DOI string

        Returns:
            Paper object or None
        """
        import requests

        self.openalex_client._rate_limit()

        # OpenAlex can filter by DOI
        url = f"{self.openalex_client.BASE_URL}/works/https://doi.org/{doi}"
        params = self.openalex_client._params()

        try:
            response = requests.get(url, params=params)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()

            abstract = OpenAlexClient.reconstruct_abstract(data.get("abstract_inverted_index", {}))

            authors = []
            for authorship in data.get("authorships", []):
                author = authorship.get("author", {})
                if author.get("display_name"):
                    authors.append(author["display_name"])

            openalex_id = data.get("id", "").split("/")[-1]

            return Paper(
                id=openalex_id,
                title=data.get("title", ""),
                abstract=abstract,
                authors=authors,
                year=data.get("publication_year"),
                doi=doi,
            )
        except requests.RequestException:
            return None

    def enrich_from_bibtex_entry(self, entry: dict) -> Paper:
        """Create Paper from BibTeX entry, enriching with APIs if possible.

        Always returns a Paper - degrades gracefully to BibTeX-only data.

        Args:
            entry: Dict from BibTeXParser with keys: key, title, authors, year, doi, abstract

        Returns:
            Paper object (always succeeds)
        """
        doi = entry.get("doi")

        # Try API enrichment if we have a DOI
        if doi:
            enriched = self.enrich_from_doi(doi)
            if enriched:
                # Preserve bibtex_key from the entry
                enriched.bibtex_key = entry.get("key")
                # If API didn't have abstract but BibTeX does, use BibTeX's
                if not enriched.abstract and entry.get("abstract"):
                    enriched.abstract = entry["abstract"]
                return enriched

        # Fall back to BibTeX-only data
        # Generate a stable ID from title (since we don't have S2/OpenAlex ID)
        title = entry.get("title", "")
        paper_id = self._generate_id(title, entry.get("key", ""))

        return Paper(
            id=paper_id,
            title=title,
            abstract=entry.get("abstract", ""),
            authors=entry.get("authors", []),
            year=entry.get("year"),
            doi=doi,
            bibtex_key=entry.get("key"),
        )

    @staticmethod
    def _generate_id(title: str, bibtex_key: str) -> str:
        """Generate a stable ID from title and bibtex key.

        Args:
            title: Paper title
            bibtex_key: BibTeX citation key

        Returns:
            Hash-based ID string
        """
        content = f"{title.lower().strip()}:{bibtex_key}"
        return f"bib:{hashlib.md5(content.encode()).hexdigest()[:16]}"

    def enrich_batch(
        self,
        entries: list[dict],
        show_progress: bool = False,
    ) -> list[Paper]:
        """Enrich a batch of BibTeX entries.

        Args:
            entries: List of entry dicts from BibTeXParser
            show_progress: Whether to show progress bar

        Returns:
            List of Paper objects
        """
        papers = []
        iterator = tqdm(entries, desc="Enriching") if show_progress else entries

        for entry in iterator:
            paper = self.enrich_from_bibtex_entry(entry)
            papers.append(paper)

        return papers


@dataclass(frozen=True)
class EnrichmentResult:
    """Summary of abstract enrichment results."""

    total_missing: int
    found_by_doi: int
    found_by_title: int
    still_missing: int


# Cap title-based lookups to bound latency (~1s each)
_TITLE_SEARCH_CAP = 50


def enrich_abstracts_batch(
    papers: list[Paper],
    s2_client: SemanticScholarClient | None = None,
    openalex_client: OpenAlexClient | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> EnrichmentResult:
    """Fetch missing abstracts from Semantic Scholar and OpenAlex.

    Modifies Paper.abstract in-place for papers where an abstract is found.
    Uses batch APIs to minimize request count.

    Args:
        papers: List of Paper objects (mutated in-place)
        s2_client: Semantic Scholar client (for batch + title search)
        openalex_client: OpenAlex client (DOI batch fallback)
        progress_callback: Optional callback for status messages

    Returns:
        EnrichmentResult summarizing what was found
    """
    missing = [p for p in papers if not p.abstract]
    if not missing:
        return EnrichmentResult(total_missing=0, found_by_doi=0, found_by_title=0, still_missing=0)

    total_missing = len(missing)
    found_by_doi = 0
    found_by_title = 0

    def _report(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        logger.info(msg)

    _report(f"Abstract enrichment: {total_missing} papers missing abstracts")

    # --- DOI path: batch lookups ---
    with_doi = [p for p in missing if p.doi]
    still_need_doi: list[Paper] = []

    if with_doi and s2_client:
        _report(f"Looking up {len(with_doi)} papers by DOI (Semantic Scholar)...")
        doi_ids = [f"DOI:{p.doi}" for p in with_doi]
        s2_results = s2_client.get_papers_batch(doi_ids)

        for paper, input_id in zip(with_doi, doi_ids):
            found = s2_results.get(input_id)
            if found and found.abstract:
                paper.abstract = found.abstract
                found_by_doi += 1
            else:
                still_need_doi.append(paper)
    else:
        still_need_doi = list(with_doi)

    # OpenAlex fallback for DOI papers S2 missed
    if still_need_doi and openalex_client:
        _report(f"OpenAlex fallback for {len(still_need_doi)} papers...")
        fallback_dois = [p.doi for p in still_need_doi]
        oa_results = openalex_client.get_works_batch_by_doi(fallback_dois)

        for paper in still_need_doi:
            found = oa_results.get(paper.doi.lower())
            if found and found.abstract:
                paper.abstract = found.abstract
                found_by_doi += 1

    # --- Title path: individual S2 search for no-DOI papers ---
    without_doi = [p for p in missing if not p.doi and not p.abstract]
    if without_doi and s2_client:
        capped = without_doi[:_TITLE_SEARCH_CAP]
        if len(without_doi) > _TITLE_SEARCH_CAP:
            _report(
                f"Title search: {len(without_doi)} no-DOI papers, capped at {_TITLE_SEARCH_CAP}"
            )
        else:
            _report(f"Title search for {len(capped)} no-DOI papers...")

        for paper in capped:
            try:
                results = s2_client.search_papers(paper.title, limit=3)
                for candidate in results:
                    if not candidate.abstract:
                        continue
                    # Fuzzy title match: Jaccard similarity on word sets
                    paper_words = _word_set(paper.title)
                    candidate_words = _word_set(candidate.title)
                    if not paper_words or not candidate_words:
                        continue
                    jaccard = len(paper_words & candidate_words) / len(
                        paper_words | candidate_words
                    )
                    if jaccard >= 0.7:
                        paper.abstract = candidate.abstract
                        found_by_title += 1
                        break
            except Exception:
                logger.debug("Title search failed for: %s", paper.title, exc_info=True)

    still_missing = sum(1 for p in missing if not p.abstract)
    result = EnrichmentResult(
        total_missing=total_missing,
        found_by_doi=found_by_doi,
        found_by_title=found_by_title,
        still_missing=still_missing,
    )
    _report(
        f"Abstract enrichment: {found_by_doi} by DOI, {found_by_title} by title, "
        f"{still_missing} still missing"
    )
    return result


def enrich_bibtex_to_corpus(
    bibtex_path: str,
    output_path: str,
    s2_api_key: Optional[str] = None,
    openalex_email: Optional[str] = None,
    skip_existing: bool = True,
) -> dict:
    """Load BibTeX, enrich metadata, save as corpus.jsonl.

    Args:
        bibtex_path: Path to .bib file
        output_path: Path to output corpus.jsonl
        s2_api_key: Optional Semantic Scholar API key
        openalex_email: Optional email for OpenAlex polite pool
        skip_existing: If True, don't re-fetch papers already in output

    Returns:
        Stats dict: {total, enriched, doi_found, abstract_found, skipped}
    """
    # Parse BibTeX
    print(f"Parsing BibTeX file: {bibtex_path}")
    entries = BibTeXParser.parse_file(bibtex_path)
    print(f"Found {len(entries)} entries")

    # Load existing corpus to check for duplicates
    existing_dois = set()
    existing_titles = set()
    existing_papers = []

    output_file = Path(output_path)
    if skip_existing and output_file.exists():
        try:
            existing_papers = load_corpus(output_path)
            for p in existing_papers:
                if p.doi:
                    existing_dois.add(p.doi.lower())
                existing_titles.add(p.title.lower().strip())
            print(f"Found {len(existing_papers)} existing papers in corpus")
        except Exception as e:
            print(f"Warning: Could not load existing corpus: {e}")

    # Filter out already-processed entries
    new_entries = []
    skipped = 0
    for entry in entries:
        doi = entry.get("doi")
        title = entry.get("title", "").lower().strip()

        if doi and doi.lower() in existing_dois:
            skipped += 1
            continue
        if title and title in existing_titles:
            skipped += 1
            continue

        new_entries.append(entry)

    if skipped:
        print(f"Skipping {skipped} entries already in corpus")

    # Create API clients
    s2_client = SemanticScholarClient(api_key=s2_api_key) if s2_api_key else None
    openalex_client = OpenAlexClient(email=openalex_email) if openalex_email else None

    if not s2_client and not openalex_client:
        print("Warning: No API credentials provided. Using BibTeX data only.")
    else:
        apis = []
        if s2_client:
            apis.append("Semantic Scholar")
        if openalex_client:
            apis.append("OpenAlex")
        print(f"Using APIs: {', '.join(apis)}")

    # Enrich entries
    enricher = MetadataEnricher(s2_client=s2_client, openalex_client=openalex_client)
    new_papers = enricher.enrich_batch(new_entries, show_progress=True)

    # Combine with existing papers
    all_papers = existing_papers + new_papers

    # Calculate stats
    stats = {
        "total": len(all_papers),
        "new": len(new_papers),
        "skipped": skipped,
        "doi_found": sum(1 for p in new_papers if p.doi),
        "abstract_found": sum(1 for p in new_papers if p.abstract),
        "api_enriched": sum(1 for p in new_papers if not p.id.startswith("bib:")),
    }

    # Save corpus
    save_corpus(all_papers, output_path)
    print(f"Saved {len(all_papers)} papers to {output_path}")

    return stats
