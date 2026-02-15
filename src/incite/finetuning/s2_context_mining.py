"""Mine citation contexts from the Semantic Scholar Graph API.

Searches for highly-cited seed papers across diverse academic fields, fetches
their citation contexts (text snippets where they are cited), and produces
TrainingExamples with co-citation hard negatives. Resumable via per-seed
output files and state tracking.

Complementary to the S2ORC shard streaming pipeline — simpler (API calls vs
GB shard downloads), faster (~30 min vs hours), and produces the same quality
data (real citation contexts with co-citation hard negatives).

Usage (via scripts/mine_s2_citation_contexts.py):
    python scripts/mine_s2_citation_contexts.py \\
        --target 25000 --output-dir data/finetuning/s2_contexts
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from incite.finetuning.quality import clean_text
from incite.finetuning.types import TrainingExample, save_training_data
from incite.models import (
    Paper,
    clean_citation_markers,
    format_paper_embedding_text,
    strip_metadata_prefix,
)

logger = logging.getLogger(__name__)

S2_GRAPH_API = "https://api.semanticscholar.org/graph/v1"
STATE_FILENAME = "state.json"
SEEDS_FILENAME = "seeds.jsonl"
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 5.0  # seconds

# Diverse academic search queries for seed paper selection (~25 queries, 8+ fields)
FIELD_QUERIES = [
    {
        "field": "Computer Science",
        "queries": [
            "deep learning neural networks",
            "natural language processing transformers",
            "computer vision object detection",
            "reinforcement learning policy optimization",
        ],
    },
    {
        "field": "Biology",
        "queries": [
            "gene expression regulation",
            "protein structure prediction",
            "CRISPR genome editing",
            "gut microbiome diversity",
        ],
    },
    {
        "field": "Medicine",
        "queries": [
            "clinical trial randomized controlled",
            "cancer immunotherapy checkpoint",
            "vaccine efficacy safety",
        ],
    },
    {
        "field": "Physics",
        "queries": [
            "quantum computing qubits",
            "dark matter detection",
            "gravitational waves observation",
        ],
    },
    {
        "field": "Psychology",
        "queries": [
            "cognitive development children",
            "mental health intervention treatment",
            "social cognition theory of mind",
        ],
    },
    {
        "field": "Economics",
        "queries": [
            "monetary policy inflation",
            "economic growth inequality",
            "behavioral economics decision making",
        ],
    },
    {
        "field": "Materials Science",
        "queries": [
            "lithium ion battery cathode",
            "nanomaterials synthesis characterization",
        ],
    },
    {
        "field": "Environmental Science",
        "queries": [
            "climate change adaptation mitigation",
            "biodiversity conservation species",
            "renewable energy solar photovoltaic",
        ],
    },
]


def _paper_to_dict(paper: Paper) -> dict:
    """Serialize a Paper to a dict for JSONL storage."""
    return {
        "id": paper.id,
        "title": paper.title,
        "abstract": paper.abstract,
        "authors": paper.authors,
        "year": paper.year,
        "journal": paper.journal,
    }


def _paper_from_dict(d: dict) -> Paper:
    """Deserialize a Paper from a dict."""
    return Paper(
        id=d["id"],
        title=d["title"],
        abstract=d.get("abstract", ""),
        authors=d.get("authors", []),
        year=d.get("year"),
        journal=d.get("journal"),
    )


def save_seeds(seeds: list[Paper], path: Path) -> None:
    """Save seed papers to JSONL for resumability and backfill."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in seeds:
            f.write(json.dumps(_paper_to_dict(s)) + "\n")


def load_seeds(path: Path) -> list[Paper]:
    """Load seed papers from JSONL."""
    seeds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(_paper_from_dict(json.loads(line)))
    return seeds


def select_seed_papers(
    api_key: Optional[str] = None,
    target: int = 1500,
    min_citations: int = 50,
    max_per_field: int = 200,
    delay: float = 1.0,
    show_progress: bool = True,
) -> list[Paper]:
    """Select highly-cited seed papers from diverse academic fields.

    Uses the S2 Graph API search endpoint with citationCount field to find
    papers with >= min_citations and non-empty abstracts, then deduplicates
    by paper ID and caps per field to prevent disciplinary skew.

    Args:
        api_key: Semantic Scholar API key for higher rate limits.
        target: Target number of seed papers.
        min_citations: Minimum citation count for seeds.
        max_per_field: Cap per field to prevent skew.
        delay: Delay between API requests in seconds.
        show_progress: Whether to show progress bar.

    Returns:
        List of Paper objects with high citation counts.
    """
    headers = {"x-api-key": api_key} if api_key else {}
    seen_ids: set[str] = set()
    papers_by_field: dict[str, list[Paper]] = {}
    last_request = 0.0

    all_queries = []
    for field_group in FIELD_QUERIES:
        field_name = field_group["field"]
        for query in field_group["queries"]:
            all_queries.append((field_name, query))

    desc = "Selecting seed papers"
    iterable = tqdm(all_queries, desc=desc, disable=not show_progress)

    for field_name, query in iterable:
        # Rate limit
        elapsed = time.time() - last_request
        if elapsed < delay:
            time.sleep(delay - elapsed)

        url = (
            f"{S2_GRAPH_API}/paper/search"
            f"?query={requests.utils.quote(query)}"
            f"&fields=paperId,title,abstract,authors,year,journal,citationCount"
            f"&limit=100"
        )

        try:
            last_request = time.time()
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.warning("Search failed for '%s': %s", query, e)
            continue

        field_papers = papers_by_field.setdefault(field_name, [])

        for item in data.get("data", []):
            pid = item.get("paperId", "")
            if not pid or pid in seen_ids:
                continue

            citation_count = item.get("citationCount", 0) or 0
            if citation_count < min_citations:
                continue

            title = item.get("title", "")
            abstract = item.get("abstract", "")
            if not title or not abstract:
                continue

            seen_ids.add(pid)
            authors = [a.get("name", "") for a in item.get("authors", [])]
            journal_info = item.get("journal") or {}
            journal_name = journal_info.get("name") if isinstance(journal_info, dict) else None

            paper = Paper(
                id=pid,
                title=title,
                abstract=abstract,
                authors=authors,
                year=item.get("year"),
                journal=journal_name,
            )

            if len(field_papers) < max_per_field:
                field_papers.append(paper)

        total = sum(len(v) for v in papers_by_field.values())
        iterable.set_postfix(seeds=total)

        if total >= target:
            break

    # Flatten and limit to target
    all_papers = []
    for papers in papers_by_field.values():
        all_papers.extend(papers)

    if len(all_papers) > target:
        rng = random.Random(42)
        rng.shuffle(all_papers)
        all_papers = all_papers[:target]

    logger.info(
        "Selected %d seed papers from %d fields",
        len(all_papers),
        len(papers_by_field),
    )
    for field_name, papers in sorted(papers_by_field.items()):
        logger.info("  %s: %d seeds", field_name, len(papers))

    return all_papers


class CitationContextHarvester:
    """Harvests citation contexts from S2 Graph API for seed papers.

    For each seed paper, fetches all papers that cite it along with the
    actual citation context text. Produces TrainingExamples with co-citation
    hard negatives (other seeds cited by the same citing paper).

    The co-citation map grows incrementally: as more seeds are processed,
    later seeds benefit from richer hard negative pools. The optional
    backfill pass enriches early examples using the complete map.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        min_context_length: int = 50,
        max_hard_negatives: int = 5,
        max_contexts_per_seed: int = 50,
        delay: float = 1.0,
    ):
        """Initialize the harvester.

        Args:
            api_key: Semantic Scholar API key for higher rate limits.
            min_context_length: Minimum character length for citation contexts.
            max_hard_negatives: Maximum co-citation hard negatives per example.
            max_contexts_per_seed: Maximum citation contexts to fetch per seed
                paper. Prevents highly-cited seeds from dominating the budget.
            delay: Delay between API requests in seconds.
        """
        self.api_key = api_key
        self.min_context_length = min_context_length
        self.max_hard_negatives = max_hard_negatives
        self.max_contexts_per_seed = max_contexts_per_seed
        self.delay = delay
        self._last_request = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between API requests."""
        elapsed = time.time() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.time()

    def _headers(self) -> dict:
        """Get request headers with optional API key."""
        headers: dict[str, str] = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _fetch_citations_page(
        self,
        paper_id: str,
        offset: int = 0,
        limit: int = 500,
    ) -> dict:
        """Fetch one page of citations with contexts from the S2 API.

        Retries with exponential backoff on 429/5xx errors.

        Args:
            paper_id: Semantic Scholar paper ID of the cited paper.
            offset: Pagination offset.
            limit: Number of results per page (max 1000).

        Returns:
            Raw JSON response dict with 'data' and optional 'next' keys.
        """
        url = (
            f"{S2_GRAPH_API}/paper/{paper_id}/citations"
            f"?fields=contexts,citingPaper.paperId"
            f"&offset={offset}&limit={limit}"
        )

        for attempt in range(MAX_RETRIES):
            try:
                self._rate_limit()
                resp = requests.get(url, headers=self._headers(), timeout=30)

                if resp.status_code == 429 or resp.status_code >= 500:
                    wait = RETRY_BACKOFF_BASE * (2**attempt)
                    logger.warning(
                        "API %d for %s (offset=%d), retrying in %.0fs",
                        resp.status_code,
                        paper_id,
                        offset,
                        wait,
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()

            except requests.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF_BASE * (2**attempt)
                    logger.warning(
                        "Request failed for %s (%s), retrying in %.0fs",
                        paper_id,
                        e,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "Failed after %d attempts for %s: %s",
                        MAX_RETRIES,
                        paper_id,
                        e,
                    )
                    return {"data": []}

        return {"data": []}

    def _fetch_all_citation_contexts(
        self,
        paper_id: str,
    ) -> list[tuple[str, str]]:
        """Fetch all citation contexts for a paper, paginating through results.

        Args:
            paper_id: Semantic Scholar paper ID of the cited paper.

        Returns:
            List of (citing_paper_id, context_text) tuples.
            Only includes non-empty contexts meeting minimum length.
        """
        results: list[tuple[str, str]] = []
        offset = 0
        limit = 500

        while True:
            data = self._fetch_citations_page(paper_id, offset, limit)
            items = data.get("data", [])

            if not items:
                break

            for item in items:
                citing_paper = item.get("citingPaper") or {}
                citing_id = citing_paper.get("paperId", "")
                if not citing_id:
                    continue

                contexts = item.get("contexts") or []
                for ctx in contexts:
                    if ctx and len(ctx.strip()) >= self.min_context_length:
                        results.append((citing_id, ctx.strip()))

                # Early exit: cap per-seed to prevent highly-cited papers
                # from dominating the entire budget
                if len(results) >= self.max_contexts_per_seed:
                    return results

            # Check for next page
            next_offset = data.get("next")
            if next_offset is None or len(items) < limit:
                break
            offset = next_offset

        return results

    def harvest_seed(
        self,
        seed: Paper,
        seed_texts: dict[str, str],
        co_citation_map: dict[str, set[str]],
    ) -> list[TrainingExample]:
        """Harvest citation contexts for one seed paper.

        Fetches all citations of the seed, cleans context text, builds
        TrainingExamples with co-citation hard negatives, and updates the
        co_citation_map as new citations are discovered.

        Args:
            seed: The seed paper being cited.
            seed_texts: Mapping of seed paper ID -> canonical embedding text.
            co_citation_map: Mutable map of citing_paper_id -> {seed_ids}.
                Updated in place as new citations are discovered.

        Returns:
            List of TrainingExamples from this seed's citation contexts.
        """
        raw_contexts = self._fetch_all_citation_contexts(seed.id)
        if not raw_contexts:
            return []

        positive_text = seed_texts[seed.id]
        positive_core = strip_metadata_prefix(positive_text)
        examples: list[TrainingExample] = []

        for citing_id, raw_ctx in raw_contexts:
            # Update co-citation map
            co_citation_map.setdefault(citing_id, set()).add(seed.id)

            # Clean context: remove citation markers, HTML, normalize whitespace
            cleaned = clean_citation_markers(raw_ctx)
            cleaned = clean_text(cleaned)

            if len(cleaned) < self.min_context_length:
                continue

            # Build co-citation hard negatives: other seeds cited by same paper
            co_cited_seeds = co_citation_map.get(citing_id, set())
            other_seeds = [sid for sid in co_cited_seeds if sid != seed.id]

            hard_negatives: list[str] = []
            rng = random.Random(hash(f"{seed.id}_{citing_id}_{len(examples)}"))
            if other_seeds:
                neg_ids = rng.sample(
                    other_seeds,
                    min(self.max_hard_negatives, len(other_seeds)),
                )
                for neg_id in neg_ids:
                    neg_text = seed_texts.get(neg_id, "")
                    if not neg_text:
                        continue
                    neg_core = strip_metadata_prefix(neg_text)
                    # Format-aware dedup: don't use positive as hard negative
                    if neg_core == positive_core:
                        continue
                    hard_negatives.append(neg_text)

            example = TrainingExample(
                query=cleaned,
                positive=positive_text,
                hard_negatives=hard_negatives,
                source_paper_id=citing_id,
                cited_paper_id=seed.id,
                source="s2_contexts",
                scale="narrow",
            )
            examples.append(example)

        return examples

    def harvest_all(
        self,
        seeds: list[Paper],
        output_dir: Path,
        target: int = 25000,
        show_progress: bool = True,
    ) -> dict:
        """Harvest citation contexts from all seed papers with resumable state.

        Iterates over seeds, calls harvest_seed() for each, writes per-seed
        JSONL output files, and saves state after each seed for resumability.
        Stops when the target example count is reached.

        Args:
            seeds: List of seed papers to harvest from.
            output_dir: Directory for per-seed output files and state.
            target: Target number of training examples.
            show_progress: Whether to show progress bars.

        Returns:
            Stats dict with total_examples, seeds_processed, etc.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        state_path = output_dir / STATE_FILENAME

        # Save seeds for backfill/resume
        seeds_path = output_dir / SEEDS_FILENAME
        if not seeds_path.exists():
            save_seeds(seeds, seeds_path)

        state = _load_state(state_path)
        completed_seeds = set(state.get("completed_seeds", []))
        total_examples = state.get("total_examples", 0)

        # Restore co-citation map (JSON lists -> sets)
        co_citation_raw = state.get("co_citation_map", {})
        co_citation_map: dict[str, set[str]] = {k: set(v) for k, v in co_citation_raw.items()}

        seed_stats = state.get("seed_stats", {})

        # Pre-compute embedding texts for all seeds (canonical format)
        seed_texts: dict[str, str] = {}
        for s in seeds:
            seed_texts[s.id] = format_paper_embedding_text(
                title=s.title,
                abstract=s.abstract,
                author_lastnames=s.author_lastnames,
                year=s.year,
                journal=s.journal,
                include_abstract=True,
                include_metadata=True,
            )

        desc = "Harvesting citation contexts"
        iterable = tqdm(
            enumerate(seeds),
            total=len(seeds),
            desc=desc,
            disable=not show_progress,
        )

        for i, seed in iterable:
            if seed.id in completed_seeds:
                continue

            if total_examples >= target:
                logger.info(
                    "Target reached (%d/%d examples), stopping.",
                    total_examples,
                    target,
                )
                break

            examples = self.harvest_seed(seed, seed_texts, co_citation_map)

            if examples:
                seed_output = output_dir / f"seed_{i:04d}.jsonl"
                save_training_data(examples, seed_output)

            total_examples += len(examples)
            completed_seeds.add(seed.id)
            seed_stats[seed.id] = {
                "contexts_harvested": len(examples),
                "seed_index": i,
            }

            # Save state after each seed for resumability
            state["completed_seeds"] = sorted(completed_seeds)
            state["total_examples"] = total_examples
            state["co_citation_map"] = {k: sorted(v) for k, v in co_citation_map.items()}
            state["seed_stats"] = seed_stats
            _save_state(state_path, state)

            iterable.set_postfix(
                examples=total_examples,
                seed_ctx=len(examples),
            )

        stats = {
            "total_examples": total_examples,
            "seeds_processed": len(completed_seeds),
            "seeds_total": len(seeds),
            "co_citation_pairs": len(co_citation_map),
        }

        logger.info(
            "Harvest complete: %d examples from %d/%d seeds (%d co-citation pairs)",
            total_examples,
            len(completed_seeds),
            len(seeds),
            len(co_citation_map),
        )

        return stats


def merge_to_splits(
    output_dir: Path,
    dev_fraction: float = 0.2,
    seed: int = 42,
) -> dict:
    """Merge per-seed output files into train/dev splits.

    Groups examples by cited_paper_id (seed paper), then splits seeds into
    train and dev sets to prevent train/dev leakage. Shuffles within each
    split for training stability.

    Args:
        output_dir: Directory containing seed_*.jsonl files.
        dev_fraction: Fraction of seed papers allocated to dev set.
        seed: Random seed for reproducible splits.

    Returns:
        Stats dict with train_count, dev_count, seeds, etc.
    """
    seed_files = sorted(output_dir.glob("seed_*.jsonl"))
    if not seed_files:
        logger.warning("No seed files found in %s", output_dir)
        return {"train_count": 0, "dev_count": 0, "seeds": 0}

    # Load all examples and group by cited paper (seed)
    examples_by_seed: dict[str, list[TrainingExample]] = {}
    total_loaded = 0

    for path in seed_files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = TrainingExample.from_dict(json.loads(line))
                seed_key = ex.cited_paper_id or "unknown"
                examples_by_seed.setdefault(seed_key, []).append(ex)
                total_loaded += 1

    logger.info(
        "Loaded %d examples from %d files (%d seeds)",
        total_loaded,
        len(seed_files),
        len(examples_by_seed),
    )

    # Split by seed paper (no train/dev leakage)
    rng = random.Random(seed)
    seed_ids = list(examples_by_seed.keys())
    rng.shuffle(seed_ids)

    dev_count = max(1, int(len(seed_ids) * dev_fraction))
    dev_seed_ids = set(seed_ids[:dev_count])

    train_examples: list[TrainingExample] = []
    dev_examples: list[TrainingExample] = []

    for sid in seed_ids:
        if sid in dev_seed_ids:
            dev_examples.extend(examples_by_seed[sid])
        else:
            train_examples.extend(examples_by_seed[sid])

    rng.shuffle(train_examples)
    rng.shuffle(dev_examples)

    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"
    save_training_data(train_examples, train_path)
    save_training_data(dev_examples, dev_path)

    stats = {
        "train_count": len(train_examples),
        "dev_count": len(dev_examples),
        "seeds": len(seed_ids),
        "train_seeds": len(seed_ids) - dev_count,
        "dev_seeds": dev_count,
    }

    logger.info(
        "Merge complete: %d train, %d dev (%d seeds: %d train, %d dev)",
        stats["train_count"],
        stats["dev_count"],
        stats["seeds"],
        stats["train_seeds"],
        stats["dev_seeds"],
    )

    return stats


def backfill_hard_negatives(
    output_dir: Path,
    seeds: list[Paper],
    max_negatives: int = 5,
    show_progress: bool = True,
) -> dict:
    """Second pass: backfill hard negatives using the complete co-citation map.

    Early seeds may have sparse hard negatives because the co-citation map
    was incomplete when they were processed. This pass loads the final map
    and enriches all examples that have fewer than max_negatives.

    Args:
        output_dir: Directory containing seed_*.jsonl and state.json.
        seeds: Original list of seed papers (for computing embedding texts).
        max_negatives: Target number of hard negatives per example.
        show_progress: Whether to show progress.

    Returns:
        Stats dict with files_processed, enriched_count, negatives_added.
    """
    state_path = output_dir / STATE_FILENAME
    state = _load_state(state_path)

    co_citation_raw = state.get("co_citation_map", {})
    co_citation_map: dict[str, set[str]] = {k: set(v) for k, v in co_citation_raw.items()}

    # Build seed text lookup (canonical format)
    seed_texts: dict[str, str] = {}
    for s in seeds:
        seed_texts[s.id] = format_paper_embedding_text(
            title=s.title,
            abstract=s.abstract,
            author_lastnames=s.author_lastnames,
            year=s.year,
            journal=s.journal,
            include_abstract=True,
            include_metadata=True,
        )

    seed_files = sorted(output_dir.glob("seed_*.jsonl"))
    stats = {"files_processed": 0, "enriched_count": 0, "negatives_added": 0}

    for path in tqdm(seed_files, desc="Backfilling negatives", disable=not show_progress):
        examples: list[TrainingExample] = []
        modified = False

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                examples.append(TrainingExample.from_dict(json.loads(line)))

        for ex in examples:
            if len(ex.hard_negatives) >= max_negatives:
                continue

            citing_id = ex.source_paper_id
            seed_id = ex.cited_paper_id
            positive_core = strip_metadata_prefix(ex.positive)

            co_cited = co_citation_map.get(citing_id, set())
            other_seeds = [sid for sid in co_cited if sid != seed_id]

            existing_cores = {strip_metadata_prefix(n) for n in ex.hard_negatives}

            new_negs: list[str] = []
            rng = random.Random(hash(f"backfill_{seed_id}_{citing_id}"))
            if other_seeds:
                shuffled = list(other_seeds)
                rng.shuffle(shuffled)
                for neg_id in shuffled:
                    neg_text = seed_texts.get(neg_id, "")
                    if not neg_text:
                        continue
                    neg_core = strip_metadata_prefix(neg_text)
                    if neg_core == positive_core or neg_core in existing_cores:
                        continue
                    new_negs.append(neg_text)
                    existing_cores.add(neg_core)
                    if len(ex.hard_negatives) + len(new_negs) >= max_negatives:
                        break

            if new_negs:
                ex.hard_negatives = list(ex.hard_negatives) + new_negs
                stats["enriched_count"] += 1
                stats["negatives_added"] += len(new_negs)
                modified = True

        if modified:
            save_training_data(examples, path)

        stats["files_processed"] += 1

    logger.info(
        "Backfill complete: enriched %d examples with %d new negatives across %d files",
        stats["enriched_count"],
        stats["negatives_added"],
        stats["files_processed"],
    )

    return stats


def _load_state(state_path: Path) -> dict:
    """Load resumable processing state from JSON file."""
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {
        "completed_seeds": [],
        "total_examples": 0,
        "co_citation_map": {},
        "seed_stats": {},
    }


def _save_state(state_path: Path, state: dict) -> None:
    """Save processing state to JSON file for resumability.

    Uses atomic write (write to .tmp then rename) to prevent corruption
    from interruptions.
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    tmp_path.rename(state_path)
