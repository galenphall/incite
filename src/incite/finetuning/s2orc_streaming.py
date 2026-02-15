"""Stream S2ORC shards and extract citation training data without local storage.

Processes gzip-compressed JSONL from the S2 Datasets API shard by shard,
extracting real citation contexts and producing TrainingExamples with
co-citation hard negatives. Each shard is processed independently and
written to its own output file, with a resumable state file tracking
progress. Final merge combines per-shard files into train/dev splits.

Usage (via scripts/stream_s2orc_citations.py):
    python scripts/stream_s2orc_citations.py \\
        --target 50000 --shards 5 \\
        --output-dir data/finetuning/s2orc_citation_v2
"""

import gzip
import json
import logging
import random
import time
from pathlib import Path
from typing import Iterator, Optional

import requests
from tqdm import tqdm

from incite.finetuning.fulltext_passages import (
    _clean_fulltext,
    _parse_s2orc_annotation,
    _safe_span,
)
from incite.finetuning.s2orc_citation_mining import (
    S2CorpusIDResolver,
    _get_sentence_windows,
    _is_skip_section,
)
from incite.finetuning.types import TrainingExample, save_training_data
from incite.models import Paper, clean_citation_markers, format_paper_embedding_text

logger = logging.getLogger(__name__)

DATASETS_API = "https://api.semanticscholar.org/datasets/v1"
STATE_FILENAME = "shard_state.json"

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 5.0  # seconds


def get_s2orc_shard_urls(api_key: Optional[str] = None) -> tuple[str, list[str]]:
    """Fetch the latest S2ORC release shard URLs from the Datasets API.

    Args:
        api_key: Semantic Scholar API key for higher rate limits.

    Returns:
        Tuple of (release_id, list of shard download URLs).

    Raises:
        requests.RequestException: On network or API errors.
    """
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    resp = requests.get(f"{DATASETS_API}/release/latest", headers=headers, timeout=30)
    resp.raise_for_status()
    release_id = resp.json()["release_id"]

    resp = requests.get(
        f"{DATASETS_API}/release/{release_id}/dataset/s2orc",
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    urls = resp.json().get("files", [])

    return release_id, urls


def _stream_shard_records(shard_url: str, api_key: Optional[str] = None) -> "Iterator[dict]":
    """Stream and decompress a single S2ORC shard, yielding parsed records.

    Streams the gzip-compressed JSONL over HTTP without writing to disk.
    Retries on transient network failures with exponential backoff.

    Args:
        shard_url: Download URL for the shard.
        api_key: S2 API key for authentication.

    Yields:
        Parsed JSON dicts for each record in the shard.
    """
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(shard_url, headers=headers, stream=True, timeout=120)
            resp.raise_for_status()
            decompressor = gzip.GzipFile(fileobj=resp.raw)

            for raw_line in decompressor:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "content" in record and "corpusid" in record:
                    yield record

            return  # Success, exit retry loop

        except (
            requests.RequestException,
            gzip.BadGzipFile,
            OSError,
        ) as e:
            wait = RETRY_BACKOFF_BASE * (2**attempt)
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    "Shard download attempt %d failed (%s), retrying in %.0fs",
                    attempt + 1,
                    e,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Shard download failed after %d attempts: %s",
                    MAX_RETRIES,
                    e,
                )
                raise


class S2ORCShardProcessor:
    """Processes S2ORC shards via streaming to extract citation training data.

    Streams gzip-compressed JSONL from S2 Datasets API URLs, extracts
    citation contexts using bibref/bibentry annotations, resolves cited
    papers via S2 batch API, and produces TrainingExamples with co-citation
    hard negatives.

    Per-shard output is written to separate files with a resumable state
    file tracking which shards have been processed. Final merge combines
    per-shard output into train/dev splits.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        min_context_length: int = 50,
        min_refs: int = 15,
        max_hard_negatives: int = 5,
        s2_resolver: Optional[S2CorpusIDResolver] = None,
    ):
        """Initialize the shard processor.

        Args:
            api_key: Semantic Scholar API key.
            min_context_length: Minimum character length for citation contexts.
            min_refs: Minimum resolved references per source paper.
            max_hard_negatives: Maximum co-citation hard negatives per example.
            s2_resolver: Pre-configured S2 corpus ID resolver (created if None).
        """
        self.api_key = api_key
        self.min_context_length = min_context_length
        self.min_refs = min_refs
        self.max_hard_negatives = max_hard_negatives
        self.s2_resolver = s2_resolver or S2CorpusIDResolver(api_key=api_key)

    def _extract_bibref_contexts(self, data: dict) -> list[dict]:
        """Extract citation contexts from S2ORC v2 bibref annotations.

        Parses bibref and bibentry annotations, finds the containing
        paragraph, extracts narrow/broad sentence windows, and cleans
        citation markers. Skips references and acknowledgment sections.

        Args:
            data: A single S2ORC v2 record dict.

        Returns:
            List of context dicts with keys: narrow, broad, section,
            corpus_id, ref_id.
        """
        content = data.get("content", {})
        text = content.get("text", "")
        annotations = content.get("annotations", {})
        if not text or not annotations:
            return []

        bibrefs = _parse_s2orc_annotation(annotations, "bibref")
        bibentries = _parse_s2orc_annotation(annotations, "bibentry")
        paragraphs = _parse_s2orc_annotation(annotations, "paragraph")
        section_headers = _parse_s2orc_annotation(annotations, "sectionheader")

        if not bibrefs or not bibentries:
            return []

        # Build bibentry lookup: local ref_id -> (matched_paper_id, text)
        bibentry_map: dict[str, tuple[Optional[int], str]] = {}
        for entry in bibentries:
            attrs = entry.get("attributes", {})
            local_id = attrs.get("id", "")
            matched_id = attrs.get("matched_paper_id")
            entry_text = _safe_span(text, entry)
            if local_id:
                try:
                    matched_int = int(matched_id) if matched_id is not None else None
                except (ValueError, TypeError):
                    matched_int = None
                bibentry_map[local_id] = (matched_int, entry_text)

        # Sort paragraphs and section headers by offset
        paragraphs = sorted(
            (p for p in paragraphs if isinstance(p.get("start"), (int, float))),
            key=lambda p: int(p["start"]),
        )
        section_headers = sorted(
            (s for s in section_headers if isinstance(s.get("start"), (int, float))),
            key=lambda s: int(s["start"]),
        )

        para_spans = [(int(p["start"]), int(p["end"])) for p in paragraphs]

        # Map paragraph start offsets to section names
        sec_idx = 0
        section_for_para: dict[int, str] = {}
        for ps, _pe in para_spans:
            while sec_idx < len(section_headers) and int(section_headers[sec_idx]["start"]) <= ps:
                sec_idx += 1
            if sec_idx > 0:
                section_for_para[ps] = _safe_span(text, section_headers[sec_idx - 1])

        contexts = []
        for bibref in bibrefs:
            attrs = bibref.get("attributes", {})
            ref_id = attrs.get("ref_id", "")
            if not ref_id or ref_id not in bibentry_map:
                continue

            corpus_id, _ = bibentry_map[ref_id]
            if corpus_id is None:
                continue

            try:
                ref_start = int(bibref["start"])
                ref_end = int(bibref["end"])
            except (KeyError, ValueError, TypeError):
                continue

            # Find containing paragraph
            para_text = None
            para_offset = 0
            section = ""
            for ps, pe in para_spans:
                if ps <= ref_start < pe:
                    para_text = text[ps:pe].strip()
                    para_offset = ps
                    section = section_for_para.get(ps, "")
                    break

            if not para_text:
                continue

            if _is_skip_section(section):
                continue

            cite_offset_in_para = ref_start - para_offset

            cleaned_para = _clean_fulltext(para_text)
            if len(cleaned_para) < self.min_context_length:
                continue

            cite_marker = text[ref_start:ref_end]

            narrow, broad = _get_sentence_windows(para_text, cite_offset_in_para)

            if cite_marker:
                narrow = narrow.replace(cite_marker, "[CITE]", 1)
                broad = broad.replace(cite_marker, "[CITE]", 1)

            narrow_clean = clean_citation_markers(narrow)
            broad_clean = clean_citation_markers(broad)

            if len(narrow_clean) < self.min_context_length:
                continue

            contexts.append(
                {
                    "narrow": narrow_clean,
                    "broad": broad_clean,
                    "section": section,
                    "corpus_id": corpus_id,
                    "ref_id": ref_id,
                }
            )

        return contexts

    def process_shard(
        self,
        shard_url: str,
        output_path: Path,
        target: int = 10000,
        show_progress: bool = True,
    ) -> dict:
        """Process one S2ORC shard and write training examples to output_path.

        Streams the shard, extracts citation contexts, resolves cited paper
        metadata via the S2 batch API, and builds TrainingExamples with
        co-citation hard negatives.

        Args:
            shard_url: Download URL for the shard.
            output_path: Path for per-shard JSONL output.
            target: Target number of training examples from this shard.
            show_progress: Whether to display progress bars.

        Returns:
            Stats dict with keys: scanned, qualifying_papers, contexts_total,
            contexts_skipped_no_metadata, examples_created.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        shard_name = shard_url.split("/")[-1]
        logger.info("Processing shard: %s (target: %d examples)", shard_name, target)

        # Phase 1: Stream shard and collect citation contexts + corpus IDs
        qualifying_papers: list[tuple[str, list[dict]]] = []
        all_corpus_ids: set[int] = set()
        scanned = 0
        skipped = 0

        desc = f"Scanning {shard_name}"
        stream = _stream_shard_records(shard_url, self.api_key)

        for record in tqdm(stream, desc=desc, disable=not show_progress):
            scanned += 1
            paper_id = str(record.get("corpusid", ""))

            contexts = self._extract_bibref_contexts(record)
            if not contexts:
                skipped += 1
                continue

            paper_corpus_ids = {ctx["corpus_id"] for ctx in contexts}
            if len(paper_corpus_ids) < self.min_refs:
                skipped += 1
                continue

            qualifying_papers.append((paper_id, contexts))
            all_corpus_ids.update(paper_corpus_ids)

            # Rough estimate: ~10 examples per qualifying paper
            estimated_examples = sum(len(ctxs) for _, ctxs in qualifying_papers)
            if estimated_examples >= target * 1.2:
                break

        logger.info(
            "Phase 1 complete: %d qualifying papers from %d scanned (%d skipped). "
            "%d unique corpus IDs.",
            len(qualifying_papers),
            scanned,
            skipped,
            len(all_corpus_ids),
        )

        if not qualifying_papers:
            return {
                "scanned": scanned,
                "qualifying_papers": 0,
                "contexts_total": 0,
                "contexts_skipped_no_metadata": 0,
                "examples_created": 0,
            }

        # Phase 2: Resolve corpus IDs to Paper metadata via S2 batch API
        s2_cache_path = output_path.parent / "s2_papers_cache.jsonl"
        papers_by_cid: dict[int, Paper] = self.s2_resolver.resolve_batch_to_papers(
            list(all_corpus_ids),
            cache_path=s2_cache_path,
            show_progress=show_progress,
        )
        logger.info(
            "Phase 2 complete: metadata for %d/%d corpus IDs",
            len(papers_by_cid),
            len(all_corpus_ids),
        )

        # Phase 3: Build training examples
        examples: list[TrainingExample] = []
        stats = {
            "scanned": scanned,
            "qualifying_papers": len(qualifying_papers),
            "contexts_total": 0,
            "contexts_skipped_no_metadata": 0,
            "examples_created": 0,
        }

        for paper_id, contexts in qualifying_papers:
            resolved_cids: set[int] = set()
            for ctx in contexts:
                if ctx["corpus_id"] in papers_by_cid:
                    resolved_cids.add(ctx["corpus_id"])

            if len(resolved_cids) < self.min_refs:
                continue

            # Pre-compute embedding texts for the reference set
            ref_texts: dict[int, str] = {}
            for cid in resolved_cids:
                paper = papers_by_cid[cid]
                ref_texts[cid] = format_paper_embedding_text(
                    title=paper.title,
                    abstract=paper.abstract,
                    author_lastnames=paper.author_lastnames,
                    year=paper.year,
                    journal=paper.journal,
                    include_abstract=True,
                    include_metadata=True,
                )

            for ctx in contexts:
                stats["contexts_total"] += 1
                cid = ctx["corpus_id"]

                if cid not in papers_by_cid:
                    stats["contexts_skipped_no_metadata"] += 1
                    continue

                # Co-citation hard negatives: other papers cited in the
                # same source paper
                other_cids = [c for c in resolved_cids if c != cid]
                rng = random.Random(hash(f"{paper_id}_{cid}_{stats['examples_created']}"))
                neg_cids = rng.sample(
                    other_cids,
                    min(self.max_hard_negatives, len(other_cids)),
                )
                hard_negatives = [ref_texts[nc] for nc in neg_cids]

                example = TrainingExample(
                    query=ctx["narrow"],
                    positive=ref_texts[cid],
                    hard_negatives=hard_negatives,
                    source_paper_id=paper_id,
                    cited_paper_id=f"s2:{cid}",
                    source="s2orc_citation",
                )
                examples.append(example)
                stats["examples_created"] += 1

                if stats["examples_created"] >= target:
                    break

            if stats["examples_created"] >= target:
                break

        # Write per-shard output
        save_training_data(examples, output_path)
        logger.info(
            "Shard complete: %d examples written to %s",
            len(examples),
            output_path,
        )

        return stats

    def process_shards(
        self,
        shard_urls: list[str],
        output_dir: Path,
        target: int = 50000,
        show_progress: bool = True,
    ) -> dict:
        """Orchestrate multi-shard processing with resumable state.

        Processes shards sequentially, skipping any that were already
        completed according to the state file. Stops when the cumulative
        example count reaches the target.

        Args:
            shard_urls: List of shard download URLs to process.
            output_dir: Directory for per-shard output files and state.
            target: Total target number of training examples.
            show_progress: Whether to display progress bars.

        Returns:
            Aggregate stats dict across all processed shards.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        state_path = output_dir / STATE_FILENAME
        state = _load_state(state_path)

        total_examples = state.get("total_examples", 0)
        completed_shards = set(state.get("completed_shards", []))

        aggregate_stats = {
            "shards_processed": len(completed_shards),
            "total_scanned": state.get("total_scanned", 0),
            "total_examples": total_examples,
            "shard_stats": state.get("shard_stats", {}),
        }

        for i, url in enumerate(shard_urls):
            shard_key = url.split("/")[-1]

            if shard_key in completed_shards:
                logger.info(
                    "Skipping already-completed shard %d/%d: %s",
                    i + 1,
                    len(shard_urls),
                    shard_key,
                )
                continue

            if total_examples >= target:
                logger.info(
                    "Target reached (%d/%d examples), stopping.",
                    total_examples,
                    target,
                )
                break

            remaining = target - total_examples
            shard_target = min(remaining, target // len(shard_urls) + 1000)

            logger.info(
                "Processing shard %d/%d: %s (target: %d, total so far: %d)",
                i + 1,
                len(shard_urls),
                shard_key,
                shard_target,
                total_examples,
            )

            shard_output = output_dir / f"shard_{i:03d}.jsonl"

            try:
                shard_stats = self.process_shard(
                    shard_url=url,
                    output_path=shard_output,
                    target=shard_target,
                    show_progress=show_progress,
                )
            except Exception:
                logger.exception("Failed to process shard %s", shard_key)
                continue

            total_examples += shard_stats["examples_created"]
            aggregate_stats["shards_processed"] += 1
            aggregate_stats["total_scanned"] += shard_stats["scanned"]
            aggregate_stats["total_examples"] = total_examples
            aggregate_stats["shard_stats"][shard_key] = shard_stats

            # Update state for resumability
            completed_shards.add(shard_key)
            state["completed_shards"] = sorted(completed_shards)
            state["total_examples"] = total_examples
            state["total_scanned"] = aggregate_stats["total_scanned"]
            state["shard_stats"] = aggregate_stats["shard_stats"]
            _save_state(state_path, state)

            logger.info(
                "Shard %d/%d done: %d examples (cumulative: %d/%d)",
                i + 1,
                len(shard_urls),
                shard_stats["examples_created"],
                total_examples,
                target,
            )

        return aggregate_stats

    def merge_shards(
        self,
        output_dir: Path,
        dev_fraction: float = 0.2,
        seed: int = 42,
    ) -> dict:
        """Merge per-shard output files into final train/dev splits.

        Groups examples by source paper, then splits source papers into
        train and dev sets (no leakage between splits). Shuffles within
        each split for training stability.

        Args:
            output_dir: Directory containing shard_*.jsonl files.
            dev_fraction: Fraction of source papers allocated to dev set.
            seed: Random seed for reproducible splits.

        Returns:
            Stats dict with train_count, dev_count, source_papers.
        """
        shard_files = sorted(output_dir.glob("shard_*.jsonl"))
        if not shard_files:
            logger.warning("No shard files found in %s", output_dir)
            return {"train_count": 0, "dev_count": 0, "source_papers": 0}

        # Load all examples and group by source paper
        examples_by_source: dict[str, list[TrainingExample]] = {}
        total_loaded = 0

        for shard_path in shard_files:
            with open(shard_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ex = TrainingExample.from_dict(json.loads(line))
                    source_key = ex.source_paper_id or "unknown"
                    examples_by_source.setdefault(source_key, []).append(ex)
                    total_loaded += 1

        logger.info(
            "Loaded %d examples from %d shards (%d source papers)",
            total_loaded,
            len(shard_files),
            len(examples_by_source),
        )

        # Split by source paper (no train/dev leakage)
        rng = random.Random(seed)
        source_ids = list(examples_by_source.keys())
        rng.shuffle(source_ids)

        dev_count = max(1, int(len(source_ids) * dev_fraction))
        dev_source_ids = set(source_ids[:dev_count])

        train_examples: list[TrainingExample] = []
        dev_examples: list[TrainingExample] = []
        for sid in source_ids:
            if sid in dev_source_ids:
                dev_examples.extend(examples_by_source[sid])
            else:
                train_examples.extend(examples_by_source[sid])

        rng.shuffle(train_examples)
        rng.shuffle(dev_examples)

        train_path = output_dir / "train.jsonl"
        dev_path = output_dir / "dev.jsonl"
        save_training_data(train_examples, train_path)
        save_training_data(dev_examples, dev_path)

        stats = {
            "train_count": len(train_examples),
            "dev_count": len(dev_examples),
            "source_papers": len(source_ids),
            "train_source_papers": len(source_ids) - dev_count,
            "dev_source_papers": dev_count,
        }

        logger.info(
            "Merge complete: %d train, %d dev (%d source papers)",
            stats["train_count"],
            stats["dev_count"],
            stats["source_papers"],
        )

        return stats


def _load_state(state_path: Path) -> dict:
    """Load resumable processing state from JSON file."""
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {
        "completed_shards": [],
        "total_examples": 0,
        "total_scanned": 0,
        "shard_stats": {},
    }


def _save_state(state_path: Path, state: dict) -> None:
    """Save processing state to JSON file for resumability."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    tmp_path.rename(state_path)
