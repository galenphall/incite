"""Mine real citation contexts from S2ORC v2 full-text papers.

S2ORC v2 stores inline citation references as `bibref` annotations (with
`attributes.ref_id`) that link to `bibentry` annotations (with
`attributes.matched_paper_id` for S2 corpus IDs and `attributes.id` for
local ref IDs).  Each bibref can be mapped to its containing `paragraph`
annotation to extract narrow/broad context windows around the citation.

The resulting training examples use *real* author-written citation contexts
(not LLM-generated) paired with co-reference-set hard negatives, producing
the highest-quality contrastive learning signal.

Usage (via scripts/mine_s2orc_citations.py):
    python scripts/mine_s2orc_citations.py \\
        --data data/raw/s2orc/s2orc_sample.jsonl \\
        --output-dir data/finetuning/s2orc_citation \\
        --target 2000 --max-negatives 5
"""

import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Iterator, Optional

import requests
from tqdm import tqdm

from incite.corpus.openalex import OpenAlexClient
from incite.finetuning.data_preparation import TrainingExample
from incite.finetuning.fulltext_passages import (
    _clean_fulltext,
    _parse_s2orc_annotation,
    _safe_span,
)
from incite.models import Paper, clean_citation_markers

logger = logging.getLogger(__name__)

# Sections to skip (references, acknowledgments, etc.)
_SKIP_SECTIONS = frozenset(
    {
        "references",
        "bibliography",
        "works cited",
        "references and notes",
        "acknowledgments",
        "acknowledgements",
        "acknowledgment",
        "acknowledgement",
        "appendix",
        "supplementary material",
        "funding",
    }
)


def _is_skip_section(section: str) -> bool:
    return section.strip().lower() in _SKIP_SECTIONS


def _split_sentences_regex(text: str) -> list[tuple[int, int, str]]:
    """Regex-based sentence splitting with character offsets.

    Reuses the same approach as UnarXivProcessor._split_into_sentences_regex.
    """
    pattern = r"(?<=[.!?])\s+(?=[A-Z])"
    sentences = []
    pos = 0
    parts = re.split(pattern, text)
    for part in parts:
        part_stripped = part.strip()
        if part_stripped:
            start = text.find(part, pos)
            if start == -1:
                start = pos
            end = start + len(part)
            sentences.append((start, end, part_stripped))
            pos = end
    return sentences


def _get_sentence_windows(
    para_text: str,
    cite_offset: int,
    narrow_before: int = 2,
    broad_before: int = 5,
    forward_n: int = 1,
) -> tuple[str, str]:
    """Extract narrow and broad sentence windows around a citation offset.

    Reuses logic from UnarXivProcessor._get_sentence_windows.
    """
    sentences = _split_sentences_regex(para_text)
    if not sentences:
        return para_text.strip(), para_text.strip()

    # Find sentence containing the citation
    cite_idx = len(sentences) - 1
    for i, (start, end, _) in enumerate(sentences):
        if start <= cite_offset < end:
            cite_idx = i
            break

    # Build windows
    narrow_start = max(0, cite_idx - narrow_before)
    broad_start = max(0, cite_idx - broad_before)
    end_idx = min(len(sentences), cite_idx + 1 + forward_n)

    narrow = " ".join(s[2] for s in sentences[narrow_start:end_idx])
    broad = " ".join(s[2] for s in sentences[broad_start:end_idx])
    return narrow.strip(), broad.strip()


class S2CorpusIDResolver:
    """Resolve S2 corpus IDs to OpenAlex IDs via Semantic Scholar API.

    Caches results to disk to avoid redundant API calls.
    """

    S2_API = "https://api.semanticscholar.org/graph/v1"

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        api_key: Optional[str] = None,
        delay: float = 0.35,
    ):
        self.cache: dict[int, Optional[str]] = {}  # corpus_id -> openalex_id
        self.cache_path = cache_path
        self.api_key = api_key
        self.delay = delay
        self._last_request = 0.0

        if cache_path and cache_path.exists():
            self._load_cache()

    def _load_cache(self) -> None:
        with open(self.cache_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    self.cache[entry["corpus_id"]] = entry.get("openalex_id")

    def _save_entry(self, corpus_id: int, openalex_id: Optional[str]) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "corpus_id": corpus_id,
                            "openalex_id": openalex_id,
                        }
                    )
                    + "\n"
                )

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.time()

    def resolve_batch(
        self,
        corpus_ids: list[int],
        show_progress: bool = True,
        batch_size: int = 500,
    ) -> dict[int, str]:
        """Resolve S2 corpus IDs to OpenAlex IDs using S2 batch API.

        Uses POST /paper/batch (up to 500 IDs per request) for fast resolution.
        Returns dict of corpus_id -> openalex_id (only for successful lookups).
        """
        to_fetch = [cid for cid in corpus_ids if cid not in self.cache]
        result: dict[int, str] = {}

        # Return cached results
        for cid in corpus_ids:
            if cid in self.cache and self.cache[cid]:
                result[cid] = self.cache[cid]

        if not to_fetch:
            return result

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        # Process in batches of batch_size using POST /paper/batch
        n_batches = (len(to_fetch) + batch_size - 1) // batch_size
        batch_iter = range(n_batches)
        if show_progress:
            batch_iter = tqdm(
                batch_iter,
                desc=f"Resolving S2 IDs ({len(to_fetch):,} in {n_batches} batches)",
                total=n_batches,
            )

        for batch_idx in batch_iter:
            start = batch_idx * batch_size
            end = min(start + batch_size, len(to_fetch))
            batch_cids = to_fetch[start:end]

            # Format as CorpusId:{id} for the batch API
            paper_ids = [f"CorpusId:{cid}" for cid in batch_cids]

            self._rate_limit()
            try:
                resp = requests.post(
                    f"{self.S2_API}/paper/batch",
                    params={"fields": "externalIds,corpusId"},
                    headers=headers,
                    json={"ids": paper_ids},
                    timeout=30,
                )
                if resp.status_code == 200:
                    results_list = resp.json()
                    for i, paper_data in enumerate(results_list):
                        cid = batch_cids[i]
                        if paper_data is None:
                            # Paper not found
                            self.cache[cid] = None
                            self._save_entry(cid, None)
                            continue
                        ext_ids = paper_data.get("externalIds", {})
                        oa_id = ext_ids.get("OpenAlex")
                        if oa_id:
                            self.cache[cid] = oa_id
                            self._save_entry(cid, oa_id)
                            result[cid] = oa_id
                        else:
                            self.cache[cid] = None
                            self._save_entry(cid, None)
                elif resp.status_code == 429:
                    # Rate limited, back off and retry individually
                    logger.warning("S2 batch API rate limited, backing off...")
                    time.sleep(5.0)
                    self.delay = min(self.delay * 2, 5.0)
                    # Fall back to individual requests for this batch
                    for cid in batch_cids:
                        self._resolve_single(cid, headers, result)
                else:
                    logger.warning(
                        "S2 batch API %d for batch %d",
                        resp.status_code,
                        batch_idx,
                    )
            except requests.RequestException as e:
                logger.warning("S2 batch API error for batch %d: %s", batch_idx, e)
                # Fall back to individual requests
                for cid in batch_cids:
                    self._resolve_single(cid, headers, result)

        return result

    def _resolve_single(
        self,
        cid: int,
        headers: dict,
        result: dict[int, str],
    ) -> None:
        """Resolve a single S2 corpus ID (fallback for batch failures)."""
        if cid in self.cache:
            if self.cache[cid]:
                result[cid] = self.cache[cid]
            return
        self._rate_limit()
        try:
            resp = requests.get(
                f"{self.S2_API}/paper/CorpusId:{cid}",
                params={"fields": "externalIds"},
                headers=headers,
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                ext_ids = data.get("externalIds", {})
                oa_id = ext_ids.get("OpenAlex")
                if oa_id:
                    self.cache[cid] = oa_id
                    self._save_entry(cid, oa_id)
                    result[cid] = oa_id
                else:
                    self.cache[cid] = None
                    self._save_entry(cid, None)
            elif resp.status_code == 404:
                self.cache[cid] = None
                self._save_entry(cid, None)
            elif resp.status_code == 429:
                time.sleep(2.0)
        except requests.RequestException:
            pass

    def resolve_batch_to_papers(
        self,
        corpus_ids: list[int],
        cache_path: Optional[Path] = None,
        show_progress: bool = True,
        batch_size: int = 500,
    ) -> dict[int, "Paper"]:
        """Fetch paper metadata directly from S2 batch API.

        Bypasses OpenAlex entirely — most S2 papers don't have OpenAlex IDs.
        Uses POST /paper/batch with title, abstract, authors, year fields.

        Returns dict of corpus_id -> Paper (only for papers with abstracts).
        """
        # Load existing paper cache
        papers: dict[int, Paper] = {}
        cached_cids: set[int] = set()
        if cache_path and cache_path.exists():
            with open(cache_path) as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        cid = d.get("corpus_id")
                        if cid is not None:
                            cached_cids.add(cid)
                            if d.get("abstract"):
                                p = Paper(
                                    id=f"s2:{cid}",
                                    title=d.get("title", ""),
                                    abstract=d.get("abstract", ""),
                                    authors=d.get("authors", []),
                                    year=d.get("year"),
                                    doi=d.get("doi"),
                                    journal=d.get("journal"),
                                )
                                papers[cid] = p

        to_fetch = [cid for cid in corpus_ids if cid not in cached_cids]

        if not to_fetch:
            logger.info("All %d corpus IDs found in cache", len(corpus_ids))
            return papers

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        n_batches = (len(to_fetch) + batch_size - 1) // batch_size
        batch_iter = range(n_batches)
        if show_progress:
            batch_iter = tqdm(
                batch_iter,
                desc=f"Fetching S2 metadata ({len(to_fetch):,} in {n_batches} batches)",
                total=n_batches,
            )

        new_entries: list[dict] = []
        for batch_idx in batch_iter:
            start = batch_idx * batch_size
            end = min(start + batch_size, len(to_fetch))
            batch_cids = to_fetch[start:end]

            paper_ids = [f"CorpusId:{cid}" for cid in batch_cids]

            self._rate_limit()
            try:
                resp = requests.post(
                    f"{self.S2_API}/paper/batch",
                    params={"fields": "title,abstract,authors,year,externalIds,corpusId"},
                    headers=headers,
                    json={"ids": paper_ids},
                    timeout=60,
                )
                if resp.status_code == 200:
                    results_list = resp.json()
                    for i, paper_data in enumerate(results_list):
                        cid = batch_cids[i]
                        if paper_data is None:
                            new_entries.append({"corpus_id": cid, "title": None, "abstract": None})
                            continue

                        title = paper_data.get("title", "") or ""
                        abstract = paper_data.get("abstract", "") or ""
                        authors_raw = paper_data.get("authors", []) or []
                        author_names = [a.get("name", "") for a in authors_raw if a.get("name")]
                        year = paper_data.get("year")
                        ext_ids = paper_data.get("externalIds", {}) or {}
                        doi = ext_ids.get("DOI")

                        entry = {
                            "corpus_id": cid,
                            "title": title,
                            "abstract": abstract,
                            "authors": author_names,
                            "year": year,
                            "doi": doi,
                        }
                        new_entries.append(entry)

                        if abstract and title:
                            papers[cid] = Paper(
                                id=f"s2:{cid}",
                                title=title,
                                abstract=abstract,
                                authors=author_names,
                                year=year,
                                doi=doi,
                            )
                elif resp.status_code == 429:
                    logger.warning("S2 batch API rate limited, backing off 10s...")
                    time.sleep(10.0)
                    self.delay = min(self.delay * 2, 5.0)
                else:
                    logger.warning("S2 batch API %d for batch %d", resp.status_code, batch_idx)
            except requests.RequestException as e:
                logger.warning("S2 batch API error for batch %d: %s", batch_idx, e)

        # Append new entries to cache
        if cache_path and new_entries:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "a") as f:
                for entry in new_entries:
                    f.write(json.dumps(entry) + "\n")

        return papers


class S2ORCCitationMiner:
    """Extract real citation contexts from S2ORC v2 papers.

    Parses bibref/bibentry annotations, resolves to OpenAlex papers,
    and creates TrainingExamples with co-reference-set hard negatives.
    """

    def __init__(
        self,
        openalex_client: Optional[OpenAlexClient] = None,
        s2_resolver: Optional[S2CorpusIDResolver] = None,
        min_context_length: int = 50,
        max_hard_negatives: int = 5,
    ):
        self.openalex_client = openalex_client or OpenAlexClient()
        self.s2_resolver = s2_resolver or S2CorpusIDResolver()
        self.min_context_length = min_context_length
        self.max_hard_negatives = max_hard_negatives

    def iter_s2orc_papers(self, jsonl_path: Path) -> Iterator[dict]:
        """Iterate over S2ORC v2 records from a JSONL file."""
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "content" in data and "corpusid" in data:
                    yield data

    def _extract_bibref_contexts(self, data: dict) -> list[dict]:
        """Extract citation contexts from S2ORC v2 bibref annotations.

        Returns list of dicts with keys:
            context_text, narrow, broad, section, corpus_id, ref_id
        """
        content = data.get("content", {})
        text = content.get("text", "")
        annotations = content.get("annotations", {})
        if not text or not annotations:
            return []

        # Parse annotation types
        bibrefs = _parse_s2orc_annotation(annotations, "bibref")
        bibentries = _parse_s2orc_annotation(annotations, "bibentry")
        paragraphs = _parse_s2orc_annotation(annotations, "paragraph")
        section_headers = _parse_s2orc_annotation(annotations, "sectionheader")

        if not bibrefs or not bibentries:
            return []

        # Build bibentry lookup: local ref_id -> (matched_paper_id, bibentry_text)
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

        # Build paragraph lookup for quick containment checks
        para_spans = [(int(p["start"]), int(p["end"])) for p in paragraphs]

        # Build section header lookup
        sec_idx = 0
        section_for_para: dict[int, str] = {}  # para_start -> section name
        for ps, pe in para_spans:
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

            # Get bibref offset in full text
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

            # Skip reference/acknowledgment sections
            if _is_skip_section(section):
                continue

            # Convert absolute offset to paragraph-relative
            cite_offset_in_para = ref_start - para_offset

            # Clean the paragraph text
            cleaned_para = _clean_fulltext(para_text)
            if len(cleaned_para) < self.min_context_length:
                continue

            # Get citation marker text for replacement
            cite_marker = text[ref_start:ref_end]

            # Extract sentence windows
            narrow, broad = _get_sentence_windows(para_text, cite_offset_in_para)

            # Replace citation marker with [CITE]
            if cite_marker:
                narrow = narrow.replace(cite_marker, "[CITE]", 1)
                broad = broad.replace(cite_marker, "[CITE]", 1)

            # Clean the contexts
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

    def mine(
        self,
        data_dir: Path,
        output_dir: Path,
        target_papers: int = 2000,
        min_refs: int = 15,
        dev_fraction: float = 0.2,
        seed: int = 42,
        show_progress: bool = True,
        openalex_email: Optional[str] = None,
        dry_run: bool = False,
    ) -> dict:
        """Mine citation training data from S2ORC v2 papers.

        Args:
            data_dir: Directory with S2ORC JSONL files
            output_dir: Where to write train.jsonl and dev.jsonl
            target_papers: Stop after this many qualifying source papers
            min_refs: Minimum resolved references per paper
            dev_fraction: Fraction of source papers for dev set
            seed: Random seed
            show_progress: Show progress bars
            openalex_email: Email for OpenAlex polite pool
            dry_run: Show stats without writing

        Returns:
            Stats dict
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if openalex_email:
            self.openalex_client = OpenAlexClient(email=openalex_email)

        # Set up S2 resolver cache
        if self.s2_resolver.cache_path is None:
            self.s2_resolver.cache_path = output_dir / "s2_to_openalex_cache.jsonl"
            if self.s2_resolver.cache_path.exists():
                self.s2_resolver._load_cache()

        # Find JSONL files
        jsonl_files = sorted(data_dir.glob("**/*.jsonl"))
        if not jsonl_files:
            print(f"No JSONL files found in {data_dir}")
            return {"error": "No JSONL files found"}
        print(f"Found {len(jsonl_files)} JSONL files")

        # Phase 1: Scan papers and extract citation contexts
        print("\nPhase 1: Scanning for citation contexts...")
        qualifying_papers: list[tuple[str, list[dict]]] = []  # (paper_id, contexts)
        all_corpus_ids: set[int] = set()
        scanned = 0
        skipped = 0

        pbar = tqdm(jsonl_files, desc="Scanning files") if show_progress else jsonl_files
        for jsonl_path in pbar:
            for record in self.iter_s2orc_papers(jsonl_path):
                scanned += 1
                paper_id = str(record.get("corpusid", ""))

                contexts = self._extract_bibref_contexts(record)
                if not contexts:
                    skipped += 1
                    continue

                # Collect all corpus IDs for this paper
                paper_corpus_ids = {ctx["corpus_id"] for ctx in contexts}
                if len(paper_corpus_ids) < min_refs:
                    skipped += 1
                    continue

                qualifying_papers.append((paper_id, contexts))
                all_corpus_ids.update(paper_corpus_ids)

                if show_progress and hasattr(pbar, "set_postfix"):
                    pbar.set_postfix({"sources": len(qualifying_papers)})

                if len(qualifying_papers) >= target_papers:
                    break

            if len(qualifying_papers) >= target_papers:
                break

        print(
            f"Found {len(qualifying_papers)} qualifying papers "
            f"(scanned {scanned}, skipped {skipped})"
        )
        print(f"Unique corpus IDs to resolve: {len(all_corpus_ids)}")

        if not qualifying_papers:
            return {"error": "No qualifying papers found", "scanned": scanned}

        if dry_run:
            total_contexts = sum(len(ctxs) for _, ctxs in qualifying_papers)
            print(
                f"\nDry run: {len(qualifying_papers)} papers, {total_contexts} contexts, "
                f"{len(all_corpus_ids)} corpus IDs to resolve"
            )
            return {
                "dry_run": True,
                "qualifying_papers": len(qualifying_papers),
                "total_contexts": total_contexts,
                "unique_corpus_ids": len(all_corpus_ids),
            }

        # Phase 2: Fetch paper metadata directly from S2 API
        print("\nPhase 2: Fetching paper metadata from Semantic Scholar...")
        s2_cache_path = output_dir / "s2_papers_cache.jsonl"
        papers_by_cid: dict[int, Paper] = self.s2_resolver.resolve_batch_to_papers(
            list(all_corpus_ids),
            cache_path=s2_cache_path,
            show_progress=show_progress,
        )
        print(f"Got metadata for {len(papers_by_cid)}/{len(all_corpus_ids)} corpus IDs")

        # Phase 3: Build training examples
        print("\nPhase 3: Building training examples...")
        examples_by_source: dict[str, list[TrainingExample]] = {}
        stats = {
            "scanned": scanned,
            "qualifying_papers": len(qualifying_papers),
            "contexts_total": 0,
            "contexts_skipped_no_metadata": 0,
            "examples_created": 0,
        }

        for paper_id, contexts in (
            tqdm(qualifying_papers, desc="Building examples")
            if show_progress
            else qualifying_papers
        ):
            # Build reference set: corpus IDs with available metadata
            resolved_cids: set[int] = set()
            for ctx in contexts:
                cid = ctx["corpus_id"]
                if cid in papers_by_cid:
                    resolved_cids.add(cid)

            if len(resolved_cids) < min_refs:
                continue

            # Build embedding texts for reference set
            ref_texts: dict[int, str] = {}
            for cid in resolved_cids:
                ref_texts[cid] = papers_by_cid[cid].to_embedding_text(
                    include_abstract=True, include_metadata=True
                )

            source_examples = []
            for ctx in contexts:
                stats["contexts_total"] += 1
                cid = ctx["corpus_id"]

                if cid not in papers_by_cid:
                    stats["contexts_skipped_no_metadata"] += 1
                    continue

                # Build hard negatives from co-reference set
                other_cids = [c for c in resolved_cids if c != cid]
                rng = random.Random(hash(f"{paper_id}_{cid}_{stats['examples_created']}"))
                neg_cids = rng.sample(other_cids, min(self.max_hard_negatives, len(other_cids)))
                hard_negatives = [ref_texts[nc] for nc in neg_cids]

                example = TrainingExample(
                    query=ctx["narrow"],
                    positive=ref_texts[cid],
                    hard_negatives=hard_negatives,
                    source_paper_id=paper_id,
                    cited_paper_id=f"s2:{cid}",
                    source="s2orc_citation",
                )
                source_examples.append(example)
                stats["examples_created"] += 1

            if source_examples:
                examples_by_source[paper_id] = source_examples

        print(
            f"Created {stats['examples_created']} examples from "
            f"{len(examples_by_source)} source papers"
        )

        # Phase 4: Split and save
        print("\nPhase 4: Splitting into train/dev...")
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

        train_path = output_dir / "train.jsonl"
        dev_path = output_dir / "dev.jsonl"

        with open(train_path, "w") as f:
            for ex in train_examples:
                f.write(json.dumps(ex.to_dict()) + "\n")

        with open(dev_path, "w") as f:
            for ex in dev_examples:
                f.write(json.dumps(ex.to_dict()) + "\n")

        stats["train_examples"] = len(train_examples)
        stats["dev_examples"] = len(dev_examples)
        stats["train_source_papers"] = len(source_ids) - dev_count
        stats["dev_source_papers"] = dev_count

        print("\nMining complete:")
        print(f"  Papers scanned:           {stats['scanned']}")
        print(f"  Qualifying papers:        {stats['qualifying_papers']}")
        print(f"  Contexts total:           {stats['contexts_total']}")
        print(f"  Skipped (no metadata):    {stats['contexts_skipped_no_metadata']}")
        print(f"  Examples created:         {stats['examples_created']}")
        print(f"  Train: {len(train_examples)} ({stats['train_source_papers']} source papers)")
        print(f"  Dev:   {len(dev_examples)} ({stats['dev_source_papers']} source papers)")
        print(f"\nOutput: {train_path}, {dev_path}")

        return stats
