"""Data preparation for fine-tuning.

Backward-compatible module: TrainingExample and load_training_data are now
defined in types.py; mine_hard_negatives is in quality.py. This module
re-exports them for existing import sites.

Standalone preparation functions (mine_training_data, prepare_from_existing)
remain here as they are specific to the unarXiv/test-set data pipelines.
"""

import json
import random
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from incite.corpus.openalex import OpenAlexClient
from incite.corpus.unarxiv import UnarXivProcessor

# Backward compat re-exports — existing code imports these from here
from incite.finetuning.quality import mine_hard_negatives  # noqa: F401
from incite.finetuning.types import TrainingExample, load_training_data  # noqa: F401
from incite.models import Paper, clean_citation_markers


def _paper_to_embedding_text(paper: Paper) -> str:
    """Format paper as embedding text matching Paper.to_embedding_text() default."""
    return paper.to_embedding_text(include_abstract=True, include_metadata=True)


def _load_test_set_source_papers(test_set_path: Path) -> set[str]:
    """Load source paper IDs from the test set to exclude from training."""
    source_ids = set()
    if not test_set_path.exists():
        return source_ids
    with open(test_set_path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                source_id = data.get("source_paper_id", "")
                if source_id:
                    source_ids.add(source_id)
    return source_ids


def mine_training_data(
    data_dir: Path,
    output_dir: Path,
    test_set_path: Path = Path("data/processed/test_set.jsonl"),
    openalex_email: Optional[str] = None,
    target_source_papers: int = 500,
    max_hard_negatives: int = 5,
    min_context_length: int = 50,
    max_cite_markers: int = 3,
    dev_fraction: float = 0.2,
    seed: int = 42,
    show_progress: bool = True,
) -> dict:
    """Mine training data from unarXiv JSONL files.

    Scans unarXiv files for papers with 15+ resolvable references,
    extracts citation contexts, fetches metadata for cited papers,
    and splits into train/dev by source paper.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    excluded_ids = _load_test_set_source_papers(test_set_path)
    print(f"Excluding {len(excluded_ids)} test-set source papers from training")

    jsonl_files = sorted(data_dir.glob("**/*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {data_dir}")
        return {"error": "No JSONL files found"}
    print(f"Found {len(jsonl_files)} JSONL files")

    client = OpenAlexClient(email=openalex_email)
    processor = UnarXivProcessor(openalex_client=client)

    print("\nPhase 1: Scanning for qualifying source papers...")
    qualifying_papers = []
    all_openalex_ids: set[str] = set()

    pbar = tqdm(jsonl_files, desc="Scanning files") if show_progress else jsonl_files
    for jsonl_path in pbar:
        for source in processor.iter_papers(jsonl_path):
            if source.paper_id in excluded_ids:
                continue
            if not source.citation_contexts:
                continue

            ref_ids = source.reference_openalex_ids
            if len(ref_ids) < 15:
                continue

            qualifying_papers.append(source)
            all_openalex_ids.update(ref_ids)

            if show_progress:
                pbar.set_postfix({"sources": len(qualifying_papers)})

            if len(qualifying_papers) >= target_source_papers:
                break

        if len(qualifying_papers) >= target_source_papers:
            break

    print(f"Found {len(qualifying_papers)} qualifying source papers")
    print(f"Need to fetch metadata for {len(all_openalex_ids)} unique references")

    if not qualifying_papers:
        return {"error": "No qualifying source papers found"}

    cache_path = output_dir / "openalex_cache.jsonl"
    papers_cache: dict[str, Paper] = {}

    if cache_path.exists():
        print(f"\nPhase 2: Loading cached metadata from {cache_path}...")
        with open(cache_path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    p = Paper(
                        id=d["id"],
                        title=d["title"],
                        abstract=d.get("abstract", ""),
                        authors=d.get("authors", []),
                        year=d.get("year"),
                        doi=d.get("doi"),
                        journal=d.get("journal"),
                    )
                    if p.abstract:
                        papers_cache[p.id] = p
        print(f"Loaded {len(papers_cache)} cached papers")

    ids_to_fetch = [oid for oid in all_openalex_ids if oid not in papers_cache]
    if ids_to_fetch:
        print(f"\nPhase 2: Fetching metadata for {len(ids_to_fetch)} new IDs from OpenAlex...")
        batch_size = 50
        new_papers: list[Paper] = []

        iterator = range(0, len(ids_to_fetch), batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc="Fetching metadata",
                total=len(ids_to_fetch) // batch_size + 1,
            )
        for i in iterator:
            batch = ids_to_fetch[i : i + batch_size]
            papers = client.get_works_batch(batch)
            for paper in papers:
                if paper.abstract:
                    papers_cache[paper.id] = paper
                    new_papers.append(paper)

        if new_papers:
            with open(cache_path, "a") as f:
                for p in new_papers:
                    f.write(
                        json.dumps(
                            {
                                "id": p.id,
                                "title": p.title,
                                "abstract": p.abstract,
                                "authors": p.authors,
                                "year": p.year,
                                "doi": p.doi,
                                "journal": p.journal,
                            }
                        )
                        + "\n"
                    )
            print(f"Cached {len(new_papers)} new papers to {cache_path}")
    else:
        print("\nPhase 2: All metadata found in cache, no API calls needed.")

    print(f"Total papers with abstracts: {len(papers_cache)}")

    print("\nPhase 3: Extracting training examples...")
    examples_by_source: dict[str, list[TrainingExample]] = {}

    stats = {
        "source_papers_scanned": len(qualifying_papers),
        "source_papers_with_examples": 0,
        "contexts_scanned": 0,
        "contexts_skipped_short": 0,
        "contexts_skipped_multi_cite": 0,
        "contexts_skipped_no_abstract": 0,
        "examples_created": 0,
    }

    for source in (
        tqdm(qualifying_papers, desc="Extracting") if show_progress else qualifying_papers
    ):
        ref_ids = source.reference_openalex_ids
        resolved = [rid for rid in ref_ids if rid in papers_cache]
        if len(resolved) < 15:
            continue

        ref_texts: dict[str, str] = {}
        for rid in resolved:
            ref_texts[rid] = _paper_to_embedding_text(papers_cache[rid])

        source_examples = []
        for ctx in source.citation_contexts:
            stats["contexts_scanned"] += 1
            cited_id = ctx["openalex_id"]

            if cited_id not in papers_cache:
                stats["contexts_skipped_no_abstract"] += 1
                continue

            query = ctx.get("narrow", ctx["text"])
            cleaned = clean_citation_markers(query)

            if len(cleaned) < min_context_length:
                stats["contexts_skipped_short"] += 1
                continue

            cite_count = query.count("[CITE]") + query.count("{{cite:")
            if cite_count > max_cite_markers:
                stats["contexts_skipped_multi_cite"] += 1
                continue

            other_ids = [rid for rid in resolved if rid != cited_id]
            random.seed(hash(f"{source.paper_id}_{cited_id}_{stats['examples_created']}"))
            neg_ids = random.sample(other_ids, min(max_hard_negatives, len(other_ids)))
            hard_negatives = [ref_texts[nid] for nid in neg_ids]

            example = TrainingExample(
                query=cleaned,
                positive=ref_texts[cited_id],
                hard_negatives=hard_negatives,
                source_paper_id=source.paper_id,
                cited_paper_id=cited_id,
            )
            source_examples.append(example)
            stats["examples_created"] += 1

        if source_examples:
            examples_by_source[source.paper_id] = source_examples
            stats["source_papers_with_examples"] += 1

    print(
        f"Created {stats['examples_created']} training examples "
        f"from {stats['source_papers_with_examples']} source papers"
    )

    print("\nPhase 4: Splitting into train/dev...")
    rng = random.Random(seed)
    source_ids = list(examples_by_source.keys())
    rng.shuffle(source_ids)

    dev_count = max(1, int(len(source_ids) * dev_fraction))
    dev_source_ids = set(source_ids[:dev_count])
    train_source_ids = set(source_ids[dev_count:])

    train_examples = []
    dev_examples = []
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
    stats["train_source_papers"] = len(train_source_ids)
    stats["dev_source_papers"] = len(dev_source_ids)

    print("\nData preparation complete:")
    print(f"  Source papers scanned:     {stats['source_papers_scanned']}")
    print(f"  Source papers with data:   {stats['source_papers_with_examples']}")
    print(f"  Total examples:            {stats['examples_created']}")
    print(
        f"  Train examples:            {stats['train_examples']} "
        f"({stats['train_source_papers']} source papers)"
    )
    print(
        f"  Dev examples:              {stats['dev_examples']} "
        f"({stats['dev_source_papers']} source papers)"
    )
    print(f"  Skipped (short):           {stats['contexts_skipped_short']}")
    print(f"  Skipped (multi-cite):      {stats['contexts_skipped_multi_cite']}")
    print(f"  Skipped (no abstract):     {stats['contexts_skipped_no_abstract']}")
    print("\nOutput files:")
    print(f"  {train_path}")
    print(f"  {dev_path}")

    return stats


def prepare_from_existing(
    test_set_path: Path,
    corpus_path: Path,
    output_dir: Path,
    max_hard_negatives: int = 5,
    min_context_length: int = 50,
    dev_fraction: float = 0.2,
    seed: int = 42,
) -> dict:
    """Create training data from existing test set and corpus — no API calls."""
    from incite.corpus.loader import load_corpus, load_test_set

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading corpus from {corpus_path}...")
    papers = load_corpus(corpus_path)
    paper_dict = {p.id: p for p in papers}
    print(f"Loaded {len(papers)} papers")

    print(f"Loading test set from {test_set_path}...")
    contexts = load_test_set(test_set_path)
    print(f"Loaded {len(contexts)} citation contexts")

    contexts_by_source: dict[str, list] = {}
    for ctx in contexts:
        sid = ctx.source_paper_id or "unknown"
        contexts_by_source.setdefault(sid, []).append(ctx)
    print(f"Source papers: {len(contexts_by_source)}")

    rng = random.Random(seed)
    source_ids = sorted(contexts_by_source.keys())
    rng.shuffle(source_ids)

    dev_count = max(1, int(len(source_ids) * dev_fraction))
    dev_source_ids = set(source_ids[:dev_count])
    train_source_ids = set(source_ids[dev_count:])

    print(f"Split: {len(train_source_ids)} train / {len(dev_source_ids)} dev source papers")

    stats = {
        "total_contexts": len(contexts),
        "examples_created": 0,
        "skipped_short": 0,
        "skipped_no_positive": 0,
        "skipped_few_refs": 0,
    }

    train_examples = []
    dev_examples = []

    from incite.finetuning.data_sources import _sample_scale

    example_idx = 0
    for sid in source_ids:
        is_dev = sid in dev_source_ids
        for ctx in contexts_by_source[sid]:
            rng_ex = random.Random(hash(f"{sid}_{ctx.id}_{example_idx}"))
            example_idx += 1

            scale = _sample_scale(rng_ex)
            query = ctx.get_query(scale=scale, clean=True)
            if len(query) < min_context_length:
                stats["skipped_short"] += 1
                continue

            if not ctx.ground_truth_ids:
                stats["skipped_no_positive"] += 1
                continue
            gt_id = ctx.ground_truth_ids[0]
            if gt_id not in paper_dict:
                stats["skipped_no_positive"] += 1
                continue

            positive_text = _paper_to_embedding_text(paper_dict[gt_id])

            ref_ids = [rid for rid in ctx.reference_set_ids if rid != gt_id and rid in paper_dict]
            if len(ref_ids) < 5:
                stats["skipped_few_refs"] += 1
                continue

            rng_neg = random.Random(hash(f"{sid}_{ctx.id}"))
            neg_ids = rng_neg.sample(ref_ids, min(max_hard_negatives, len(ref_ids)))
            hard_negatives = [_paper_to_embedding_text(paper_dict[nid]) for nid in neg_ids]

            example = TrainingExample(
                query=query,
                positive=positive_text,
                hard_negatives=hard_negatives,
                source_paper_id=sid,
                cited_paper_id=gt_id,
                scale=scale,
            )
            stats["examples_created"] += 1

            if is_dev:
                dev_examples.append(example)
            else:
                train_examples.append(example)

    rng.shuffle(train_examples)

    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"

    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex.to_dict()) + "\n")

    with open(dev_path, "w") as f:
        for ex in dev_examples:
            f.write(json.dumps(ex.to_dict()) + "\n")

    split_path = output_dir / "split_info.json"
    with open(split_path, "w") as f:
        json.dump(
            {
                "train_source_papers": sorted(train_source_ids),
                "dev_source_papers": sorted(dev_source_ids),
                "seed": seed,
                "dev_fraction": dev_fraction,
            },
            f,
            indent=2,
        )

    stats["train_examples"] = len(train_examples)
    stats["dev_examples"] = len(dev_examples)
    stats["train_source_papers"] = len(train_source_ids)
    stats["dev_source_papers"] = len(dev_source_ids)

    print("\nData preparation complete (from existing data):")
    print(f"  Total contexts:       {stats['total_contexts']}")
    print(f"  Examples created:     {stats['examples_created']}")
    print(
        f"  Train:                {stats['train_examples']} "
        f"({stats['train_source_papers']} source papers)"
    )
    print(
        f"  Dev:                  {stats['dev_examples']} "
        f"({stats['dev_source_papers']} source papers)"
    )
    print(f"  Skipped (short):      {stats['skipped_short']}")
    print(f"  Skipped (no positive):{stats['skipped_no_positive']}")
    print(f"  Skipped (few refs):   {stats['skipped_few_refs']}")
    print("\nOutput files:")
    print(f"  {train_path}")
    print(f"  {dev_path}")
    print(f"  {split_path}")

    return stats
