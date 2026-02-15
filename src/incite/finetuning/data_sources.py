"""Data sources for fine-tuning training data.

Each source streams TrainingExamples from an external dataset or local data.
Sources are combined by the DataPipeline in data_pipeline.py.

DataSource is a Protocol (structural typing) -- any class with `name`, `stream()`,
and `count_available()` satisfies it without inheriting.
"""

import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Iterator, Optional, Protocol, runtime_checkable

import requests

from incite.finetuning.data_preparation import TrainingExample
from incite.models import clean_citation_markers, format_paper_embedding_text

logger = logging.getLogger(__name__)

# Minimum query length after cleaning (chars)
MIN_QUERY_LENGTH = 50

# Multi-scale context distribution for training examples.
# Exposes the model to varying context sizes during training so it
# handles local (1-sentence) through section-level queries at inference.
SCALE_DISTRIBUTION: dict[str, float] = {
    "local": 0.30,  # 1 sentence
    "narrow": 0.40,  # 3 sentences
    "broad": 0.20,  # 6 sentences
    "section": 0.10,  # full section
}
_SCALE_NAMES = list(SCALE_DISTRIBUTION.keys())
_SCALE_WEIGHTS = list(SCALE_DISTRIBUTION.values())


def _sample_scale(rng: random.Random) -> str:
    """Sample a context scale from SCALE_DISTRIBUTION using the given RNG."""
    return rng.choices(_SCALE_NAMES, weights=_SCALE_WEIGHTS, k=1)[0]


def _format_positive(title: str, abstract: str) -> str:
    """Format title + abstract as positive text, matching Paper.to_embedding_text().

    Delegates to format_paper_embedding_text() for consistent ". " separator
    (not [SEP]) across retrieval and training.
    """
    return format_paper_embedding_text(
        title=title,
        abstract=abstract,
        include_abstract=True,
        include_metadata=True,
    )


def _clean_query(text: str) -> str:
    """Clean a citation context for use as a query."""
    # Remove citation markers like [1], [2,3], (Author, 2020), etc.
    text = clean_citation_markers(text)
    # Remove common S2ORC artifacts
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@runtime_checkable
class DataSource(Protocol):
    """Protocol for training data sources."""

    name: str

    def stream(self, limit: int = 0) -> Iterator[TrainingExample]:
        """Stream training examples. limit=0 means unlimited."""
        ...

    def count_available(self) -> Optional[int]:
        """Return estimated number of available examples, or None if unknown."""
        ...


class S2ORCSource:
    """Stream citation context -> paper pairs from S2ORC on HuggingFace.

    Uses the sentence-transformers/s2orc dataset, abstract-citation-pair subset.
    Fields: abstract (cited paper), citation (inline citation context).
    39.6M rows; we stream a configurable subset.
    """

    name = "s2orc"

    def stream(self, limit: int = 100_000) -> Iterator[TrainingExample]:
        from datasets import load_dataset

        ds = load_dataset(
            "sentence-transformers/s2orc",
            "abstract-citation-pair",
            split="train",
            streaming=True,
        )

        count = 0
        for ex in ds:
            abstract = ex.get("abstract", "")
            citation = ex.get("citation", "")

            if not abstract or not citation:
                continue

            query = _clean_query(citation)
            if len(query) < MIN_QUERY_LENGTH:
                continue

            positive = abstract.strip()
            if len(positive) < 30:
                continue

            yield TrainingExample(
                query=query,
                positive=positive,
                source=self.name,
            )

            count += 1
            if limit and count >= limit:
                return

    def count_available(self) -> Optional[int]:
        return 39_600_000


class SciCiteSource:
    """Stream from allenai/scicite, fetching cited paper abstracts via S2 API.

    SciCite has ~8K training examples with citation contexts and S2 paper IDs.
    We batch-fetch abstracts from Semantic Scholar.
    """

    name = "scicite"

    def __init__(self, s2_api_key: Optional[str] = None):
        self.s2_api_key = s2_api_key

    def _fetch_s2_papers_batch(self, paper_ids: list[str]) -> dict[str, dict]:
        """Batch-fetch paper metadata from S2 API.

        Returns:
            Dict mapping paper_id -> {title, abstract}
        """
        results = {}
        headers = {}
        if self.s2_api_key:
            headers["x-api-key"] = self.s2_api_key

        url = "https://api.semanticscholar.org/graph/v1/paper/batch"
        batch_size = 100

        for i in range(0, len(paper_ids), batch_size):
            batch = paper_ids[i : i + batch_size]
            try:
                resp = requests.post(
                    url,
                    json={"ids": batch},
                    params={"fields": "paperId,title,abstract"},
                    headers=headers,
                    timeout=30,
                )
                resp.raise_for_status()
                for paper in resp.json():
                    if paper and paper.get("title") and paper.get("abstract"):
                        results[paper["paperId"]] = {
                            "title": paper["title"],
                            "abstract": paper["abstract"],
                        }
            except requests.RequestException as e:
                logger.warning("S2 batch fetch error: %s", e)

            time.sleep(1.0)

        return results

    def stream(self, limit: int = 0) -> Iterator[TrainingExample]:
        import io
        import tarfile

        tar_url = "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz"
        try:
            print("    Downloading SciCite from S3...")
            resp = requests.get(tar_url, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning("Could not download SciCite: %s", e)
            return

        raw_data = []
        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith("train.jsonl"):
                    f = tar.extractfile(member)
                    if f:
                        for line in f:
                            if line.strip():
                                raw_data.append(json.loads(line))
                    break

        if not raw_data:
            logger.warning("No train.jsonl found in SciCite tarball")
            return

        cited_ids = list(
            {ex["citedPaperId"] for ex in raw_data if ex.get("citedPaperId") and ex.get("string")}
        )

        logger.info("SciCite: %d examples, %d unique cited papers", len(raw_data), len(cited_ids))

        paper_cache = self._fetch_s2_papers_batch(cited_ids)
        logger.info("SciCite: fetched %d paper abstracts from S2", len(paper_cache))

        count = 0
        for ex in raw_data:
            cited_id = ex.get("citedPaperId", "")
            context = ex.get("string", "")

            if not cited_id or not context:
                continue
            if cited_id not in paper_cache:
                continue

            query = _clean_query(context)
            if len(query) < MIN_QUERY_LENGTH:
                continue

            paper = paper_cache[cited_id]
            positive = _format_positive(paper["title"], paper["abstract"])

            yield TrainingExample(
                query=query,
                positive=positive,
                cited_paper_id=cited_id,
                source=self.name,
            )

            count += 1
            if limit and count >= limit:
                return

    def count_available(self) -> Optional[int]:
        return 8_000


class SyntheticZoteroSource:
    """Read from existing synthetic_contexts.db + zotero_corpus.jsonl.

    Synthetic contexts were generated by Claude Haiku for papers in the
    user's Zotero library. Each has a target paper and a K-NN reference set.
    """

    name = "synthetic"

    def __init__(
        self,
        db_path: Path = Path("data/processed/synthetic_contexts.db"),
        corpus_path: Path = Path("data/processed/zotero_corpus.jsonl"),
    ):
        self.db_path = db_path
        self.corpus_path = corpus_path

    def stream(self, limit: int = 0) -> Iterator[TrainingExample]:
        from incite.corpus.loader import load_corpus
        from incite.corpus.synthetic_db import SyntheticDB

        if not self.db_path.exists():
            logger.warning("Synthetic DB not found at %s", self.db_path)
            return
        if not self.corpus_path.exists():
            logger.warning("Zotero corpus not found at %s", self.corpus_path)
            return

        papers = load_corpus(self.corpus_path)
        paper_dict = {p.id: p for p in papers}

        db = SyntheticDB(self.db_path)
        contexts = db.get_contexts()

        count = 0
        for ctx in contexts:
            paper_id = ctx["paper_id"]
            text = ctx["text"]

            if paper_id not in paper_dict:
                continue

            query = _clean_query(text)
            if len(query) < MIN_QUERY_LENGTH:
                continue

            paper = paper_dict[paper_id]
            positive = paper.to_embedding_text(include_abstract=True, include_metadata=True)

            yield TrainingExample(
                query=query,
                positive=positive,
                cited_paper_id=paper_id,
                source=self.name,
            )

            count += 1
            if limit and count >= limit:
                return

    def count_available(self) -> Optional[int]:
        if self.db_path.exists():
            from incite.corpus.synthetic_db import SyntheticDB

            db = SyntheticDB(self.db_path)
            stats = db.stats()
            return stats.get("total_contexts", 0)
        return 0


class UnarXivSource:
    """Wrapper around existing mine_training_data() for unarXiv.

    Mines citation contexts from unarXiv JSONL files, fetches metadata
    from OpenAlex, and includes hard negatives from reference sets.
    """

    name = "unarxiv"

    def __init__(
        self,
        data_dir: Path = Path("data/raw/unarxiv"),
        test_set_path: Path = Path("data/processed/test_set.jsonl"),
        openalex_email: Optional[str] = None,
        target_source_papers: int = 500,
    ):
        self.data_dir = data_dir
        self.test_set_path = test_set_path
        self.openalex_email = openalex_email
        self.target_source_papers = target_source_papers

    def stream(self, limit: int = 0) -> Iterator[TrainingExample]:
        """Stream from unarXiv by running the mining pipeline.

        Note: This is expensive (API calls to OpenAlex). Results are
        yielded progressively but fetching metadata requires upfront work.
        For large runs, use mine_training_data() directly.
        """
        import tempfile

        from incite.finetuning.data_preparation import mine_training_data

        # Mine to temp dir, then read back
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stats = mine_training_data(
                data_dir=self.data_dir,
                output_dir=tmp_path,
                test_set_path=self.test_set_path,
                openalex_email=self.openalex_email,
                target_source_papers=self.target_source_papers,
            )

            if "error" in stats:
                logger.warning("unarXiv mining failed: %s", stats["error"])
                return

            # Read back and yield
            count = 0
            for split in ["train.jsonl", "dev.jsonl"]:
                path = tmp_path / split
                if not path.exists():
                    continue
                with open(path) as f:
                    for line in f:
                        if line.strip():
                            ex = TrainingExample.from_dict(json.loads(line))
                            ex.source = self.name
                            yield ex
                            count += 1
                            if limit and count >= limit:
                                return

    def count_available(self) -> Optional[int]:
        return None


class ExistingDataSource:
    """Stream training examples from existing test set + corpus.

    Pairs citation contexts with their ground-truth papers and samples
    hard negatives from the reference set. No temp-dir roundtrip -- yields
    TrainingExamples directly.
    """

    name = "existing"

    def __init__(
        self,
        test_set_path: Path = Path("data/processed/test_set.jsonl"),
        corpus_path: Path = Path("data/processed/corpus.jsonl"),
        max_hard_negatives: int = 5,
        min_context_length: int = 50,
    ):
        self.test_set_path = test_set_path
        self.corpus_path = corpus_path
        self.max_hard_negatives = max_hard_negatives
        self.min_context_length = min_context_length

    def stream(self, limit: int = 0) -> Iterator[TrainingExample]:
        from incite.corpus.loader import load_corpus, load_test_set
        from incite.finetuning.data_preparation import _paper_to_embedding_text

        papers = load_corpus(self.corpus_path)
        paper_dict = {p.id: p for p in papers}

        contexts = load_test_set(self.test_set_path)

        count = 0
        for idx, ctx in enumerate(contexts):
            sid = ctx.source_paper_id or "unknown"
            # Deterministic RNG seeded per-example for reproducibility
            rng = random.Random(hash(f"{sid}_{ctx.id}_{idx}"))

            # Randomly sample context scale
            scale = _sample_scale(rng)
            query = ctx.get_query(scale=scale, clean=True)
            if len(query) < self.min_context_length:
                continue

            if not ctx.ground_truth_ids:
                continue
            gt_id = ctx.ground_truth_ids[0]
            if gt_id not in paper_dict:
                continue

            positive_text = _paper_to_embedding_text(paper_dict[gt_id])

            ref_ids = [rid for rid in ctx.reference_set_ids if rid != gt_id and rid in paper_dict]
            if len(ref_ids) < 5:
                continue

            neg_ids = rng.sample(ref_ids, min(self.max_hard_negatives, len(ref_ids)))
            hard_negatives = [_paper_to_embedding_text(paper_dict[nid]) for nid in neg_ids]

            yield TrainingExample(
                query=query,
                positive=positive_text,
                hard_negatives=hard_negatives,
                source_paper_id=sid,
                cited_paper_id=gt_id,
                source=self.name,
                scale=scale,
            )

            count += 1
            if limit and count >= limit:
                return

    def count_available(self) -> Optional[int]:
        if self.test_set_path.exists():
            count = 0
            with open(self.test_set_path) as f:
                for line in f:
                    if line.strip():
                        count += 1
            return count
        return 0


class FileSource:
    """Read TrainingExamples from existing JSONL file(s).

    Useful for including pre-generated passage-level data (from
    generate-passages or generate-fulltext-passages) in the DataPipeline.
    """

    def __init__(self, paths: list[Path], source_name: str = "file"):
        self.paths = paths
        self.name = source_name

    def stream(self, limit: int = 0) -> Iterator[TrainingExample]:
        count = 0
        for path in self.paths:
            if not path.exists():
                logger.warning("FileSource: %s not found, skipping", path)
                continue
            with open(path) as f:
                for line in f:
                    if line.strip():
                        ex = TrainingExample.from_dict(json.loads(line))
                        if not ex.source:
                            ex.source = self.name
                        yield ex
                        count += 1
                        if limit and count >= limit:
                            return

    def count_available(self) -> Optional[int]:
        total = 0
        for path in self.paths:
            if path.exists():
                with open(path) as f:
                    total += sum(1 for line in f if line.strip())
        return total
