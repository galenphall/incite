"""Corpus loading and saving utilities."""

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from incite.corpus.chunking import is_bibliography
from incite.models import Chunk, CitationContext, Paper


@contextmanager
def _atomic_write(path: Path):
    """Context manager for atomic file writes.

    Writes to a temporary file in the same directory, then atomically
    renames to the target path. This prevents data loss if the process
    is interrupted mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            yield f
        os.replace(tmp_path, path)
    except BaseException:
        os.unlink(tmp_path)
        raise


class CorpusFileSource:
    """CorpusSource implementation that reads from a JSONL corpus file.

    Satisfies the CorpusSource protocol via structural typing.
    """

    name: str = "file"

    def __init__(self, path: str | Path):
        """Initialize CorpusFileSource.

        Args:
            path: Path to corpus.jsonl file
        """
        self.path = Path(path)
        self._last_mtime: float = 0.0

    def load_papers(self) -> list[Paper]:
        """Load papers from the JSONL file."""
        papers = load_corpus(self.path)
        self._last_mtime = self.path.stat().st_mtime
        return papers

    def needs_refresh(self) -> bool:
        """Check if the file has been modified since last load."""
        if not self.path.exists():
            return False
        return self.path.stat().st_mtime > self._last_mtime

    def cache_key(self) -> str:
        """Return cache key based on the file path."""
        return f"file_{self.path.stem}"


def load_corpus(path: str | Path) -> list[Paper]:
    """Load corpus from JSONL file.

    Args:
        path: Path to corpus.jsonl file

    Returns:
        List of Paper objects
    """
    papers = []
    path = Path(path)

    with open(path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                papers.append(
                    Paper(
                        id=data["id"],
                        title=data["title"],
                        abstract=data.get("abstract", ""),
                        authors=data.get("authors", []),
                        year=data.get("year"),
                        doi=data.get("doi"),
                        bibtex_key=data.get("bibtex_key"),
                        journal=data.get("journal"),
                        full_text=data.get("full_text"),
                        paragraphs=data.get("paragraphs", []),
                        source_file=data.get("source_file"),
                        llm_description=data.get("llm_description"),
                        zotero_uri=data.get("zotero_uri"),
                    )
                )

    return papers


def save_corpus(papers: list[Paper], path: str | Path) -> None:
    """Save corpus to JSONL file.

    Args:
        papers: List of Paper objects
        path: Output path
    """
    path = Path(path)

    with _atomic_write(path) as f:
        for paper in papers:
            data = {
                "id": paper.id,
                "title": paper.title,
                "abstract": paper.abstract,
                "authors": paper.authors,
                "year": paper.year,
                "doi": paper.doi,
                "bibtex_key": paper.bibtex_key,
                "journal": paper.journal,
                "full_text": paper.full_text,
                "paragraphs": paper.paragraphs,
                "source_file": paper.source_file,
                "llm_description": paper.llm_description,
                "zotero_uri": paper.zotero_uri,
            }
            f.write(json.dumps(data) + "\n")


def load_test_set(path: str | Path) -> list[CitationContext]:
    """Load test set from JSONL file.

    Args:
        path: Path to test_set.jsonl file

    Returns:
        List of CitationContext objects
    """
    contexts = []
    path = Path(path)

    with open(path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                contexts.append(
                    CitationContext(
                        id=data["id"],
                        local_context=data["local_context"],
                        narrow_context=data.get("narrow_context", ""),
                        broad_context=data.get("broad_context", ""),
                        section_context=data.get("section_context", ""),
                        global_context=data.get("global_context", ""),
                        source_paper_id=data.get("source_paper_id"),
                        ground_truth_ids=data.get("ground_truth_ids", []),
                        reference_set_ids=list(dict.fromkeys(data.get("reference_set_ids", []))),
                        mentioned_authors=data.get("mentioned_authors", []),
                        mentioned_years=data.get("mentioned_years", []),
                        reformulated_query=data.get("reformulated_query", ""),
                        difficulty=data.get("difficulty", ""),
                    )
                )

    return contexts


def save_test_set(contexts: list[CitationContext], path: str | Path) -> None:
    """Save test set to JSONL file.

    Args:
        contexts: List of CitationContext objects
        path: Output path
    """
    path = Path(path)

    with _atomic_write(path) as f:
        for context in contexts:
            data = {
                "id": context.id,
                "local_context": context.local_context,
                "narrow_context": context.narrow_context,
                "broad_context": context.broad_context,
                "section_context": context.section_context,
                "global_context": context.global_context,
                "source_paper_id": context.source_paper_id,
                "ground_truth_ids": context.ground_truth_ids,
                "reference_set_ids": context.reference_set_ids,
                "mentioned_authors": context.mentioned_authors,
                "mentioned_years": context.mentioned_years,
                "reformulated_query": context.reformulated_query,
                "difficulty": context.difficulty,
            }
            f.write(json.dumps(data) + "\n")


def load_chunks(path: str | Path, filter_bibliography: bool = True) -> list[Chunk]:
    """Load chunks from JSONL file.

    Args:
        path: Path to chunks.jsonl file
        filter_bibliography: If True (default), filter out bibliography chunks.
            This catches chunks that slipped through during initial chunking
            (e.g., from stale caches or edge cases).

    Returns:
        List of Chunk objects
    """
    chunks = []
    path = Path(path)

    with open(path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                chunk = Chunk(
                    id=data["id"],
                    paper_id=data["paper_id"],
                    text=data["text"],
                    section=data.get("section"),
                    char_offset=data.get("char_offset", 0),
                    page_number=data.get("page_number"),
                    source=data.get("source"),
                    context_text=data.get("context_text"),
                    parent_text=data.get("parent_text"),
                )
                # Filter out bibliography chunks if requested
                if filter_bibliography and is_bibliography(chunk):
                    continue
                chunks.append(chunk)

    return chunks


def save_chunks(chunks: list[Chunk], path: str | Path) -> None:
    """Save chunks to JSONL file.

    Args:
        chunks: List of Chunk objects
        path: Output path
    """
    path = Path(path)

    with _atomic_write(path) as f:
        for chunk in chunks:
            data = {
                "id": chunk.id,
                "paper_id": chunk.paper_id,
                "text": chunk.text,
                "section": chunk.section,
                "char_offset": chunk.char_offset,
                "page_number": chunk.page_number,
                "source": chunk.source,
                "context_text": chunk.context_text,
                "parent_text": chunk.parent_text,
            }
            f.write(json.dumps(data) + "\n")
