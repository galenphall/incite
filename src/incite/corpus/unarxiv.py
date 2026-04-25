"""unarXiv dataset processing for citation context extraction.

This module provides the core extraction machinery for unarXiv JSONL files:
parsing papers, segmenting sentences, and assembling citation contexts.

The top-level pipeline orchestrator (``process_unarxiv_directory``) lives in
``incite.corpus.unarxiv_pipeline`` and is re-exported here for backward
compatibility.

unarXiv data format:
- paper_id: arXiv ID
- metadata: {title, authors, abstract, ...}
- body_text: [{section, text, cite_spans: [{start, end, ref_id}]}]
- bib_entries: {ref_id: {bib_entry_raw, ids: {open_alex_id, doi, ...}}}

Classes:
    BibEntry: A parsed bibliography entry with resolved external IDs.
    SourcePaper: A source paper with extracted citation contexts.
    UnarXivProcessor: Core processor — parses JSONL files, extracts contexts,
        fetches OpenAlex metadata.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from tqdm import tqdm

from incite.corpus.openalex import OpenAlexClient
from incite.models import CitationContext, Paper, clean_citation_markers

# Lazy-loaded spaCy model for sentence segmentation
_nlp_parser = None
_parser_available: Optional[bool] = None


def _get_parser_model():
    """Get parser model for sentence segmentation.

    Returns model configured for dependency parsing (needed for sentence boundaries).
    Returns None if no NLP model is available.
    """
    global _nlp_parser, _parser_available

    if _parser_available is not None:
        return _nlp_parser

    try:
        import spacy

        _nlp_parser = spacy.load("en_core_web_sm")
        _nlp_parser.select_pipes(enable=["tok2vec", "parser"])
        _parser_available = True
        return _nlp_parser
    except (ImportError, OSError):
        _nlp_parser = None
        _parser_available = False
        return None


@dataclass
class BibEntry:
    """A bibliography entry from unarXiv."""

    ref_id: str
    raw_text: str
    openalex_id: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None

    @classmethod
    def from_unarxiv(cls, ref_id: str, data: dict) -> "BibEntry":
        """Create from unarXiv bib_entries format."""
        ids = data.get("ids", {})

        # Extract OpenAlex ID from URL if present
        openalex_url = ids.get("open_alex_id", "")
        openalex_id = openalex_url.split("/")[-1] if openalex_url else None

        return cls(
            ref_id=ref_id,
            raw_text=data.get("bib_entry_raw", ""),
            openalex_id=openalex_id,
            doi=ids.get("doi") or None,
            arxiv_id=ids.get("arxiv_id") or None,
        )


@dataclass
class SourcePaper:
    """A source paper from unarXiv with its citation contexts."""

    paper_id: str
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    bib_entries: dict[str, BibEntry] = field(default_factory=dict)
    citation_contexts: list[dict] = field(default_factory=list)

    @property
    def reference_openalex_ids(self) -> list[str]:
        """Get all OpenAlex IDs from bibliography."""
        return [entry.openalex_id for entry in self.bib_entries.values() if entry.openalex_id]


class UnarXivProcessor:
    """Process unarXiv JSONL files to extract citation contexts.

    Designed to be extensible - can process any number of JSONL files
    from the unarXiv dataset.
    """

    def __init__(
        self,
        openalex_client: Optional[OpenAlexClient] = None,
        min_context_length: int = 50,
        max_context_length: int = 500,
    ):
        """Initialize processor.

        Args:
            openalex_client: Client for fetching paper metadata
            min_context_length: Minimum characters for valid context
            max_context_length: Maximum characters for context extraction
        """
        self.openalex_client = openalex_client or OpenAlexClient()
        self.min_context_length = min_context_length
        self.max_context_length = max_context_length
        # Cache for sentence splits (cleared per paper)
        self._sentence_cache: dict[str, list[tuple[int, int, str]]] = {}

    def _batch_split_sentences(self, texts: list[str]) -> None:
        """Batch process texts with spaCy and cache sentence splits.

        Uses nlp.pipe() for efficient batch processing (~5-10x faster).
        Results are cached in self._sentence_cache keyed by text.
        """
        nlp = _get_parser_model()
        if nlp is None:
            # Fall back to processing one at a time with regex
            for text in texts:
                if text and text not in self._sentence_cache:
                    self._sentence_cache[text] = self._split_into_sentences_regex(text)
            return

        # Filter to texts we haven't cached yet
        texts_to_process = [t for t in texts if t and t not in self._sentence_cache]
        if not texts_to_process:
            return

        # Batch process with spaCy - much faster than individual calls
        for doc, text in zip(nlp.pipe(texts_to_process, batch_size=50), texts_to_process):
            sentences = []
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if sent_text:
                    sentences.append((sent.start_char, sent.end_char, sent_text))
            self._sentence_cache[text] = sentences

    def iter_papers(self, jsonl_path: Path | str) -> Iterator[SourcePaper]:
        """Iterate over papers in a unarXiv JSONL file.

        Args:
            jsonl_path: Path to JSONL file

        Yields:
            SourcePaper objects
        """
        path = Path(jsonl_path)

        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)
                yield self._parse_paper(data)

    def _parse_paper(self, data: dict) -> SourcePaper:
        """Parse a single paper from unarXiv JSON."""
        metadata = data.get("metadata", {})

        # Parse authors
        authors = []
        for author_parts in metadata.get("authors_parsed", []):
            if len(author_parts) >= 2:
                # Format: [Last, First, Suffix]
                authors.append(f"{author_parts[1]} {author_parts[0]}".strip())
            elif author_parts:
                authors.append(author_parts[0])

        # Parse bibliography entries
        bib_entries = {}
        for ref_id, bib_data in data.get("bib_entries", {}).items():
            bib_entries[ref_id] = BibEntry.from_unarxiv(ref_id, bib_data)

        # Clear sentence cache for this paper and batch-process all paragraphs
        self._sentence_cache.clear()
        paragraphs = data.get("body_text", [])
        all_texts = [p.get("text", "") for p in paragraphs]
        self._batch_split_sentences(all_texts)

        # Extract citation contexts from body text
        # Track previous paragraph to allow context from adjacent paragraphs
        citation_contexts = []
        prev_paragraph_text = ""
        for paragraph in paragraphs:
            contexts = self._extract_contexts_from_paragraph(
                paragraph, bib_entries, prev_paragraph_text
            )
            citation_contexts.extend(contexts)
            # Update previous paragraph for next iteration
            prev_paragraph_text = paragraph.get("text", "")

        return SourcePaper(
            paper_id=data.get("paper_id", ""),
            title=metadata.get("title", "").replace("\n", " ").strip(),
            abstract=metadata.get("abstract", "").strip(),
            authors=authors,
            bib_entries=bib_entries,
            citation_contexts=citation_contexts,
        )

    def _extract_contexts_from_paragraph(
        self,
        paragraph: dict,
        bib_entries: dict[str, BibEntry],
        prev_paragraph_text: str = "",
    ) -> list[dict]:
        """Extract citation contexts from a paragraph.

        Args:
            paragraph: Paragraph dict with text, cite_spans, section
            bib_entries: Bibliography entries for reference lookup
            prev_paragraph_text: Text from the previous paragraph (for cross-boundary context)

        Returns:
            List of context dicts with text and reference info
        """
        text = paragraph.get("text", "")
        cite_spans = paragraph.get("cite_spans", [])
        section = paragraph.get("section", "")

        if not cite_spans:
            return []

        contexts = []

        for span in cite_spans:
            ref_id = span.get("ref_id")
            if not ref_id or ref_id not in bib_entries:
                continue

            bib_entry = bib_entries[ref_id]

            # Skip if no OpenAlex ID (can't resolve)
            if not bib_entry.openalex_id:
                continue

            start = span.get("start", 0)
            end = span.get("end", start)

            # Extract context around citation
            context_text = self._get_context_window(text, start, end)

            # Validate cleaned text meets minimum length
            cleaned = clean_citation_markers(context_text)
            if len(cleaned) < self.min_context_length:
                # Context is mostly citations, skip it
                continue

            # Extract narrow and broad sentence windows (with cross-paragraph support)
            narrow, broad = self._get_sentence_windows(
                text, start, prev_paragraph_text=prev_paragraph_text
            )

            # Replace the citation marker with [CITE]
            cite_marker = text[start:end] if start < len(text) and end <= len(text) else ""
            if cite_marker:
                context_text = context_text.replace(cite_marker, "[CITE]", 1)
                narrow = narrow.replace(cite_marker, "[CITE]", 1)
                broad = broad.replace(cite_marker, "[CITE]", 1)

            # Note: mentioned_authors/years are no longer extracted via NER.
            # Author boosting is now handled at retrieval time by matching
            # candidate paper authors against the query text.

            contexts.append(
                {
                    "text": context_text,
                    "narrow": narrow,
                    "broad": broad,
                    "section": section,
                    "ref_id": ref_id,
                    "openalex_id": bib_entry.openalex_id,
                }
            )

        return contexts

    def _split_into_sentences(self, text: str) -> list[tuple[int, int, str]]:
        """Split text into sentences with their character offsets.

        Uses batch-processed cache if available (from _batch_split_sentences),
        otherwise processes with spaCy directly. Falls back to regex
        if spaCy is unavailable.

        Returns:
            List of (start_offset, end_offset, sentence_text) tuples
        """
        # Check cache first (populated by _batch_split_sentences)
        if text in self._sentence_cache:
            return self._sentence_cache[text]

        nlp = _get_parser_model()

        if nlp is not None:
            # Use spaCy for accurate sentence segmentation
            doc = nlp(text)
            sentences = []
            for sent in doc.sents:
                sentences.append((sent.start_char, sent.end_char, sent.text.strip()))
            self._sentence_cache[text] = sentences
            return sentences
        else:
            # Fallback: regex-based splitting (less accurate for scientific text)
            result = self._split_into_sentences_regex(text)
            self._sentence_cache[text] = result
            return result

    def _split_into_sentences_regex(self, text: str) -> list[tuple[int, int, str]]:
        """Fallback regex-based sentence splitting.

        Less accurate than spaCy - may incorrectly split on abbreviations
        like "et al.", "Fig.", "Eq.", etc.

        Returns:
            List of (start_offset, end_offset, sentence_text) tuples
        """
        # Split on sentence-ending punctuation followed by space and capital letter
        pattern = r"(?<=[.!?])\s+(?=[A-Z])"

        sentences = []
        pos = 0
        parts = re.split(pattern, text)

        for part in parts:
            part_stripped = part.strip()
            if part_stripped:
                # Find the actual start position in original text
                start = text.find(part, pos)
                if start == -1:
                    start = pos
                end = start + len(part)
                sentences.append((start, end, part_stripped))
                pos = end

        return sentences

    def _find_sentence_idx_containing(
        self, sentences: list[tuple[int, int, str]], cite_start: int
    ) -> int:
        """Find the index of the sentence containing the citation."""
        for i, (start, end, _) in enumerate(sentences):
            if start <= cite_start < end:
                return i
        # Default to last sentence if not found
        return len(sentences) - 1 if sentences else 0

    def _get_sentence_windows(
        self,
        text: str,
        cite_start: int,
        narrow_before: int = 2,
        broad_before: int = 5,
        forward_n: int = 1,
        prev_paragraph_text: str = "",
    ) -> tuple[str, str]:
        """Extract narrow and broad context windows by sentence count.

        Args:
            text: Full paragraph text
            cite_start: Character offset of citation start (relative to current paragraph)
            narrow_before: Sentences BEFORE citing sentence for narrow window
            broad_before: Sentences BEFORE citing sentence for broad window
            forward_n: Sentences AFTER citing sentence (included in both windows)
            prev_paragraph_text: Text from the previous paragraph for cross-boundary context

        Returns:
            Tuple of (narrow_context, broad_context), both including citing sentence
            and forward context. Narrow = narrow_before + citing + forward_n.
            Broad = broad_before + citing + forward_n.
        """
        sentences = self._split_into_sentences(text)

        if not sentences:
            return text.strip(), text.strip()

        cite_idx = self._find_sentence_idx_containing(sentences, cite_start)

        # Calculate how many sentences we need before the citing sentence
        narrow_needed_before = narrow_before - cite_idx  # How many more we need from prev para
        broad_needed_before = broad_before - cite_idx

        # Get sentences from previous paragraph if needed and available
        prev_sentences: list[tuple[int, int, str]] = []
        if prev_paragraph_text and (narrow_needed_before > 0 or broad_needed_before > 0):
            # Filter out very short "paragraphs" that are likely section headings
            # A useful paragraph should have at least 30 characters of content
            if len(prev_paragraph_text.strip()) >= 30:
                prev_sentences = self._split_into_sentences(prev_paragraph_text)

        # Include sentences before + citing sentence + sentences after
        narrow_start = max(0, cite_idx - narrow_before)
        broad_start = max(0, cite_idx - broad_before)
        # End index is exclusive, so cite_idx + 1 + forward_n includes forward sentences
        end_idx = min(len(sentences), cite_idx + 1 + forward_n)

        # Build narrow context
        narrow_parts = []
        if prev_sentences and narrow_needed_before > 0:
            # Take sentences from the end of the previous paragraph
            prev_start = max(0, len(prev_sentences) - narrow_needed_before)
            narrow_parts.extend(s[2] for s in prev_sentences[prev_start:])
        narrow_parts.extend(s[2] for s in sentences[narrow_start:end_idx])
        narrow = " ".join(narrow_parts)

        # Build broad context
        broad_parts = []
        if prev_sentences and broad_needed_before > 0:
            # Take sentences from the end of the previous paragraph
            prev_start = max(0, len(prev_sentences) - broad_needed_before)
            broad_parts.extend(s[2] for s in prev_sentences[prev_start:])
        broad_parts.extend(s[2] for s in sentences[broad_start:end_idx])
        broad = " ".join(broad_parts)

        return narrow.strip(), broad.strip()

    def _get_context_window(self, text: str, cite_start: int, cite_end: int) -> str:
        """Extract a window of text around a citation.

        Strategy:
        1. Find the sentence containing the citation
        2. Expand to include adjacent sentences if cleaned text is too short
        3. Fall back to full paragraph if needed
        4. Validate cleaned text meets minimum length
        """
        sentences = self._split_into_sentences(text)

        if not sentences:
            return text.strip()

        # Find which sentence contains the citation
        cite_sentence_idx = self._find_sentence_idx_containing(sentences, cite_start)

        # Start with the citation sentence
        start_idx = cite_sentence_idx
        end_idx = cite_sentence_idx + 1

        # Expand until we have enough cleaned content
        while True:
            # Build context from selected sentences
            selected = sentences[start_idx:end_idx]
            context = " ".join(s[2] for s in selected)
            cleaned = clean_citation_markers(context)

            # Check if we have enough content after cleaning
            if len(cleaned) >= self.min_context_length:
                break

            # Try to expand
            expanded = False
            # Prefer previous sentence (more context before citation)
            if start_idx > 0:
                start_idx -= 1
                expanded = True
            elif end_idx < len(sentences):
                end_idx += 1
                expanded = True

            if not expanded:
                # Can't expand further, use what we have
                break

        # If still too short after max expansion, use full paragraph
        if len(cleaned) < self.min_context_length:
            context = text.strip()
            cleaned = clean_citation_markers(context)

        # Truncate if too long, keeping citation in context
        if len(context) > self.max_context_length:
            context = self._truncate_around_citation(context, cite_start, cite_end, sentences)

        return context.strip()

    def _truncate_around_citation(
        self,
        context: str,
        cite_start: int,
        cite_end: int,
        sentences: list[tuple[int, int, str]],
    ) -> str:
        """Truncate context while keeping the citation visible.

        Simple approach: keep sentences around citation until we hit max length.
        """
        if not sentences:
            return context[: self.max_context_length]

        cite_sent_idx = self._find_sentence_idx_containing(sentences, cite_start)

        # Start with just the citation sentence
        start_idx = cite_sent_idx
        end_idx = cite_sent_idx + 1

        # Try to add sentences alternately before/after until we'd exceed max
        while start_idx > 0 or end_idx < len(sentences):
            # Try adding sentence before
            if start_idx > 0:
                candidate = sentences[start_idx - 1 : end_idx]
                candidate_text = " ".join(s[2] for s in candidate)
                if len(candidate_text) <= self.max_context_length:
                    start_idx -= 1
                    continue

            # Try adding sentence after
            if end_idx < len(sentences):
                candidate = sentences[start_idx : end_idx + 1]
                candidate_text = " ".join(s[2] for s in candidate)
                if len(candidate_text) <= self.max_context_length:
                    end_idx += 1
                    continue

            # Can't add more without exceeding limit
            break

        # Build final context
        selected = sentences[start_idx:end_idx]
        result = " ".join(s[2] for s in selected)

        # Hard truncate if single sentence is still too long
        if len(result) > self.max_context_length:
            result = result[: self.max_context_length]

        return result

    def process_papers(
        self,
        jsonl_paths: list[Path | str],
        min_coverage: float = 0.3,
        min_reference_set_size: int = 1,
        skip_paper_ids: Optional[set[str]] = None,
        show_progress: bool = True,
    ) -> tuple[list[Paper], list[CitationContext], dict]:
        """Process unarXiv files and fetch metadata for cited works.

        Args:
            jsonl_paths: List of paths to unarXiv JSONL files
            min_coverage: Minimum fraction of citations with abstracts to include paper
            min_reference_set_size: Minimum number of resolved references to include paper
            skip_paper_ids: Paper IDs to skip (already processed)
            show_progress: Show progress bars

        Returns:
            Tuple of (corpus papers, citation contexts, stats dict)
        """
        skip_paper_ids = skip_paper_ids or set()

        # Collect all papers and their references
        source_papers: list[SourcePaper] = []
        all_openalex_ids: set[str] = set()

        print("Scanning unarXiv files...")
        for jsonl_path in jsonl_paths:
            path = Path(jsonl_path)
            if not path.exists():
                print(f"Warning: {path} not found, skipping")
                continue

            for paper in self.iter_papers(path):
                if paper.paper_id in skip_paper_ids:
                    continue
                if not paper.citation_contexts:
                    continue

                source_papers.append(paper)
                all_openalex_ids.update(paper.reference_openalex_ids)

        print(f"Found {len(source_papers)} papers with {len(all_openalex_ids)} unique references")

        # Fetch metadata for all references
        print("Fetching metadata from OpenAlex...")
        papers_by_id = self._fetch_papers_batch(list(all_openalex_ids), show_progress)

        # Filter papers by coverage and build output
        corpus_papers: list[Paper] = []
        citation_contexts: list[CitationContext] = []
        corpus_ids: set[str] = set()

        stats = {
            "papers_scanned": len(source_papers),
            "papers_included": 0,
            "papers_skipped_coverage": 0,
            "papers_skipped_ref_size": 0,
            "contexts_total": 0,
            "contexts_included": 0,
            "references_resolved": len(papers_by_id),
            "references_total": len(all_openalex_ids),
        }

        iterator = tqdm(source_papers, desc="Processing") if show_progress else source_papers

        for source in iterator:
            # Calculate coverage for this paper
            ref_ids = source.reference_openalex_ids
            resolved = [
                rid for rid in ref_ids if rid in papers_by_id and papers_by_id[rid].abstract
            ]
            coverage = len(resolved) / len(ref_ids) if ref_ids else 0

            if coverage < min_coverage:
                stats["papers_skipped_coverage"] += 1
                continue

            # Build reference set for this paper (only papers with abstracts)
            reference_set_ids = resolved

            # Check minimum reference set size
            if len(reference_set_ids) < min_reference_set_size:
                stats["papers_skipped_ref_size"] += 1
                continue

            stats["papers_included"] += 1

            # Add referenced papers to corpus
            for ref_id in reference_set_ids:
                if ref_id not in corpus_ids:
                    corpus_papers.append(papers_by_id[ref_id])
                    corpus_ids.add(ref_id)

            # Create citation contexts
            cite_num = 0
            for ctx in source.citation_contexts:
                openalex_id = ctx["openalex_id"]

                # Only include contexts where ground truth is in reference set
                if openalex_id not in reference_set_ids:
                    continue

                stats["contexts_total"] += 1
                cite_num += 1

                citation_contexts.append(
                    CitationContext(
                        id=f"{source.paper_id}_cite_{cite_num}",
                        local_context=ctx["text"],
                        narrow_context=ctx.get("narrow", ""),
                        broad_context=ctx.get("broad", ""),
                        section_context=ctx["section"],
                        global_context=source.title,
                        source_paper_id=source.paper_id,
                        ground_truth_ids=[openalex_id],
                        reference_set_ids=reference_set_ids,
                        mentioned_authors=ctx.get("mentioned_authors", []),
                        mentioned_years=ctx.get("mentioned_years", []),
                    )
                )
                stats["contexts_included"] += 1

        return corpus_papers, citation_contexts, stats

    def _fetch_papers_batch(
        self,
        openalex_ids: list[str],
        show_progress: bool = True,
    ) -> dict[str, Paper]:
        """Fetch papers from OpenAlex in batches.

        Args:
            openalex_ids: List of OpenAlex IDs
            show_progress: Show progress bar

        Returns:
            Dict mapping ID to Paper
        """
        papers_by_id: dict[str, Paper] = {}

        # Process in batches
        batch_size = 50
        iterator = range(0, len(openalex_ids), batch_size)
        if show_progress:
            total_batches = len(openalex_ids) // batch_size + 1
            iterator = tqdm(iterator, desc="Fetching metadata", total=total_batches)

        for i in iterator:
            batch = openalex_ids[i : i + batch_size]
            papers = self.openalex_client.get_works_batch(batch)

            for paper in papers:
                papers_by_id[paper.id] = paper

        return papers_by_id


# --- Re-exports for backward compatibility ---
# Lazy import via __getattr__ avoids a circular import:
# unarxiv_pipeline imports UnarXivProcessor from this module, so a top-level
# import here would create a cycle when unarxiv_pipeline is loaded first.
__all__ = [
    "BibEntry",
    "SourcePaper",
    "UnarXivProcessor",
]


def __getattr__(name: str):
    if name == "process_unarxiv_directory":
        from incite.corpus.unarxiv_pipeline import process_unarxiv_directory

        return process_unarxiv_directory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
