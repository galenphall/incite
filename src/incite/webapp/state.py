"""Session state and caching for the webapp."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from incite.corpus.loader import load_chunks, load_corpus, save_chunks, save_corpus
from incite.corpus.zotero_reader import find_zotero_data_dir, read_zotero_library
from incite.interfaces import Retriever
from incite.models import Chunk, Paper
from incite.retrieval.factory import (
    CHUNKING_STRATEGIES,
    DEFAULT_CHUNKING,
    DEFAULT_EMBEDDER,
    EMBEDDERS,
    build_chunk_index,
    build_index,
    create_paragraph_retriever,
    create_retriever,
    get_chunker,
)

if TYPE_CHECKING:
    from incite.embeddings.chunk_store import ChunkStore

# Version for chunk cache invalidation.
# Increment this when chunking logic changes to force cache rebuild.
# v1: Initial chunking
# v2: Improved bibliography filtering (section prefix + content detection)
# v3: Per-chunk bibliography entry filter (catches entries that bypass section detection)
# v4: Metadata prefix on chunks (title, author, year, journal) for better retrieval
CHUNK_CACHE_VERSION = 4


def _migrate_cache_dir() -> None:
    """Migrate ~/.mayacite/ to ~/.incite/ if needed (one-time rename)."""
    import logging

    old_dir = Path.home() / ".mayacite"
    new_dir = Path.home() / ".incite"
    if old_dir.is_dir() and not new_dir.exists():
        logging.getLogger(__name__).info("Migrating cache directory: %s -> %s", old_dir, new_dir)
        old_dir.rename(new_dir)


def get_cache_dir() -> Path:
    """Get the cache directory, creating if needed."""
    _migrate_cache_dir()
    cache_dir = Path.home() / ".incite"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_config_path() -> Path:
    """Get path to config file."""
    return get_cache_dir() / "config.json"


def get_config() -> dict:
    """Load configuration from JSON file."""
    config_path = get_config_path()

    # Migrate from old TOML config if it exists
    old_toml_path = get_cache_dir() / "config.toml"
    if not config_path.exists() and old_toml_path.exists():
        try:
            import tomllib

            with open(old_toml_path, "rb") as f:
                config = tomllib.load(f)
        except ImportError:
            try:
                import tomli

                with open(old_toml_path, "rb") as f:
                    config = tomli.load(f)
            except ImportError:
                config = None
        if config:
            save_config(config)
            return config

    if not config_path.exists():
        # Auto-detect Zotero directory
        detected_dir = find_zotero_data_dir()
        return {
            "zotero": {
                "data_dir": str(detected_dir) if detected_dir else "",
            },
            "paperpile": {
                "bibtex_url": "",
                "bibtex_path": "",
                "pdf_folder": "",
            },
            "webapp": {
                "default_method": "hybrid",
                "default_k": 5,
            },
        }

    with open(config_path) as f:
        return json.loads(f.read())


def save_config(config: dict) -> None:
    """Save configuration to JSON file."""
    config_path = get_config_path()
    config_path.write_text(json.dumps(config, indent=2))


def needs_refresh_from_zotero(zotero_dir: Path, corpus_path: Path) -> bool:
    """Check if corpus needs to be refreshed from Zotero database."""
    if not corpus_path.exists():
        return True

    db_path = zotero_dir / "zotero.sqlite"
    if not db_path.exists():
        return False

    # Refresh if database is newer than cached corpus
    return db_path.stat().st_mtime > corpus_path.stat().st_mtime


def load_zotero_direct(
    zotero_dir: Path,
    force_refresh: bool = False,
    progress_callback=None,
) -> list[Paper]:
    """Load papers directly from Zotero database.

    This is the new preferred method that reads directly from Zotero's
    SQLite database, providing exact PDF paths and eliminating the need
    for BibTeX exports.

    Args:
        zotero_dir: Path to Zotero data directory (contains zotero.sqlite)
        force_refresh: If True, ignore cache and re-read from database
        progress_callback: Optional callback for progress updates

    Returns:
        List of Paper objects with id set to Zotero item key
    """
    cache_dir = get_cache_dir()
    corpus_path = cache_dir / "zotero_corpus.jsonl"

    # Check if we can use cached version
    if not force_refresh and corpus_path.exists():
        if not needs_refresh_from_zotero(zotero_dir, corpus_path):
            if progress_callback:
                progress_callback("Loading cached corpus...")
            return load_corpus(corpus_path)

    # Read directly from Zotero database
    if progress_callback:
        progress_callback("Reading Zotero library...")

    papers = read_zotero_library(zotero_dir, show_progress=False)

    if progress_callback:
        progress_callback(f"Found {len(papers)} papers, saving to cache...")

    # Save to cache
    save_corpus(papers, corpus_path)

    return papers


def load_paperpile_direct(
    force_refresh: bool = False,
    progress_callback=None,
) -> list[Paper]:
    """Load papers from Paperpile BibTeX source.

    Uses the PaperpileSource class which handles BibTeX fetching, parsing,
    enrichment, and caching to ~/.incite/paperpile_corpus.jsonl.

    Args:
        force_refresh: If True, re-fetch BibTeX ignoring ETag cache
        progress_callback: Optional callback for progress updates

    Returns:
        List of Paper objects
    """
    from incite.corpus.paperpile_source import PaperpileSource

    config = get_config()
    pp_config = config.get("paperpile", {})

    bibtex_url = pp_config.get("bibtex_url") or None
    bibtex_path = pp_config.get("bibtex_path") or None
    pdf_folder = pp_config.get("pdf_folder") or None

    if not bibtex_url and not bibtex_path:
        raise ValueError("Paperpile not configured. Run `incite paperpile setup` first.")

    if progress_callback:
        progress_callback("Loading Paperpile library...")

    source = PaperpileSource(
        bibtex_url=bibtex_url,
        bibtex_path=Path(bibtex_path) if bibtex_path else None,
        pdf_folder=Path(pdf_folder) if pdf_folder else None,
    )

    papers = source.load_papers()

    if progress_callback:
        progress_callback(f"Loaded {len(papers)} papers from Paperpile")

    return papers


def get_retriever(
    papers: list[Paper],
    method: str = "hybrid",
    embedder_type: str = DEFAULT_EMBEDDER,
    force_rebuild: bool = False,
    progress_callback=None,
) -> Retriever:
    """Get or build retriever with cached index.

    Args:
        papers: List of Paper objects
        method: Retrieval method ("neural", "bm25", "hybrid")
        embedder_type: Which embedder to use ("minilm", "specter")
        force_rebuild: If True, rebuild index even if exists
        progress_callback: Optional callback for progress updates

    Returns:
        Configured Retriever
    """
    cache_dir = get_cache_dir()
    # Separate index per embedder type
    index_path = cache_dir / f"zotero_index_{embedder_type}"

    # BM25 doesn't need index
    if method == "bm25":
        if progress_callback:
            progress_callback("Building BM25 index...")
        return create_retriever(papers, method="bm25")

    # Check if we need to rebuild neural index
    need_rebuild = force_rebuild or not (index_path / "index.faiss").exists()

    # Consistency checks: paper count and embedder identity
    if not need_rebuild:
        id_map_path = index_path / "id_map.json"
        if id_map_path.exists():
            import json

            with open(id_map_path) as f:
                id_map = json.load(f)
            cached_count = len(id_map.get("id_to_idx", {}))
            if cached_count != len(papers):
                need_rebuild = True
            # Embedder identity check: rebuild if index was built with a different
            # embedder or if it predates embedder tracking (old index without the field)
            cached_embedder = id_map.get("embedder_type")
            if cached_embedder != embedder_type:
                if cached_embedder:
                    import logging as _logging

                    _logging.getLogger(__name__).warning(
                        "Index was built with '%s' but '%s' requested — rebuilding "
                        "(~15 min for large libraries)",
                        cached_embedder,
                        embedder_type,
                    )
                    if progress_callback:
                        progress_callback(
                            f"Embedder changed ({cached_embedder} -> {embedder_type}), "
                            "rebuilding index..."
                        )
                need_rebuild = True

    if need_rebuild:
        embedder_name = EMBEDDERS.get(embedder_type, {}).get("name", embedder_type)
        if progress_callback:
            n_papers = len(papers)
            est = "~15 min" if n_papers > 1000 else "~5 min" if n_papers > 200 else "a few min"
            progress_callback(f"Building {embedder_name} index for {n_papers} papers ({est})...")
        build_index(papers, index_path, embedder_type=embedder_type, show_progress=False)

    if progress_callback:
        progress_callback(f"Creating {method} retriever...")

    return create_retriever(
        papers,
        method=method,
        embedder_type=embedder_type,
        index_path=index_path,
        show_progress=False,
    )


def get_paper_dict(papers: list[Paper]) -> dict[str, Paper]:
    """Create a dictionary mapping paper IDs to Paper objects."""
    return {p.id: p for p in papers}


def has_chunks(papers: list[Paper]) -> bool:
    """Check if any papers have full text for paragraph-level retrieval."""
    return any(p.full_text or p.paragraphs for p in papers)


def extract_and_save_pdfs(
    papers: list[Paper],
    progress_callback=None,
    max_workers: int = 8,
) -> dict:
    """Extract text from PDFs using Paper.source_file paths.

    Uses parallel processing for faster extraction.

    Args:
        papers: List of Paper objects to update (should have source_file from Zotero)
        progress_callback: Optional callback for progress updates.
            Called with (current, total, status_message) for progress bar support.
        max_workers: Number of parallel workers for PDF extraction

    Returns:
        Stats dict with extraction counts
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from incite.corpus.pdf_extractor import extract_pdf_text

    cache_dir = get_cache_dir()
    corpus_path = cache_dir / "zotero_corpus.jsonl"

    # Get papers with PDF paths
    papers_with_pdfs = [p for p in papers if p.source_file and Path(p.source_file).exists()]
    total_with_path = len(papers_with_pdfs)

    if progress_callback:
        progress_callback(0, total_with_path, f"Extracting PDFs ({max_workers} workers)...")

    # Extract PDFs in parallel
    results_map: dict[str, tuple] = {}  # paper_id -> (full_text, paragraphs)

    def extract_single(paper: Paper) -> tuple[str, str, list[str]]:
        """Extract a single PDF, returns (paper_id, full_text, paragraphs)."""
        result = extract_pdf_text(paper.source_file)
        return (paper.id, result.full_text or "", result.paragraphs or [])

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_single, p): p for p in papers_with_pdfs}

        for future in as_completed(futures):
            paper_id, full_text, paragraphs = future.result()
            if full_text:
                results_map[paper_id] = (full_text, paragraphs)

            completed += 1
            if progress_callback:
                progress_callback(
                    completed,
                    total_with_path,
                    f"Extracted {len(results_map)} of {completed} PDFs...",
                )

    # Update papers with extracted text
    for paper in papers:
        if paper.id in results_map:
            paper.full_text, paper.paragraphs = results_map[paper.id]

    if progress_callback:
        progress_callback(total_with_path, total_with_path, "Saving corpus...")

    # Save updated corpus
    save_corpus(papers, corpus_path)

    # Back up and clear cached chunks (they need to be rebuilt from new text).
    # Backups go to *.bak so they can be recovered if something goes wrong.
    import shutil

    chunks_path = cache_dir / "zotero_chunks.jsonl"
    if chunks_path.exists():
        shutil.copy2(chunks_path, chunks_path.with_suffix(".jsonl.bak"))
        chunks_path.unlink()

    for strategy in CHUNKING_STRATEGIES:
        strategy_chunks_path = cache_dir / f"zotero_chunks_{strategy}.jsonl"
        if strategy_chunks_path.exists():
            shutil.copy2(
                strategy_chunks_path,
                strategy_chunks_path.with_suffix(".jsonl.bak"),
            )
            strategy_chunks_path.unlink()

    # Clear FAISS chunk indexes (they depend on the chunks that were just invalidated)
    for embedder_key in EMBEDDERS:
        index_dir = cache_dir / f"zotero_chunks_{embedder_key}"
        if index_dir.exists():
            shutil.rmtree(index_dir)

    return {
        "extracted": len(results_map),
        "total": len(papers),
        "found_pdfs": total_with_path,
        "with_path": total_with_path,
    }


def _get_chunk_cache_version(cache_dir: Path, strategy: str) -> int:
    """Get the version number of a cached chunk file."""
    version_path = cache_dir / f"zotero_chunks_{strategy}.version"
    if not version_path.exists():
        return 0  # No version file means old cache
    try:
        return int(version_path.read_text().strip())
    except (ValueError, OSError):
        return 0


def _set_chunk_cache_version(cache_dir: Path, strategy: str, version: int) -> None:
    """Set the version number for a chunk cache."""
    version_path = cache_dir / f"zotero_chunks_{strategy}.version"
    version_path.write_text(str(version))


def load_zotero_chunks(
    papers: list[Paper],
    force_rebuild: bool = False,
    chunking_strategy: str = DEFAULT_CHUNKING,
    progress_callback=None,
) -> list[Chunk]:
    """Load or build chunks from papers.

    Args:
        papers: List of Paper objects (should have full_text or paragraphs)
        force_rebuild: If True, rebuild chunks even if cached
        chunking_strategy: Which chunking strategy to use (from CHUNKING_STRATEGIES)
        progress_callback: Optional callback for progress updates

    Returns:
        List of Chunk objects
    """
    cache_dir = get_cache_dir()
    # Include strategy in cache path so different strategies don't collide
    chunks_path = cache_dir / f"zotero_chunks_{chunking_strategy}.jsonl"

    # Check cache version - re-filter or rebuild if outdated
    cached_version = _get_chunk_cache_version(cache_dir, chunking_strategy)
    version_outdated = cached_version < CHUNK_CACHE_VERSION

    if version_outdated and not force_rebuild and chunks_path.exists():
        # Version bump with existing cache: re-filter cached chunks rather than
        # doing a full rebuild (which would lose PDF-extracted text if papers
        # only have abstracts loaded).
        if progress_callback:
            progress_callback(
                f"Chunk cache outdated (v{cached_version} -> v{CHUNK_CACHE_VERSION}), "
                "re-filtering..."
            )
        paper_dict = {p.id: p for p in papers} if papers else None
        chunks = _refilter_cached_chunks(chunks_path, papers=paper_dict)
        save_chunks(chunks, chunks_path)
        _set_chunk_cache_version(cache_dir, chunking_strategy, CHUNK_CACHE_VERSION)
        return chunks

    if version_outdated:
        force_rebuild = True

    # Check if we can use cached version
    if not force_rebuild and chunks_path.exists():
        if progress_callback:
            progress_callback("Loading cached chunks...")
        return load_chunks(chunks_path)

    # Get the chunking function from factory
    chunker = get_chunker(chunking_strategy)

    # Create chunks from papers
    if progress_callback:
        strategy_name = CHUNKING_STRATEGIES.get(chunking_strategy, {}).get(
            "name", chunking_strategy
        )
        progress_callback(f"Creating chunks ({strategy_name})...")

    chunks = chunker(papers, show_progress=False)

    # SAFETY: Refuse to overwrite a large cache with a much smaller one.
    # This prevents catastrophic data loss when papers are loaded without
    # PDF text (e.g., metadata-only from Zotero) — the rebuild would produce
    # only abstract chunks, destroying PDF-extracted/GROBID chunks.
    if chunks_path.exists():
        old_count = sum(1 for _ in open(chunks_path))
        if old_count > 1000 and len(chunks) < old_count * 0.5:
            import logging

            logging.getLogger(__name__).error(
                "SAFETY: Refusing to overwrite %d cached chunks with only %d new chunks "
                "(>50%% reduction). This usually means papers were loaded without PDF text. "
                "Use force_rebuild=True with PDF-extracted papers, or delete the cache "
                "file manually if this is intentional.",
                old_count,
                len(chunks),
            )
            # Return the existing cache instead of destroying it
            return load_chunks(chunks_path)

    # Save to cache
    if progress_callback:
        progress_callback("Saving chunks to cache...")
    save_chunks(chunks, chunks_path)

    # Save version marker so we know when to invalidate
    _set_chunk_cache_version(cache_dir, chunking_strategy, CHUNK_CACHE_VERSION)

    return chunks


def _refilter_cached_chunks(
    chunks_path: Path,
    papers: Optional[dict[str, Paper]] = None,
) -> list[Chunk]:
    """Re-filter existing cached chunks through updated chunking filters.

    Used when CHUNK_CACHE_VERSION bumps for filter-only changes (e.g., adding
    bibliography detection, metadata prefixes). Avoids a full rebuild which
    would lose PDF-extracted text when papers are loaded with abstracts only.

    Args:
        chunks_path: Path to the cached chunks JSONL file.
        papers: Optional dict mapping paper_id -> Paper. If provided, chunks
            without context_text will get a metadata prefix added (v4 upgrade).
    """
    from incite.corpus.chunking import (
        _build_paper_metadata_prefix,
        is_bibliography,
    )

    old_chunks = load_chunks(chunks_path)
    filtered = [c for c in old_chunks if not is_bibliography(c)]
    removed = len(old_chunks) - len(filtered)

    # v4 upgrade: add metadata prefix to chunks that lack context_text
    prefixed = 0
    if papers:
        # Pre-compute prefixes per paper to avoid repeated work
        prefix_cache: dict[str, str] = {}
        for chunk in filtered:
            if chunk.context_text is None and chunk.paper_id in papers:
                if chunk.paper_id not in prefix_cache:
                    prefix_cache[chunk.paper_id] = _build_paper_metadata_prefix(
                        papers[chunk.paper_id]
                    )
                chunk.context_text = prefix_cache[chunk.paper_id]
                prefixed += 1

    if removed > 0 or prefixed > 0:
        import logging

        logger = logging.getLogger(__name__)
        if removed > 0:
            logger.info(
                "Re-filtered chunks: %d -> %d (removed %d bibliography entries)",
                len(old_chunks),
                len(filtered),
                removed,
            )
        if prefixed > 0:
            logger.info(
                "Added metadata prefix to %d chunks (of %d total)",
                prefixed,
                len(filtered),
            )
    return filtered


def enrich_chunks_with_context(
    chunks: list[Chunk],
    papers: list[Paper],
    progress_callback=None,
) -> dict:
    """Enrich chunks with LLM-generated contextual information.

    This implements Anthropic's contextual retrieval approach, which can
    improve retrieval by 35-49% by prepending situating context to each chunk.

    Requires ANTHROPIC_API_KEY environment variable.

    Args:
        chunks: List of Chunk objects to enrich
        papers: List of Paper objects (for document context)
        progress_callback: Optional callback for progress updates

    Returns:
        Stats dict with enrichment counts
    """
    from incite.corpus.contextual_enrichment import enrich_chunks_sync

    if progress_callback:
        progress_callback("Enriching chunks with LLM context...")

    paper_dict = {p.id: p for p in papers}

    stats = enrich_chunks_sync(
        papers=paper_dict,
        chunks=chunks,
        show_progress=False,
        skip_existing=True,
    )

    return stats


def get_paragraph_retriever(
    chunks: list[Chunk],
    papers: list[Paper],
    method: str = "hybrid",
    embedder_type: str = "nomic",
    force_rebuild: bool = False,
    progress_callback=None,
) -> Retriever:
    """Get or build paragraph-level retriever with cached index.

    Args:
        chunks: List of Chunk objects
        papers: List of Paper objects (for BM25 and paper info)
        method: Retrieval method ("neural" or "hybrid")
        embedder_type: Which embedder to use ("nomic" recommended for 8K context)
        force_rebuild: If True, rebuild index even if exists
        progress_callback: Optional callback for progress updates

    Returns:
        Configured Retriever (ParagraphRetriever or HybridParagraphRetriever)
    """
    cache_dir = get_cache_dir()
    # Separate index per embedder type
    index_path = cache_dir / f"zotero_chunks_{embedder_type}"

    # Check if we need to rebuild
    need_rebuild = force_rebuild or not (index_path / "index.faiss").exists()

    # Consistency checks: chunk count and embedder identity
    if not need_rebuild:
        id_map_path = index_path / "id_map.json"
        if id_map_path.exists():
            import json

            with open(id_map_path) as f:
                id_map = json.load(f)
            cached_count = len(id_map.get("id_to_idx", {}))
            if cached_count != len(chunks):
                need_rebuild = True
            cached_embedder = id_map.get("embedder_type")
            if cached_embedder != embedder_type:
                need_rebuild = True

    if need_rebuild:
        embedder_name = EMBEDDERS.get(embedder_type, {}).get("name", embedder_type)
        if progress_callback:
            progress_callback(f"Building {embedder_name} chunk index ({len(chunks)} chunks)...")
        build_chunk_index(
            chunks,
            index_path,
            embedder_type=embedder_type,
            show_progress=False,
            progress_callback=progress_callback,
        )

    if progress_callback:
        progress_callback(f"Creating paragraph {method} retriever...")

    return create_paragraph_retriever(
        chunks=chunks,
        papers=papers,
        embedder_type=embedder_type,
        index_path=index_path,
        method=method if method != "bm25" else "hybrid",
        show_progress=False,
    )


def get_evidence_store(
    papers: list[Paper],
    embedder_type: str = DEFAULT_EMBEDDER,
    chunking_strategy: str = DEFAULT_CHUNKING,
    force_rebuild: bool = False,
) -> tuple[dict[str, Chunk], "ChunkStore"]:
    """Load chunk store for evidence snippet lookup (no full retriever).

    Used by paper-mode retrieval to attach paragraph evidence to results.
    Builds and caches the chunk index on first call.

    Args:
        papers: List of Paper objects (should have full_text or paragraphs)
        embedder_type: Which embedder to use (must match paper retriever)
        chunking_strategy: Chunking strategy for splitting papers
        force_rebuild: If True, rebuild index even if cached

    Returns:
        Tuple of (chunk_dict, chunk_store) for evidence lookup.
        chunk_dict maps chunk_id -> Chunk for text retrieval.
    """
    from incite.embeddings.chunk_store import ChunkStore
    from incite.embeddings.chunk_store import build_chunk_index as build_chunk_store
    from incite.retrieval.factory import get_embedder

    cache_dir = get_cache_dir()

    # Load chunks (uses existing cache)
    chunks = load_zotero_chunks(papers, chunking_strategy=chunking_strategy)
    chunk_dict = {c.id: c for c in chunks}

    # Load or build chunk index (same cache path as paragraph retriever)
    index_path = cache_dir / f"zotero_chunks_{embedder_type}"
    need_rebuild = force_rebuild or not (index_path / "index.faiss").exists()

    if not need_rebuild:
        # Consistency checks: chunk count and embedder identity
        id_map_path = index_path / "id_map.json"
        if id_map_path.exists():
            import json as _json

            with open(id_map_path) as f:
                id_map = _json.load(f)
            if len(id_map.get("id_to_idx", {})) != len(chunks):
                need_rebuild = True
            cached_embedder = id_map.get("embedder_type")
            if cached_embedder != embedder_type:
                need_rebuild = True

    if need_rebuild:
        embedder = get_embedder(embedder_type)
        chunk_store = build_chunk_store(
            chunks, embedder, output_path=index_path, show_progress=True
        )
    else:
        chunk_store = ChunkStore()
        chunk_store.load(str(index_path))

    return chunk_dict, chunk_store
