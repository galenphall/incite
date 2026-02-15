"""Direct Zotero SQLite database reader.

This module reads papers directly from Zotero's SQLite database,
eliminating the need for BibTeX exports and providing exact PDF paths.
"""

import platform
import sqlite3
from pathlib import Path
from typing import Optional

from incite.models import Paper


def find_zotero_data_dir() -> Optional[Path]:
    """Auto-detect Zotero data directory.

    Checks common default locations:
    - macOS/Linux: ~/Zotero
    - Windows: %APPDATA%/Zotero/Zotero/

    Returns:
        Path to Zotero data directory if found (contains zotero.sqlite), else None
    """
    possible_paths = []

    system = platform.system()
    home = Path.home()

    if system == "Darwin" or system == "Linux":
        possible_paths.append(home / "Zotero")
    elif system == "Windows":
        appdata = Path.home() / "AppData" / "Roaming"
        possible_paths.append(appdata / "Zotero" / "Zotero")

    # Also check the default for all platforms
    possible_paths.append(home / "Zotero")

    for path in possible_paths:
        if path.exists() and (path / "zotero.sqlite").exists():
            return path

    return None


def resolve_attachment_path(data_dir: Path, attachment_key: str, path_value: str) -> Optional[Path]:
    """Resolve Zotero attachment path to actual file path.

    Zotero attachment paths can be:
    - 'storage:filename.pdf' -> {data_dir}/storage/{attachment_key}/{filename}
    - Absolute path (linked files)
    - Relative path

    Args:
        data_dir: Zotero data directory
        attachment_key: The item key of the attachment (8 chars)
        path_value: The path string from itemAttachments table

    Returns:
        Resolved Path if file exists, else None
    """
    if not path_value:
        return None

    # Most common: storage:filename.pdf
    if path_value.startswith("storage:"):
        filename = path_value[8:]  # Remove 'storage:' prefix
        resolved = data_dir / "storage" / attachment_key / filename
        return resolved if resolved.exists() else None

    # Absolute path (linked file)
    abs_path = Path(path_value)
    if abs_path.is_absolute():
        return abs_path if abs_path.exists() else None

    # Relative path
    rel_path = data_dir / path_value
    return rel_path if rel_path.exists() else None


def _parse_year(date_str: str) -> Optional[int]:
    """Extract year from Zotero date field.

    Zotero stores dates in various formats like:
    - "2024"
    - "2024-01-15"
    - "January 2024"
    - "2024/01/15"
    """
    if not date_str:
        return None

    import re

    # Try to find a 4-digit year
    match = re.search(r"\b(19|20)\d{2}\b", date_str)
    if match:
        return int(match.group())
    return None


def read_zotero_library(
    data_dir: Path,
    show_progress: bool = True,
) -> list[Paper]:
    """Read papers directly from Zotero SQLite database.

    Queries the Zotero database to get:
    - Paper metadata (title, abstract, year, DOI)
    - Authors
    - PDF attachment paths

    Args:
        data_dir: Path to Zotero data directory (contains zotero.sqlite)
        show_progress: Whether to show progress bar

    Returns:
        List of Paper objects with id set to Zotero item key

    Raises:
        FileNotFoundError: If zotero.sqlite doesn't exist
        sqlite3.OperationalError: If database is locked (Zotero may be running)
    """
    db_path = data_dir / "zotero.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Zotero database not found at {db_path}")

    try:
        # Use immutable mode to read even when Zotero has the db open
        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro&immutable=1",
            uri=True,
            timeout=10.0,
        )
        conn.row_factory = sqlite3.Row
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower():
            raise sqlite3.OperationalError(
                "Database is locked. Zotero may be running - this is OK, "
                "the app will use immutable read mode."
            ) from e
        raise

    try:
        papers = _read_items(conn, data_dir, show_progress)
    finally:
        conn.close()

    return papers


def _read_items(
    conn: sqlite3.Connection,
    data_dir: Path,
    show_progress: bool,
) -> list[Paper]:
    """Read all paper items from database."""
    cursor = conn.cursor()

    # Build libraryID → groupID mapping for Zotero URIs
    lib_to_group: dict[int, int] = {}
    try:
        cursor.execute("SELECT libraryID, groupID FROM groups")
        lib_to_group = {row["libraryID"]: row["groupID"] for row in cursor.fetchall()}
    except Exception:
        pass  # groups table may not exist in older Zotero versions

    # Get all paper-type items (not attachments, notes, or deleted)
    cursor.execute("""
        SELECT i.itemID, i.key, i.libraryID, it.typeName
        FROM items i
        JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
        WHERE it.typeName IN (
            'journalArticle', 'conferencePaper', 'preprint',
            'book', 'bookSection', 'report', 'thesis', 'manuscript'
        )
        AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
    """)

    items = cursor.fetchall()

    if show_progress:
        try:
            from tqdm import tqdm

            items = tqdm(items, desc="Reading Zotero library")
        except ImportError:
            pass

    papers = []
    for row in items:
        item_id = row["itemID"]
        item_key = row["key"]
        library_id = row["libraryID"]

        # Get metadata fields
        metadata = _get_item_metadata(cursor, item_id)

        # Skip items without titles
        title = metadata.get("title", "").strip()
        if not title:
            continue

        # Get authors
        authors = _get_item_authors(cursor, item_id)

        # Get PDF attachment path
        source_file = _get_pdf_attachment(cursor, item_id, data_dir)

        # Build Zotero URI
        group_id = lib_to_group.get(library_id)
        if group_id is not None:
            zotero_uri = f"zotero://select/groups/{group_id}/items/{item_key}"
        else:
            zotero_uri = f"zotero://select/items/0_{item_key}"

        paper = Paper(
            id=item_key,  # Use Zotero's 8-char key as ID
            title=title,
            abstract=metadata.get("abstractNote", "") or "",
            authors=authors,
            year=_parse_year(metadata.get("date", "")),
            doi=metadata.get("DOI"),
            bibtex_key=None,  # Will be generated if needed
            journal=metadata.get("publicationTitle") or metadata.get("proceedingsTitle"),
            source_file=str(source_file) if source_file else None,
            zotero_uri=zotero_uri,
        )
        papers.append(paper)

    return papers


def _get_item_metadata(cursor: sqlite3.Cursor, item_id: int) -> dict[str, str]:
    """Get all metadata fields for an item."""
    cursor.execute(
        """
        SELECT f.fieldName, idv.value
        FROM itemData id
        JOIN itemDataValues idv ON id.valueID = idv.valueID
        JOIN fields f ON id.fieldID = f.fieldID
        WHERE id.itemID = ?
    """,
        (item_id,),
    )

    return {row["fieldName"]: row["value"] for row in cursor.fetchall()}


def _get_item_authors(cursor: sqlite3.Cursor, item_id: int) -> list[str]:
    """Get author names for an item.

    Returns names in "Last, First" format or just name if single field.
    """
    cursor.execute(
        """
        SELECT c.firstName, c.lastName
        FROM itemCreators ic
        JOIN creators c ON ic.creatorID = c.creatorID
        JOIN creatorTypes ct ON ic.creatorTypeID = ct.creatorTypeID
        WHERE ic.itemID = ?
        AND ct.creatorType IN ('author', 'contributor', 'editor')
        ORDER BY ic.orderIndex
    """,
        (item_id,),
    )

    authors = []
    for row in cursor.fetchall():
        first = (row["firstName"] or "").strip()
        last = (row["lastName"] or "").strip()

        if first and last:
            authors.append(f"{last}, {first}")
        elif last:
            authors.append(last)
        elif first:
            authors.append(first)

    return authors


def _get_pdf_attachment(cursor: sqlite3.Cursor, item_id: int, data_dir: Path) -> Optional[Path]:
    """Get the PDF attachment path for an item.

    Prefers the largest PDF if multiple exist (usually the main paper vs supplements).
    """
    # Get all PDF attachments for this item
    cursor.execute(
        """
        SELECT ia.path, i.key as attachmentKey
        FROM itemAttachments ia
        JOIN items i ON ia.itemID = i.itemID
        WHERE ia.parentItemID = ?
        AND (ia.contentType = 'application/pdf' OR ia.path LIKE '%.pdf')
    """,
        (item_id,),
    )

    attachments = cursor.fetchall()

    # Resolve paths and find actual files
    valid_pdfs: list[tuple[Path, int]] = []
    for row in attachments:
        if row["path"]:
            resolved = resolve_attachment_path(data_dir, row["attachmentKey"], row["path"])
            if resolved and resolved.exists():
                try:
                    size = resolved.stat().st_size
                    valid_pdfs.append((resolved, size))
                except OSError:
                    pass

    if not valid_pdfs:
        return None

    # Return the largest PDF (main paper vs supplement)
    valid_pdfs.sort(key=lambda x: x[1], reverse=True)
    return valid_pdfs[0][0]


class ZoteroSource:
    """CorpusSource implementation that reads from a local Zotero SQLite database.

    Wraps read_zotero_library() with caching to ~/.incite/zotero_corpus.jsonl.
    Satisfies the CorpusSource protocol via structural typing.
    """

    name: str = "zotero"

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize ZoteroSource.

        Args:
            data_dir: Path to Zotero data directory (contains zotero.sqlite).
                If None, auto-detects using find_zotero_data_dir().
        """
        if data_dir is None:
            detected = find_zotero_data_dir()
            if detected is None:
                raise ValueError(
                    "Could not auto-detect Zotero directory. Please provide data_dir parameter."
                )
            data_dir = detected
        self.data_dir = Path(data_dir)

    def load_papers(self) -> list[Paper]:
        """Load papers from Zotero, using cache if fresh."""
        from incite.corpus.loader import load_corpus, save_corpus

        cache_dir = Path.home() / ".incite"
        cache_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = cache_dir / "zotero_corpus.jsonl"

        if corpus_path.exists() and not self.needs_refresh():
            return load_corpus(corpus_path)

        papers = read_zotero_library(self.data_dir, show_progress=False)

        save_corpus(papers, corpus_path)
        return papers

    def needs_refresh(self) -> bool:
        """Check if Zotero DB has been modified since the cached corpus."""
        cache_dir = Path.home() / ".incite"
        corpus_path = cache_dir / "zotero_corpus.jsonl"

        if not corpus_path.exists():
            return True

        db_path = self.data_dir / "zotero.sqlite"
        if not db_path.exists():
            return False

        return db_path.stat().st_mtime > corpus_path.stat().st_mtime

    def cache_key(self) -> str:
        """Return cache key based on the Zotero data directory path."""
        return f"zotero_{self.data_dir.name}"


def get_library_stats(data_dir: Path) -> dict:
    """Get statistics about a Zotero library.

    Useful for displaying in webapp UI.

    Args:
        data_dir: Path to Zotero data directory

    Returns:
        Dict with counts of papers, papers with PDFs, papers with abstracts
    """
    db_path = data_dir / "zotero.sqlite"
    if not db_path.exists():
        return {"error": "Database not found"}

    try:
        # Use timeout and immutable mode to avoid lock conflicts with running Zotero
        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro&immutable=1",
            uri=True,
            timeout=5.0,
        )
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower():
            return {"error": "Database locked - Zotero may be running"}
        return {"error": f"Cannot open database: {e}"}

    try:
        cursor = conn.cursor()

        # Count paper-type items
        cursor.execute("""
            SELECT COUNT(*)
            FROM items i
            JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
            WHERE it.typeName IN (
                'journalArticle', 'conferencePaper', 'preprint',
                'book', 'bookSection', 'report', 'thesis', 'manuscript'
            )
            AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
        """)
        total_papers = cursor.fetchone()[0]

        # Count papers with abstracts
        cursor.execute("""
            SELECT COUNT(DISTINCT id.itemID)
            FROM itemData id
            JOIN fields f ON id.fieldID = f.fieldID
            JOIN items i ON id.itemID = i.itemID
            JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
            WHERE f.fieldName = 'abstractNote'
            AND it.typeName IN (
                'journalArticle', 'conferencePaper', 'preprint',
                'book', 'bookSection', 'report', 'thesis', 'manuscript'
            )
            AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
        """)
        with_abstracts = cursor.fetchone()[0]

        # Count papers with PDF attachments
        cursor.execute("""
            SELECT COUNT(DISTINCT ia.parentItemID)
            FROM itemAttachments ia
            JOIN items i ON ia.parentItemID = i.itemID
            JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
            WHERE (ia.contentType = 'application/pdf' OR ia.path LIKE '%.pdf')
            AND it.typeName IN (
                'journalArticle', 'conferencePaper', 'preprint',
                'book', 'bookSection', 'report', 'thesis', 'manuscript'
            )
            AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
        """)
        with_pdfs = cursor.fetchone()[0]

        return {
            "total_papers": total_papers,
            "with_abstracts": with_abstracts,
            "with_pdfs": with_pdfs,
        }
    finally:
        conn.close()
