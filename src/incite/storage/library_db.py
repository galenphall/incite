"""SQLite-backed storage for papers, chunks, and embeddings.

Provides a lightweight local database at ~/.incite/library.db that stores
papers, chunks, and their embeddings as BLOBs. FAISS indexes are built
in-memory from stored embeddings on startup.

This avoids the JSONL + FAISS file sprawl and supports incremental updates.
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

import numpy as np

from incite.models import Chunk, Paper

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384
EMBEDDING_BYTES = EMBEDDING_DIM * 4  # float32

SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT DEFAULT '',
    authors TEXT DEFAULT '[]',
    year INTEGER,
    doi TEXT,
    journal TEXT,
    source_file TEXT,
    zotero_uri TEXT,
    embedding BLOB,
    embedding_text TEXT,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    text TEXT NOT NULL,
    section TEXT,
    char_offset INTEGER DEFAULT 0,
    source TEXT,
    context_text TEXT,
    embedding BLOB,
    embedding_text TEXT,
    FOREIGN KEY (paper_id) REFERENCES papers(id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
"""


class LibraryDB:
    """SQLite storage for papers, chunks, and embeddings."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path.home() / ".incite" / "library.db"
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _ensure_schema(self):
        """Create tables if they don't exist, run migrations if needed."""
        self.conn.executescript(_SCHEMA_SQL)
        row = self.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self.conn.commit()

    # ── Papers ──────────────────────────────────────────────────────────

    def upsert_papers(self, papers: list[Paper]) -> list[str]:
        """Insert or update papers. Returns IDs of papers needing embedding."""
        now = time.time()
        needs_embed = []

        for p in papers:
            authors_json = json.dumps(p.authors)
            embed_text = p.to_embedding_text()

            # Check if embedding is already up-to-date
            existing = self.conn.execute(
                "SELECT embedding_text FROM papers WHERE id = ?", (p.id,)
            ).fetchone()

            if existing and existing["embedding_text"] == embed_text:
                # Metadata may have changed but embedding text is same — skip re-embed
                self.conn.execute(
                    """UPDATE papers SET title=?, abstract=?, authors=?, year=?, doi=?,
                       journal=?, source_file=?, zotero_uri=?, updated_at=?
                       WHERE id=?""",
                    (
                        p.title,
                        p.abstract,
                        authors_json,
                        p.year,
                        p.doi,
                        p.journal,
                        p.source_file,
                        p.zotero_uri,
                        now,
                        p.id,
                    ),
                )
            else:
                # New paper or embedding text changed — clear embedding
                self.conn.execute(
                    """INSERT INTO papers (id, title, abstract, authors, year, doi,
                       journal, source_file, zotero_uri, embedding, embedding_text, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                       ON CONFLICT(id) DO UPDATE SET
                       title=excluded.title, abstract=excluded.abstract, authors=excluded.authors,
                       year=excluded.year, doi=excluded.doi, journal=excluded.journal,
                       source_file=excluded.source_file, zotero_uri=excluded.zotero_uri,
                       embedding=NULL, embedding_text=excluded.embedding_text,
                       updated_at=excluded.updated_at""",
                    (
                        p.id,
                        p.title,
                        p.abstract,
                        authors_json,
                        p.year,
                        p.doi,
                        p.journal,
                        p.source_file,
                        p.zotero_uri,
                        embed_text,
                        now,
                    ),
                )
                needs_embed.append(p.id)

        self.conn.commit()
        return needs_embed

    def upsert_chunks(self, chunks: list[Chunk]) -> list[str]:
        """Insert or update chunks. Returns IDs of chunks needing embedding."""
        needs_embed = []

        for c in chunks:
            embed_text = c.to_embedding_text()

            existing = self.conn.execute(
                "SELECT embedding_text FROM chunks WHERE id = ?", (c.id,)
            ).fetchone()

            if existing and existing["embedding_text"] == embed_text:
                # Update non-embedding fields only
                self.conn.execute(
                    """UPDATE chunks SET paper_id=?, text=?, section=?,
                       char_offset=?, source=?, context_text=?
                       WHERE id=?""",
                    (c.paper_id, c.text, c.section, c.char_offset, c.source, c.context_text, c.id),
                )
            else:
                self.conn.execute(
                    """INSERT INTO chunks (id, paper_id, text, section, char_offset,
                       source, context_text, embedding, embedding_text)
                       VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)
                       ON CONFLICT(id) DO UPDATE SET
                       paper_id=excluded.paper_id, text=excluded.text,
                       section=excluded.section, char_offset=excluded.char_offset,
                       source=excluded.source, context_text=excluded.context_text,
                       embedding=NULL, embedding_text=excluded.embedding_text""",
                    (
                        c.id,
                        c.paper_id,
                        c.text,
                        c.section,
                        c.char_offset,
                        c.source,
                        c.context_text,
                        embed_text,
                    ),
                )
                needs_embed.append(c.id)

        self.conn.commit()
        return needs_embed

    # ── Embeddings ──────────────────────────────────────────────────────

    def store_embeddings(self, ids: list[str], embeddings: np.ndarray, table: str):
        """Store embedding vectors as BLOBs.

        Args:
            ids: Row IDs matching the embeddings.
            embeddings: (N, 384) float32 array.
            table: "papers" or "chunks".
        """
        if table not in ("papers", "chunks"):
            raise ValueError(f"Invalid table: {table}")
        if len(ids) != len(embeddings):
            raise ValueError(f"Mismatch: {len(ids)} ids vs {len(embeddings)} embeddings")

        for row_id, vec in zip(ids, embeddings):
            blob = vec.astype(np.float32).tobytes()
            self.conn.execute(
                f"UPDATE {table} SET embedding = ? WHERE id = ?",  # noqa: S608
                (blob, row_id),
            )
        self.conn.commit()

    def load_paper_embeddings(self) -> tuple[list[str], np.ndarray]:
        """Load all paper embeddings for FAISS index building.

        Returns:
            Tuple of (paper_ids, embeddings array of shape (N, 384)).
            Only includes papers that have embeddings.
        """
        rows = self.conn.execute(
            "SELECT id, embedding FROM papers WHERE embedding IS NOT NULL"
        ).fetchall()
        if not rows:
            return [], np.array([]).reshape(0, EMBEDDING_DIM).astype(np.float32)

        ids = [r["id"] for r in rows]
        vecs = np.array(
            [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows],
            dtype=np.float32,
        )
        return ids, vecs

    def load_chunk_embeddings(self) -> tuple[list[str], np.ndarray]:
        """Load all chunk embeddings for FAISS index building.

        Returns:
            Tuple of (chunk_ids, embeddings array of shape (N, 384)).
            Only includes chunks that have embeddings.
        """
        rows = self.conn.execute(
            "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL"
        ).fetchall()
        if not rows:
            return [], np.array([]).reshape(0, EMBEDDING_DIM).astype(np.float32)

        ids = [r["id"] for r in rows]
        vecs = np.array(
            [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows],
            dtype=np.float32,
        )
        return ids, vecs

    # ── Queries ─────────────────────────────────────────────────────────

    def get_papers(self) -> list[Paper]:
        """Load all papers as Paper objects."""
        rows = self.conn.execute("SELECT * FROM papers").fetchall()
        return [self._row_to_paper(r) for r in rows]

    def get_chunks(self, paper_ids: Optional[list[str]] = None) -> list[Chunk]:
        """Load chunks, optionally filtered by paper IDs."""
        if paper_ids is not None:
            placeholders = ",".join("?" for _ in paper_ids)
            rows = self.conn.execute(
                f"SELECT * FROM chunks WHERE paper_id IN ({placeholders})",  # noqa: S608
                paper_ids,
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM chunks").fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def needs_embedding(self, table: str) -> list[str]:
        """Return IDs where embedding is NULL."""
        if table not in ("papers", "chunks"):
            raise ValueError(f"Invalid table: {table}")
        rows = self.conn.execute(
            f"SELECT id FROM {table} WHERE embedding IS NULL"  # noqa: S608
        ).fetchall()
        return [r["id"] for r in rows]

    def get_embedding_texts(self, ids: list[str], table: str) -> dict[str, str]:
        """Get embedding_text for given IDs (for batch embedding)."""
        if table not in ("papers", "chunks"):
            raise ValueError(f"Invalid table: {table}")
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"SELECT id, embedding_text FROM {table} WHERE id IN ({placeholders})",  # noqa: S608
            ids,
        ).fetchall()
        return {r["id"]: r["embedding_text"] for r in rows}

    def paper_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM papers").fetchone()
        return row["cnt"]

    def chunk_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM chunks").fetchone()
        return row["cnt"]

    def delete_chunks_for_papers(self, paper_ids: list[str]):
        """Remove chunks for given paper IDs (before re-chunking)."""
        placeholders = ",".join("?" for _ in paper_ids)
        self.conn.execute(
            f"DELETE FROM chunks WHERE paper_id IN ({placeholders})",  # noqa: S608
            paper_ids,
        )
        self.conn.commit()

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_paper(row: sqlite3.Row) -> Paper:
        return Paper(
            id=row["id"],
            title=row["title"],
            abstract=row["abstract"] or "",
            authors=json.loads(row["authors"]) if row["authors"] else [],
            year=row["year"],
            doi=row["doi"],
            journal=row["journal"],
            source_file=row["source_file"],
            zotero_uri=row["zotero_uri"],
        )

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=row["id"],
            paper_id=row["paper_id"],
            text=row["text"],
            section=row["section"],
            char_offset=row["char_offset"] or 0,
            source=row["source"],
            context_text=row["context_text"],
        )
