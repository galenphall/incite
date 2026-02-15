"""SQLite storage layer for synthetic citation contexts.

Stores LLM-generated citation contexts and K-NN reference sets
for evaluation. Provides export to JSONL format compatible with
the existing test set evaluation pipeline.
"""

import sqlite3
from pathlib import Path
from typing import Optional


class SyntheticDB:
    """SQLite database for synthetic citation contexts."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        self._migrate()

    def _create_tables(self):
        cursor = self._conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS contexts (
                id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                citation_type TEXT NOT NULL,
                text TEXT NOT NULL,
                section_hint TEXT,
                batch_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_contexts_paper ON contexts(paper_id);
            CREATE INDEX IF NOT EXISTS idx_contexts_type ON contexts(citation_type);

            CREATE TABLE IF NOT EXISTS reference_sets (
                paper_id TEXT NOT NULL,
                neighbor_id TEXT NOT NULL,
                rank INTEGER,
                PRIMARY KEY (paper_id, neighbor_id)
            );
            CREATE INDEX IF NOT EXISTS idx_refsets_paper ON reference_sets(paper_id);

            CREATE TABLE IF NOT EXISTS generation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model TEXT,
                num_papers INTEGER,
                num_contexts INTEGER,
                batch_id TEXT,
                notes TEXT
            );
        """)
        self._conn.commit()

    def _migrate(self):
        """Run schema migrations for existing databases."""
        cursor = self._conn.cursor()
        # Check if difficulty column exists
        cursor.execute("PRAGMA table_info(contexts)")
        columns = {row["name"] for row in cursor.fetchall()}
        if "difficulty" not in columns:
            cursor.execute("ALTER TABLE contexts ADD COLUMN difficulty TEXT DEFAULT ''")
            self._conn.commit()

    def insert_contexts(self, contexts: list[dict]) -> int:
        """Bulk insert citation contexts.

        Args:
            contexts: List of dicts with keys:
                paper_id, citation_type, text, section_hint, batch_id (optional)

        Returns:
            Number of contexts inserted
        """
        cursor = self._conn.cursor()
        inserted = 0
        for ctx in contexts:
            ctx_id = ctx.get("id") or f"synth_{ctx['paper_id']}_{ctx['citation_type']}"
            try:
                cursor.execute(
                    """INSERT OR REPLACE INTO contexts
                       (id, paper_id, citation_type, text, section_hint, batch_id, difficulty)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        ctx_id,
                        ctx["paper_id"],
                        ctx["citation_type"],
                        ctx["text"],
                        ctx.get("section_hint"),
                        ctx.get("batch_id"),
                        ctx.get("difficulty", ""),
                    ),
                )
                inserted += 1
            except sqlite3.Error:
                pass
        self._conn.commit()
        return inserted

    def insert_reference_sets(self, ref_sets: dict[str, list[str]]) -> int:
        """Bulk insert K-NN reference sets.

        Args:
            ref_sets: Dict mapping paper_id -> list of neighbor paper_ids (ordered by rank)

        Returns:
            Number of rows inserted
        """
        cursor = self._conn.cursor()
        inserted = 0
        for paper_id, neighbors in ref_sets.items():
            for rank, neighbor_id in enumerate(neighbors):
                try:
                    cursor.execute(
                        """INSERT OR REPLACE INTO reference_sets
                           (paper_id, neighbor_id, rank)
                           VALUES (?, ?, ?)""",
                        (paper_id, neighbor_id, rank),
                    )
                    inserted += 1
                except sqlite3.Error:
                    pass
        self._conn.commit()
        return inserted

    def log_run(
        self,
        model: str,
        num_papers: int,
        num_contexts: int,
        batch_id: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> int:
        """Log a generation run.

        Returns:
            Run ID
        """
        cursor = self._conn.cursor()
        cursor.execute(
            """INSERT INTO generation_runs (model, num_papers, num_contexts, batch_id, notes)
               VALUES (?, ?, ?, ?, ?)""",
            (model, num_papers, num_contexts, batch_id, notes),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_contexts(
        self,
        paper_id: Optional[str] = None,
        citation_type: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> list[dict]:
        """Query contexts with optional filters.

        Args:
            paper_id: Filter by target paper
            citation_type: Filter by citation type
            difficulty: Filter by difficulty level ("standard", "moderate", or "")

        Returns:
            List of context dicts
        """
        query = "SELECT * FROM contexts WHERE 1=1"
        params = []
        if paper_id:
            query += " AND paper_id = ?"
            params.append(paper_id)
        if citation_type:
            query += " AND citation_type = ?"
            params.append(citation_type)
        if difficulty is not None:
            query += " AND difficulty = ?"
            params.append(difficulty)
        query += " ORDER BY paper_id, citation_type"

        cursor = self._conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_existing_paper_ids(self, difficulty: Optional[str] = None) -> set[str]:
        """Get set of paper IDs that already have contexts.

        Args:
            difficulty: If set, only count papers with contexts of this difficulty.
        """
        cursor = self._conn.cursor()
        if difficulty is not None:
            cursor.execute(
                "SELECT DISTINCT paper_id FROM contexts WHERE difficulty = ?",
                (difficulty,),
            )
        else:
            cursor.execute("SELECT DISTINCT paper_id FROM contexts")
        return {row["paper_id"] for row in cursor.fetchall()}

    def get_reference_set(self, paper_id: str) -> list[str]:
        """Get the K-NN reference set for a paper.

        Args:
            paper_id: Paper to get neighbors for

        Returns:
            List of neighbor paper IDs ordered by rank
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT neighbor_id FROM reference_sets WHERE paper_id = ? ORDER BY rank",
            (paper_id,),
        )
        return [row["neighbor_id"] for row in cursor.fetchall()]

    def get_all_reference_sets(self) -> dict[str, list[str]]:
        """Get all reference sets.

        Returns:
            Dict mapping paper_id -> list of neighbor IDs (ordered by rank)
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT paper_id, neighbor_id FROM reference_sets ORDER BY paper_id, rank")
        ref_sets: dict[str, list[str]] = {}
        for row in cursor.fetchall():
            ref_sets.setdefault(row["paper_id"], []).append(row["neighbor_id"])
        return ref_sets

    def stats(self) -> dict:
        """Get summary statistics.

        Returns:
            Dict with counts by type, total contexts, papers, etc.
        """
        cursor = self._conn.cursor()

        cursor.execute("SELECT COUNT(*) as n FROM contexts")
        total_contexts = cursor.fetchone()["n"]

        cursor.execute("SELECT COUNT(DISTINCT paper_id) as n FROM contexts")
        total_papers = cursor.fetchone()["n"]

        cursor.execute(
            "SELECT citation_type, COUNT(*) as n FROM contexts "
            "GROUP BY citation_type ORDER BY citation_type"
        )
        by_type = {row["citation_type"]: row["n"] for row in cursor.fetchall()}

        cursor.execute("SELECT COUNT(DISTINCT paper_id) as n FROM reference_sets")
        papers_with_refs = cursor.fetchone()["n"]

        cursor.execute("SELECT COUNT(*) as n FROM generation_runs")
        num_runs = cursor.fetchone()["n"]

        return {
            "total_contexts": total_contexts,
            "total_papers": total_papers,
            "by_type": by_type,
            "papers_with_reference_sets": papers_with_refs,
            "generation_runs": num_runs,
        }

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
