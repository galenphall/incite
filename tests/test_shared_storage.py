"""Tests for shared storage: canonical dedup, embedding cache, queue.

Canonical dedup tests (normalize_doi, normalize_title, resolve_canonical_id)
are pure functions that don't need a database. DB-dependent tests require
DATABASE_URL to be set.
"""

import importlib
import os
import threading
import time

import pytest


def _try_import(module_name: str) -> bool:
    """Check if a module is importable without raising."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Pure function tests (no DB needed)
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_normalize_doi_strips_prefix(self):
        from cloud.database import normalize_doi

        assert normalize_doi("https://doi.org/10.1234/test") == "10.1234/test"
        assert normalize_doi("http://dx.doi.org/10.1234/TEST") == "10.1234/test"
        assert normalize_doi("10.1234/test") == "10.1234/test"

    def test_normalize_doi_none(self):
        from cloud.database import normalize_doi

        assert normalize_doi(None) is None
        assert normalize_doi("") is None

    def test_normalize_title(self):
        from cloud.database import normalize_title

        assert normalize_title("The Big Paper") == "big paper"
        assert normalize_title("A Test: With Punctuation!") == "test with punctuation"
        assert normalize_title("  Multiple   Spaces  ") == "multiple spaces"

    def test_normalize_title_accents(self):
        from cloud.database import normalize_title

        assert normalize_title("Über die Quantenmechanik") == "uber die quantenmechanik"

    def test_normalize_title_none(self):
        from cloud.database import normalize_title

        assert normalize_title(None) is None
        assert normalize_title("") is None


class TestCanonicalIdResolution:
    """Test resolve_canonical_id (pure function, no DB)."""

    def test_resolve_by_doi(self):
        from cloud.database import resolve_canonical_id

        id1 = resolve_canonical_id("10.1234/test", "Title", 2023, "Smith")
        id2 = resolve_canonical_id("https://doi.org/10.1234/test", "Different Title", 2024, "Jones")
        assert id1 == id2  # Same DOI -> same ID

    def test_resolve_by_title_year(self):
        from cloud.database import resolve_canonical_id

        id1 = resolve_canonical_id(None, "My Paper Title", 2023, "Smith")
        id2 = resolve_canonical_id(None, "my paper title", 2023, "Jones")
        assert id1 == id2  # Same title+year -> same ID

    def test_resolve_by_title_author(self):
        from cloud.database import resolve_canonical_id

        id1 = resolve_canonical_id(None, "My Paper Title", None, "Smith")
        id2 = resolve_canonical_id(None, "My Paper Title", None, "smith")
        assert id1 == id2  # Same title+author -> same ID

    def test_different_papers_different_ids(self):
        from cloud.database import resolve_canonical_id

        id1 = resolve_canonical_id("10.1234/a", "A", 2023, "Smith")
        id2 = resolve_canonical_id("10.1234/b", "B", 2023, "Smith")
        assert id1 != id2


# ---------------------------------------------------------------------------
# DB-dependent tests (require Postgres)
# ---------------------------------------------------------------------------

pytestmark_pg = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — skipping Postgres-dependent tests",
)


@pytest.fixture
def pg_db():
    """Get a Postgres connection for testing."""
    from cloud.database import get_db, init_db, return_conn

    init_db()
    db = get_db()
    yield db
    return_conn(db)


@pytestmark_pg
class TestCanonicalPaperResolutionDB:
    def test_get_or_create_canonical_paper(self, pg_db):
        from cloud.database import get_or_create_canonical_paper

        id1 = get_or_create_canonical_paper(pg_db, "10.9999/test-shared", "Title", 2023, "Smith")
        id2 = get_or_create_canonical_paper(pg_db, "10.9999/test-shared", "Title", 2023, "Smith")
        assert id1 == id2  # Idempotent

    def test_map_user_paper(self, pg_db):
        import uuid

        from cloud.database import (
            create_library,
            create_user,
            get_canonical_id_for_paper,
            get_or_create_canonical_paper,
            map_user_paper,
        )

        email = f"test-shared-{uuid.uuid4().hex[:8]}@test.com"
        user_id = create_user(pg_db, "Test", email, "hash")
        lib_id = create_library(pg_db, user_id, "12345", "encrypted_key")

        cid = get_or_create_canonical_paper(pg_db, "10.9999/map-test", "Title", 2023, "Smith")
        map_user_paper(pg_db, lib_id, "ZOTERO_KEY_1", cid)

        result = get_canonical_id_for_paper(pg_db, lib_id, "ZOTERO_KEY_1")
        assert result == cid


@pytestmark_pg
class TestStatsQueries:
    def test_total_users(self, pg_db):
        from cloud.database import get_total_users

        # Just verify the function runs without error
        count = get_total_users(pg_db)
        assert isinstance(count, int)

    def test_total_canonical_papers(self, pg_db):
        from cloud.database import get_total_canonical_papers

        count = get_total_canonical_papers(pg_db)
        assert isinstance(count, int)


# ---------------------------------------------------------------------------
# GROBID cache (requires Postgres)
# ---------------------------------------------------------------------------


@pytestmark_pg
class TestGROBIDCache:
    def test_grobid_cache_roundtrip(self, pg_db):
        import json
        import uuid

        from cloud.database import cache_grobid_result, get_cached_grobid_result

        test_hash = f"test_hash_{uuid.uuid4().hex[:8]}"
        result = {"title": "Test", "abstract": "Abs", "sections": []}
        cache_grobid_result(pg_db, test_hash, json.dumps(result))

        cached = get_cached_grobid_result(pg_db, test_hash)
        assert cached is not None
        assert json.loads(cached) == result

    def test_grobid_cache_miss(self, pg_db):
        from cloud.database import get_cached_grobid_result

        assert get_cached_grobid_result(pg_db, "nonexistent_hash_xyz") is None


# ---------------------------------------------------------------------------
# Processing queue (no DB needed)
# ---------------------------------------------------------------------------

_skip_no_fastapi = pytest.mark.skipif(
    not _try_import("fastapi"), reason="fastapi not installed (cloud dep)"
)


@_skip_no_fastapi
class TestProcessingQueue:
    def test_submit_and_execute(self):
        from cloud.server import ProcessingQueue

        q = ProcessingQueue()
        results = []

        def work(x):
            results.append(x)

        q.submit(work, 42)
        q.submit(work, 43)
        q._queue.join()  # Wait for all items to be processed

        assert sorted(results) == [42, 43]

    def test_all_items_processed(self):
        from cloud.server import ProcessingQueue

        q = ProcessingQueue()
        order = []

        def work(x):
            order.append(x)
            time.sleep(0.05)

        for i in range(5):
            q.submit(work, i)
        q._queue.join()

        # All items should be processed (order may vary with multiple workers)
        assert sorted(order) == [0, 1, 2, 3, 4]

    def test_active_library_tracking(self):
        from cloud.server import ProcessingQueue

        q = ProcessingQueue()
        seen_active = []
        event = threading.Event()

        def work(lib_id):
            seen_active.append(q.active_library_id)
            event.set()
            time.sleep(0.1)

        q.submit(work, 99)
        event.wait(timeout=2)
        assert 99 in seen_active
        q._queue.join()
        assert q.active_library_id is None

    def test_queued_count(self):
        from cloud.server import ProcessingQueue

        q = ProcessingQueue()
        barrier = threading.Event()

        def slow_work(x):
            barrier.wait(timeout=5)

        # Submit several items — first blocks, rest queue
        q.submit(slow_work, 1)
        time.sleep(0.05)  # Let worker pick up first item
        q.submit(slow_work, 2)
        q.submit(slow_work, 3)
        time.sleep(0.05)

        assert q.queued_count >= 1  # At least 2 and 3 are queued
        barrier.set()
        q._queue.join()
        assert q.queued_count == 0


# ---------------------------------------------------------------------------
# Rate limiter (no DB needed)
# ---------------------------------------------------------------------------


@_skip_no_fastapi
class TestRateLimiter:
    def test_allows_within_limit_no_redis(self):
        """RedisRateLimiter falls back to allow-all without Redis (fail-open)."""
        from cloud.server import RedisRateLimiter

        limiter = RedisRateLimiter(max_requests=3, window_seconds=60)
        # Without Redis, all requests are allowed (fail-open)
        assert limiter.check("user1") is True
        assert limiter.check("user1") is True
        assert limiter.check("user1") is True
        assert limiter.check("user1") is True  # Fail-open: always allowed

    def test_different_keys_independent_no_redis(self):
        """RedisRateLimiter falls back to allow-all without Redis."""
        from cloud.server import RedisRateLimiter

        limiter = RedisRateLimiter(max_requests=1, window_seconds=60)
        # Without Redis, all requests are allowed (fail-open)
        assert limiter.check("user1") is True
        assert limiter.check("user2") is True
        assert limiter.check("user1") is True  # Fail-open
        assert limiter.check("user2") is True  # Fail-open
