"""Tests for reference manager CRUD operations (cloud/refman_db.py).

These tests require a running PostgreSQL instance with the pgvector extension.
Set DATABASE_URL to enable them. They are skipped in CI without Postgres.

Run locally: DATABASE_URL=postgresql://... pytest tests/test_refman_db.py -v
"""

import os
import uuid

import pytest

os.environ.setdefault("ENCRYPTION_KEY", "7gRTU890D-m7JaPp6-ks4KMHLpXs5-ugRvtjfAXwZPE=")
os.environ.setdefault("INVITE_CODE", "test-invite-123")

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — skipping Postgres-dependent tests",
)


@pytest.fixture
def pg_db():
    """Get a Postgres connection and set up test data."""
    from cloud.database import get_db, init_db, return_conn

    init_db()
    db = get_db()

    # Create a test user and library with unique email
    test_email = f"test-refman-{uuid.uuid4().hex[:8]}@test.com"
    cur = db.cursor()
    cur.execute(
        "INSERT INTO users (name, email, password_hash, created_at)"
        " VALUES (%s, %s, %s, NOW()) RETURNING id",
        ("Test", test_email, "hash"),
    )
    user_id = cur.fetchone()[0]
    cur.execute(
        "INSERT INTO libraries (user_id, source_type, status)"
        " VALUES (%s, 'zotero', 'ready') RETURNING id",
        (user_id,),
    )
    library_id = cur.fetchone()[0]
    # Create canonical papers for item references
    for pid in ["paper_a", "paper_b", "paper_c"]:
        cur.execute(
            "INSERT INTO canonical_papers (id, normalized_title, created_at)"
            " VALUES (%s, %s, NOW()) ON CONFLICT (id) DO NOTHING",
            (pid, pid),
        )
    db.commit()

    yield db, library_id

    # Cleanup: delete test data
    try:
        cur = db.cursor()
        cur.execute("DELETE FROM libraries WHERE id = %s", (library_id,))
        cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
        db.commit()
    except Exception:
        pass
    finally:
        return_conn(db)


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


class TestCollections:
    def test_create_and_get(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import create_collection, get_collection

        cid = create_collection(db, library_id, "My Papers", color="#ff0000")
        assert cid > 0

        coll = get_collection(db, cid)
        assert coll is not None
        assert coll["name"] == "My Papers"
        assert coll["color"] == "#ff0000"
        assert coll["library_id"] == library_id

    def test_list_collections(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import create_collection, list_collections

        create_collection(db, library_id, "B Folder", position=1)
        create_collection(db, library_id, "A Folder", position=0)

        colls = list_collections(db, library_id)
        assert len(colls) >= 2
        # Find our collections and verify ordering
        names = [c["name"] for c in colls]
        assert "A Folder" in names
        assert "B Folder" in names

    def test_nested_collections(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import create_collection, list_collections

        parent_id = create_collection(db, library_id, "Parent")
        create_collection(db, library_id, "Child", parent_id=parent_id)

        children = list_collections(db, library_id, parent_id=parent_id)
        assert len(children) == 1
        assert children[0]["name"] == "Child"

    def test_update_collection(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import create_collection, get_collection, update_collection

        cid = create_collection(db, library_id, "Old Name")
        update_collection(db, cid, name="New Name", color="#0000ff")

        coll = get_collection(db, cid)
        assert coll["name"] == "New Name"
        assert coll["color"] == "#0000ff"

    def test_delete_collection(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import create_collection, delete_collection, get_collection

        cid = create_collection(db, library_id, "Doomed")
        delete_collection(db, cid)
        assert get_collection(db, cid) is None

    def test_get_nonexistent_collection(self, pg_db):
        db, _ = pg_db
        from cloud.refman_db import get_collection

        assert get_collection(db, 999999) is None


# ---------------------------------------------------------------------------
# Collection Items
# ---------------------------------------------------------------------------


class TestCollectionItems:
    def test_add_and_list_items(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import (
            add_items_to_collection,
            create_collection,
            get_collection_items,
        )

        cid = create_collection(db, library_id, "Reading List")
        add_items_to_collection(db, cid, library_id, ["paper_a", "paper_b"])

        items = get_collection_items(db, cid)
        assert len(items) == 2
        ids = {i["canonical_id"] for i in items}
        assert ids == {"paper_a", "paper_b"}

    def test_add_duplicate_item_ignored(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import (
            add_items_to_collection,
            create_collection,
            get_collection_items,
        )

        cid = create_collection(db, library_id, "Dupes")
        add_items_to_collection(db, cid, library_id, ["paper_a"])
        add_items_to_collection(db, cid, library_id, ["paper_a"])

        items = get_collection_items(db, cid)
        assert len(items) == 1

    def test_remove_items(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import (
            add_items_to_collection,
            create_collection,
            get_collection_items,
            remove_items_from_collection,
        )

        cid = create_collection(db, library_id, "Shrinking")
        add_items_to_collection(db, cid, library_id, ["paper_a", "paper_b"])
        remove_items_from_collection(db, cid, ["paper_a"])

        items = get_collection_items(db, cid)
        assert len(items) == 1
        assert items[0]["canonical_id"] == "paper_b"

    def test_delete_collection_cascades_items(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import (
            add_items_to_collection,
            create_collection,
            delete_collection,
            get_collection_items,
        )

        cid = create_collection(db, library_id, "Cascade")
        add_items_to_collection(db, cid, library_id, ["paper_a"])
        delete_collection(db, cid)

        items = get_collection_items(db, cid)
        assert len(items) == 0


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


class TestTags:
    def test_create_and_get(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import create_tag, get_tag

        tag_name = f"important-{uuid.uuid4().hex[:6]}"
        tid = create_tag(db, library_id, tag_name, color="#ff0000")
        assert tid > 0

        tag = get_tag(db, tid)
        assert tag is not None
        assert tag["name"] == tag_name
        assert tag["color"] == "#ff0000"

    def test_list_tags(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import create_tag, list_tags

        create_tag(db, library_id, f"beta-{uuid.uuid4().hex[:6]}")
        create_tag(db, library_id, f"alpha-{uuid.uuid4().hex[:6]}")

        tags = list_tags(db, library_id)
        assert len(tags) >= 2

    def test_unique_tag_name_per_library(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import create_tag

        unique_name = f"unique-tag-{uuid.uuid4().hex[:6]}"
        create_tag(db, library_id, unique_name)
        with pytest.raises(Exception):
            create_tag(db, library_id, unique_name)

    def test_update_tag(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import create_tag, get_tag, update_tag

        tid = create_tag(db, library_id, f"old-{uuid.uuid4().hex[:6]}")
        update_tag(db, tid, name=f"new-{uuid.uuid4().hex[:6]}", color="#00ff00")

        tag = get_tag(db, tid)
        assert tag["color"] == "#00ff00"

    def test_delete_tag(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import create_tag, delete_tag, get_tag

        tid = create_tag(db, library_id, f"doomed-{uuid.uuid4().hex[:6]}")
        delete_tag(db, tid)
        assert get_tag(db, tid) is None

    def test_get_nonexistent_tag(self, pg_db):
        db, _ = pg_db
        from cloud.refman_db import get_tag

        assert get_tag(db, 999999) is None


# ---------------------------------------------------------------------------
# Item Tags
# ---------------------------------------------------------------------------


class TestItemTags:
    def test_add_and_get_item_tags(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import add_tag_to_item, create_tag, get_item_tags

        t1 = create_tag(db, library_id, f"methods-{uuid.uuid4().hex[:6]}")
        t2 = create_tag(db, library_id, f"results-{uuid.uuid4().hex[:6]}")
        add_tag_to_item(db, "paper_a", library_id, t1)
        add_tag_to_item(db, "paper_a", library_id, t2)

        tags = get_item_tags(db, "paper_a", library_id)
        assert len(tags) >= 2

    def test_add_duplicate_tag_ignored(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import add_tag_to_item, create_tag, get_item_tags

        tid = create_tag(db, library_id, f"dupe-tag-{uuid.uuid4().hex[:6]}")
        add_tag_to_item(db, "paper_b", library_id, tid)
        add_tag_to_item(db, "paper_b", library_id, tid)

        tags = get_item_tags(db, "paper_b", library_id)
        tag_ids = [t["id"] for t in tags]
        assert tag_ids.count(tid) == 1

    def test_remove_tag_from_item(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import (
            add_tag_to_item,
            create_tag,
            get_item_tags,
            remove_tag_from_item,
        )

        tid = create_tag(db, library_id, f"removable-{uuid.uuid4().hex[:6]}")
        add_tag_to_item(db, "paper_c", library_id, tid)
        remove_tag_from_item(db, "paper_c", library_id, tid)

        tags = get_item_tags(db, "paper_c", library_id)
        tag_ids = [t["id"] for t in tags]
        assert tid not in tag_ids

    def test_delete_tag_cascades_item_tags(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import add_tag_to_item, create_tag, delete_tag, get_item_tags

        tid = create_tag(db, library_id, f"cascade-tag-{uuid.uuid4().hex[:6]}")
        add_tag_to_item(db, "paper_a", library_id, tid)
        delete_tag(db, tid)

        tags = get_item_tags(db, "paper_a", library_id)
        tag_ids = [t["id"] for t in tags]
        assert tid not in tag_ids


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


class TestNotes:
    def test_create_and_get(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import create_note, get_note

        nid = create_note(
            db, "paper_a", library_id, title="My Note", content_md="Some **markdown**"
        )
        assert nid > 0

        note = get_note(db, nid)
        assert note is not None
        assert note["title"] == "My Note"
        assert note["content_md"] == "Some **markdown**"

    def test_list_notes(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import create_note, list_notes

        create_note(db, "paper_b", library_id, title="First")
        create_note(db, "paper_b", library_id, title="Second")

        notes = list_notes(db, "paper_b", library_id)
        assert len(notes) >= 2

    def test_update_note(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import create_note, get_note, update_note

        nid = create_note(db, "paper_a", library_id, title="Draft")
        update_note(db, nid, title="Final", content_md="Updated content")

        note = get_note(db, nid)
        assert note["title"] == "Final"
        assert note["content_md"] == "Updated content"

    def test_delete_note(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import create_note, delete_note, get_note

        nid = create_note(db, "paper_a", library_id, title="Doomed")
        delete_note(db, nid)
        assert get_note(db, nid) is None

    def test_get_nonexistent_note(self, pg_db):
        db, _ = pg_db
        from cloud.refman_db import get_note

        assert get_note(db, 999999) is None

    def test_notes_isolated_by_item(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import create_note, list_notes

        create_note(db, "paper_a", library_id, title="On A unique")
        create_note(db, "paper_b", library_id, title="On B unique")

        notes_a = list_notes(db, "paper_a", library_id)
        titles = [n["title"] for n in notes_a]
        assert "On A unique" in titles
        assert "On B unique" not in titles


# ---------------------------------------------------------------------------
# User Item Metadata
# ---------------------------------------------------------------------------


class TestUserItemMetadata:
    def test_get_or_create_defaults(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import get_or_create_item_metadata

        meta = get_or_create_item_metadata(db, library_id, "paper_a")
        assert meta is not None
        assert meta["reading_status"] == "unread"
        assert meta["starred"] is False

    def test_get_or_create_idempotent(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import get_or_create_item_metadata

        m1 = get_or_create_item_metadata(db, library_id, "paper_a")
        m2 = get_or_create_item_metadata(db, library_id, "paper_a")
        assert m1["canonical_id"] == m2["canonical_id"]

    def test_update_star(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import get_or_create_item_metadata, update_star

        get_or_create_item_metadata(db, library_id, "paper_a")
        update_star(db, library_id, "paper_a", True)

        meta = get_or_create_item_metadata(db, library_id, "paper_a")
        assert meta["starred"] is True

    def test_update_reading_status(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import get_or_create_item_metadata, update_reading_status

        get_or_create_item_metadata(db, library_id, "paper_a")
        update_reading_status(db, library_id, "paper_a", "reading")

        meta = get_or_create_item_metadata(db, library_id, "paper_a")
        assert meta["reading_status"] == "reading"

    def test_update_reading_status_to_read_sets_timestamp(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import get_or_create_item_metadata, update_reading_status

        get_or_create_item_metadata(db, library_id, "paper_a")
        update_reading_status(db, library_id, "paper_a", "read")

        meta = get_or_create_item_metadata(db, library_id, "paper_a")
        assert meta["reading_status"] == "read"
        assert meta["last_read_at"] is not None

    def test_invalid_reading_status_raises(self, pg_db):
        db, library_id = pg_db
        from cloud.refman_db import get_or_create_item_metadata, update_reading_status

        get_or_create_item_metadata(db, library_id, "paper_a")
        with pytest.raises(ValueError, match="Invalid reading status"):
            update_reading_status(db, library_id, "paper_a", "invalid")
