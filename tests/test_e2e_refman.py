"""E2E tests for reference manager operations.

Deep coverage of collections, tags, notes, star/reading status,
BibTeX/RIS export, paper detail page, and bulk operations.

Run:
    pytest tests/test_e2e_refman.py -v -m e2e -o "addopts=" \
        --api-url https://inciteref.com --timeout=120
"""

from __future__ import annotations

import re
import time

import pytest

from tests.conftest import e2e_csrf

pytestmark = pytest.mark.e2e


def _require(value, msg="prerequisite test did not pass"):
    if not value:
        pytest.skip(msg)
    return value


class TestRefmanOperations:
    """Deep reference manager testing."""

    _canonical_id: str | None = None
    _canonical_id_2: str | None = None
    _collection_id: int | None = None
    _sub_collection_id: int | None = None
    _tag_id: int | None = None
    _note_id: int | None = None
    _starred: bool = False

    def _get_two_paper_ids(self, e2e_session, e2e_base_url) -> tuple[str, str]:
        """Get two paper canonical IDs from the library."""
        if self.__class__._canonical_id and self.__class__._canonical_id_2:
            return self.__class__._canonical_id, self.__class__._canonical_id_2

        resp = e2e_session.get(f"{e2e_base_url}/api/library/papers")
        assert resp.status_code == 200

        ids = re.findall(r'/web/papers/([a-zA-Z0-9_%-]+)"', resp.text)
        if len(ids) < 2:
            pytest.skip("Need at least 2 papers in library for refman tests")

        self.__class__._canonical_id = ids[0]
        self.__class__._canonical_id_2 = ids[1]
        return ids[0], ids[1]

    def test_01_create_collection(self, e2e_session, e2e_base_url):
        """Create a collection."""
        resp = e2e_session.post(
            f"{e2e_base_url}/api/refman/collections",
            json={"name": f"e2e-test-collection-{int(time.time())}"},
            headers=e2e_csrf(e2e_session),
        )
        assert resp.status_code == 200, f"Create collection failed: {resp.text[:300]}"
        data = resp.json()
        assert "id" in data
        self.__class__._collection_id = data["id"]

    def test_02_add_papers_to_collection(self, e2e_session, e2e_base_url):
        """Add papers to the collection."""
        cid1, cid2 = self._get_two_paper_ids(e2e_session, e2e_base_url)
        coll_id = _require(self.__class__._collection_id, "no collection")

        resp = e2e_session.post(
            f"{e2e_base_url}/api/refman/collections/{coll_id}/items",
            json={"canonical_ids": [cid1, cid2]},
            headers=e2e_csrf(e2e_session),
        )
        assert resp.status_code == 200

    def test_03_create_sub_collection(self, e2e_session, e2e_base_url):
        """Create a nested sub-collection."""
        parent_id = _require(self.__class__._collection_id, "no collection")

        resp = e2e_session.post(
            f"{e2e_base_url}/api/refman/collections",
            json={
                "name": f"e2e-sub-collection-{int(time.time())}",
                "parent_id": parent_id,
            },
            headers=e2e_csrf(e2e_session),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        self.__class__._sub_collection_id = data["id"]

    def test_04_create_and_assign_tag(self, e2e_session, e2e_base_url):
        """Create a tag and assign it to a paper."""
        cid, _ = self._get_two_paper_ids(e2e_session, e2e_base_url)

        # Create tag
        resp = e2e_session.post(
            f"{e2e_base_url}/api/refman/tags",
            json={"name": f"e2e-test-tag-{int(time.time())}"},
            headers=e2e_csrf(e2e_session),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        self.__class__._tag_id = data["id"]

        # Assign tag to paper
        resp = e2e_session.post(
            f"{e2e_base_url}/api/refman/papers/{cid}/tags/{data['id']}",
            headers=e2e_csrf(e2e_session),
        )
        assert resp.status_code == 200

    def test_05_note_crud(self, e2e_session, e2e_base_url):
        """Create, edit, and verify a note on a paper."""
        cid, _ = self._get_two_paper_ids(e2e_session, e2e_base_url)
        headers = e2e_csrf(e2e_session)

        # Create note (form-encoded)
        resp = e2e_session.post(
            f"{e2e_base_url}/api/refman/papers/{cid}/notes",
            data={
                "title": "E2E Test Note",
                "content_md": "Original content from e2e test.",
            },
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        self.__class__._note_id = data["id"]

        # Update note (JSON)
        resp = e2e_session.put(
            f"{e2e_base_url}/api/refman/notes/{data['id']}",
            json={
                "title": "E2E Test Note (Updated)",
                "content_md": "Updated content from e2e test.",
            },
            headers=headers,
        )
        assert resp.status_code == 200

    def test_06_star_and_reading_status(self, e2e_session, e2e_base_url):
        """Star a paper and set reading status."""
        cid, _ = self._get_two_paper_ids(e2e_session, e2e_base_url)
        headers = e2e_csrf(e2e_session)

        # Star (toggle on)
        resp = e2e_session.post(
            f"{e2e_base_url}/api/refman/papers/{cid}/star",
            headers=headers,
        )
        assert resp.status_code == 200
        self.__class__._starred = True

        # Set reading status
        resp = e2e_session.post(
            f"{e2e_base_url}/api/refman/papers/{cid}/status",
            data={"status": "reading"},
            headers=headers,
        )
        assert resp.status_code == 200

    def test_07_export_bibtex(self, e2e_session, e2e_base_url):
        """Export library as BibTeX and verify format."""
        resp = e2e_session.get(f"{e2e_base_url}/api/refman/export/bibtex")
        assert resp.status_code == 200
        content = resp.text
        if not content.strip():
            pytest.skip("Library empty — no papers to export")
        # BibTeX should contain @ entries
        assert "@" in content, "BibTeX export contains no entries"

    def test_08_export_ris(self, e2e_session, e2e_base_url):
        """Export library as RIS and verify format."""
        resp = e2e_session.get(f"{e2e_base_url}/api/refman/export/ris")
        assert resp.status_code == 200
        content = resp.text
        if not content.strip():
            pytest.skip("Library empty — no papers to export")
        # RIS should contain TY - (type) tags
        assert "TY  -" in content or "TY -" in content, "RIS export contains no entries"

    def test_09_paper_detail_page(self, e2e_session, e2e_base_url):
        """Verify paper detail page loads."""
        cid, _ = self._get_two_paper_ids(e2e_session, e2e_base_url)
        resp = e2e_session.get(f"{e2e_base_url}/web/papers/{cid}")
        assert resp.status_code == 200
        assert len(resp.text) > 200, "Paper detail page too short"

    def test_10_bulk_operations(self, e2e_session, e2e_base_url):
        """Bulk tag and bulk collection assignment."""
        _, cid2 = self._get_two_paper_ids(e2e_session, e2e_base_url)
        headers = e2e_csrf(e2e_session)

        tag_id = self.__class__._tag_id
        coll_id = self.__class__._collection_id
        if not tag_id or not coll_id:
            pytest.skip("Need tag and collection from earlier tests")

        # Bulk tag
        resp = e2e_session.post(
            f"{e2e_base_url}/api/refman/bulk/tag",
            json={"canonical_ids": [cid2], "tag_id": tag_id},
            headers=headers,
        )
        assert resp.status_code == 200

        # Bulk collection
        resp = e2e_session.post(
            f"{e2e_base_url}/api/refman/bulk/collection",
            json={"canonical_ids": [cid2], "collection_id": coll_id},
            headers=headers,
        )
        assert resp.status_code == 200

    def test_99_cleanup(self, e2e_session, e2e_base_url):
        """Delete collections, tags, notes, un-star."""
        headers = e2e_csrf(e2e_session)
        cid = self.__class__._canonical_id

        # Delete note
        if self.__class__._note_id:
            e2e_session.delete(
                f"{e2e_base_url}/api/refman/notes/{self.__class__._note_id}",
                headers=headers,
            )

        # Remove tag from papers, then delete tag
        if self.__class__._tag_id:
            for paper_id in (self.__class__._canonical_id, self.__class__._canonical_id_2):
                if paper_id:
                    e2e_session.delete(
                        f"{e2e_base_url}/api/refman/papers/{paper_id}/tags/{self.__class__._tag_id}",
                        headers=headers,
                    )
            e2e_session.delete(
                f"{e2e_base_url}/api/refman/tags/{self.__class__._tag_id}",
                headers=headers,
            )

        # Delete sub-collection first (child before parent)
        if self.__class__._sub_collection_id:
            e2e_session.delete(
                f"{e2e_base_url}/api/refman/collections/{self.__class__._sub_collection_id}",
                headers=headers,
            )
        if self.__class__._collection_id:
            e2e_session.delete(
                f"{e2e_base_url}/api/refman/collections/{self.__class__._collection_id}",
                headers=headers,
            )

        # Un-star (toggle off)
        if self.__class__._starred and cid:
            e2e_session.post(
                f"{e2e_base_url}/api/refman/papers/{cid}/star",
                headers=headers,
            )

        # Reset reading status
        if cid:
            e2e_session.post(
                f"{e2e_base_url}/api/refman/papers/{cid}/status",
                data={"status": "unread"},
                headers=headers,
            )
