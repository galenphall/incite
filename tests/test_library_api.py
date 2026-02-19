"""Tests for the library API orchestration layer (cloud/library_api.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture()
def mock_db():
    """Fake DB connection for testing."""
    db = MagicMock()
    db.cursor.return_value.__enter__ = MagicMock()
    db.cursor.return_value.__exit__ = MagicMock()
    return db


@pytest.fixture()
def mock_embedder():
    """Fake embedder with embed() and get_storage_key()."""
    emb = MagicMock()
    emb.get_storage_key.return_value = "granite-ft-onnx"
    emb.embed.return_value = np.random.rand(1, 384).astype(np.float32)
    return emb


SAMPLE_PAPER = {
    "title": "Attention Is All You Need",
    "doi": "10.1234/test",
    "year": 2017,
    "authors": ["Ashish Vaswani", "Noam Shazeer"],
    "abstract": "The dominant sequence transduction models...",
    "journal": "NeurIPS",
}


@patch("cloud.pgvector_store.PgVectorStore")
@patch("cloud.database.save_library_paper")
@patch("cloud.database.get_or_create_canonical_paper", return_value="cp_abc123")
@patch("cloud.database._fetchall", return_value=[])  # no existing papers
@patch("cloud.database._ph", return_value="%s")
def test_save_single_paper(
    mock_ph,
    mock_fetchall,
    mock_get_canonical,
    mock_save_lp,
    mock_pgvector_cls,
    mock_db,
    mock_embedder,
):
    """Save one paper: canonical created, embedded, stored in pgvector."""
    from cloud.library_api import save_papers_to_library

    result = save_papers_to_library(
        db=mock_db,
        library_id=1,
        papers=[SAMPLE_PAPER],
        collection_id=None,
        tags=[],
        enrich=False,
        embedder=mock_embedder,
    )

    assert len(result["saved"]) == 1
    assert result["saved"][0]["canonical_id"] == "cp_abc123"
    assert len(result["already_existed"]) == 0
    assert len(result["errors"]) == 0

    mock_get_canonical.assert_called_once()
    mock_save_lp.assert_called_once()
    mock_embedder.embed.assert_called_once()
    mock_pgvector_cls.return_value.add.assert_called_once()


@patch("cloud.pgvector_store.PgVectorStore")
@patch("cloud.database.get_or_create_canonical_paper", return_value="cp_abc123")
@patch("cloud.database._fetchall", return_value=[{"canonical_id": "cp_abc123"}])
@patch("cloud.database._ph", return_value="%s")
def test_save_duplicate_paper(
    mock_ph,
    mock_fetchall,
    mock_get_canonical,
    mock_pgvector_cls,
    mock_db,
    mock_embedder,
):
    """Paper already in library returns in already_existed."""
    from cloud.library_api import save_papers_to_library

    result = save_papers_to_library(
        db=mock_db,
        library_id=1,
        papers=[SAMPLE_PAPER],
        collection_id=None,
        tags=[],
        enrich=False,
        embedder=mock_embedder,
    )

    assert len(result["saved"]) == 0
    assert len(result["already_existed"]) == 1
    assert result["already_existed"][0]["canonical_id"] == "cp_abc123"


@patch("cloud.database._fetchone")
@patch("cloud.database._ph", return_value="%s")
@patch("cloud.database.normalize_doi", return_value="10.1234/test")
def test_check_by_doi(mock_ndoi, mock_ph, mock_fetchone, mock_db):
    """Check finds paper by DOI."""
    mock_fetchone.return_value = {"id": "cp_found"}

    with (
        patch("cloud.refman_db.get_item_collection_ids", return_value=set()),
        patch("cloud.refman_db.get_item_tags", return_value=[]),
    ):
        from cloud.library_api import check_papers_in_library

        results = check_papers_in_library(
            mock_db, 1, [{"doi": "10.1234/test", "title": "Test Paper"}]
        )

    assert len(results) == 1
    assert results[0]["in_library"] is True
    assert results[0]["canonical_id"] == "cp_found"


@patch("cloud.database._fetchone", return_value={"id": "cp_title_match"})
@patch("cloud.database._ph", return_value="%s")
@patch("cloud.database.normalize_doi", return_value=None)
@patch("cloud.database.normalize_title", return_value="test paper")
def test_check_by_title(mock_ntitle, mock_ndoi, mock_ph, mock_fetchone, mock_db):
    """Check finds paper by normalized title when DOI is absent."""
    with (
        patch("cloud.refman_db.get_item_collection_ids", return_value=set()),
        patch("cloud.refman_db.get_item_tags", return_value=[]),
    ):
        from cloud.library_api import check_papers_in_library

        results = check_papers_in_library(mock_db, 1, [{"doi": None, "title": "Test Paper"}])

    assert results[0]["in_library"] is True
    assert results[0]["canonical_id"] == "cp_title_match"


@patch("cloud.pgvector_store.PgVectorStore")
@patch("cloud.database.save_library_paper")
@patch("cloud.database.get_or_create_canonical_paper", return_value="cp_xyz")
@patch("cloud.database._fetchall", return_value=[])
@patch("cloud.database._ph", return_value="%s")
@patch("cloud.refman_db.add_items_to_collection")
@patch("cloud.refman_db.get_or_create_tag", return_value=42)
@patch("cloud.refman_db.add_tag_to_item")
def test_save_with_collection_and_tags(
    mock_add_tag,
    mock_get_tag,
    mock_add_coll,
    mock_ph,
    mock_fetchall,
    mock_get_canonical,
    mock_save_lp,
    mock_pgvector_cls,
    mock_db,
    mock_embedder,
):
    """Saving with collection and tags calls the right refman functions."""
    from cloud.library_api import save_papers_to_library

    result = save_papers_to_library(
        db=mock_db,
        library_id=1,
        papers=[SAMPLE_PAPER],
        collection_id=5,
        tags=["important", "review"],
        enrich=False,
        embedder=mock_embedder,
    )

    assert len(result["saved"]) == 1
    mock_add_coll.assert_called_once_with(mock_db, 5, 1, ["cp_xyz"])
    assert mock_get_tag.call_count == 2
    assert mock_add_tag.call_count == 2
