"""E2E tests for library seeding and verification.

Seeds the test library via the JSON API (no GROBID needed), then verifies
papers land, diagnostics work, and recommendations return results.

Seeding is idempotent: if the expected papers already exist in the library,
the upload is skipped. Safe to run repeatedly without duplicating papers.

GROBID-dependent upload tests (PDF, BibTeX) live in test_e2e_z_grobid_uploads.py
and run last to avoid poisoning library status for other tests.

Run:
    pytest tests/test_e2e_a_upload_pipeline.py -v -m e2e -o "addopts=" \
        --api-url https://inciteref.com --timeout=600
"""

from __future__ import annotations

import time

import pytest
import requests

from tests.conftest import e2e_csrf

pytestmark = pytest.mark.e2e

# DOIs used for seeding — single source of truth
SEED_DOIS = {
    "10.48550/arXiv.1706.03762",  # Attention Is All You Need
    "10.18653/v1/N19-1423",  # BERT
}

SEED_PAPERS = [
    {
        "title": "Attention Is All You Need",
        "authors": ["Vaswani, Ashish", "Shazeer, Noam", "Parmar, Niki"],
        "year": 2017,
        "doi": "10.48550/arXiv.1706.03762",
        "abstract": (
            "The dominant sequence transduction models are based on complex "
            "recurrent or convolutional neural networks that include an "
            "encoder and a decoder."
        ),
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "authors": [
            "Devlin, Jacob",
            "Chang, Ming-Wei",
            "Lee, Kenton",
            "Toutanova, Kristina",
        ],
        "year": 2019,
        "doi": "10.18653/v1/N19-1423",
        "abstract": (
            "We introduce a new language representation model called BERT, "
            "which stands for Bidirectional Encoder Representations from "
            "Transformers."
        ),
    },
]


def _library_has_seed_papers(token: str, base_url: str) -> bool:
    """Check if the library already contains all expected seed papers."""
    resp = requests.get(
        f"{base_url}/api/v1/library/papers",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    if resp.status_code != 200:
        return False

    papers = resp.json().get("papers", [])
    existing_dois = {p.get("doi", "").strip() for p in papers if p.get("doi")}
    return SEED_DOIS.issubset(existing_dois)


class TestUploadPipeline:
    """Seeds the library and verifies papers, diagnostics, and recommendations."""

    _api_token: str | None = None

    def test_01_seed_via_json_api(self, e2e_session, e2e_base_url):
        """Seed library with real papers via JSON API (no GROBID needed).

        Idempotent: checks for existing papers first and skips upload if
        the expected seed papers are already present.
        """
        # Create a temporary API token for Bearer auth
        resp = e2e_session.post(
            f"{e2e_base_url}/api/v1/token",
            json={"name": f"e2e-seed-{int(time.time())}"},
            headers=e2e_csrf(e2e_session),
        )
        assert resp.status_code == 200, f"Token creation failed: {resp.text[:300]}"
        token = resp.json()["token"]
        self.__class__._api_token = token

        # Idempotency check: skip upload if seed papers already exist
        if _library_has_seed_papers(token, e2e_base_url):
            return  # Papers already seeded — nothing to do

        # Upload seed papers
        resp = requests.post(
            f"{e2e_base_url}/api/v1/library/papers",
            json={"papers": SEED_PAPERS, "enrich": True},
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        )
        assert resp.status_code == 201, (
            f"JSON API save failed: {resp.status_code} {resp.text[:300]}"
        )
        data = resp.json()
        saved = data.get("saved", [])
        already = data.get("already_existed", [])
        assert saved or already, f"No papers saved or found: {data}"

    def test_02_verify_papers_landed(self, e2e_session, e2e_base_url):
        """Verify seeded papers appear in the library."""
        token = self.__class__._api_token
        if not token:
            pytest.skip("No API token — test_01 did not run")

        resp = requests.get(
            f"{e2e_base_url}/api/v1/library/papers",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        papers = data.get("papers", [])
        assert len(papers) >= 2, f"Expected at least 2 papers in library, got {len(papers)}"

        # Verify seed papers are present by DOI
        existing_dois = {p.get("doi", "").strip() for p in papers if p.get("doi")}
        for doi in SEED_DOIS:
            assert doi in existing_dois, (
                f"Seed paper DOI {doi} not found in library. Found DOIs: {existing_dois}"
            )

    def test_03_verify_diagnostics(self, e2e_session, e2e_base_url):
        """Verify upload diagnostics endpoint works."""
        resp = e2e_session.get(f"{e2e_base_url}/api/v1/upload-library/diagnostics")
        assert resp.status_code == 200

    def test_04_verify_recommendations(self, e2e_session, e2e_base_url):
        """Verify recommendations work with the library."""
        resp = e2e_session.post(
            f"{e2e_base_url}/api/recommend",
            data={
                "query": "transformer architectures and self-attention mechanisms",
                "k": "5",
            },
            headers=e2e_csrf(e2e_session),
        )
        assert resp.status_code == 200
        assert len(resp.text) > 50, "Recommendation response too short"

    def test_99_cleanup(self, e2e_session, e2e_base_url):
        """Revoke temporary API token (keep seeded papers for downstream tests)."""
        if self.__class__._api_token:
            e2e_session.delete(
                f"{e2e_base_url}/api/v1/token/{self.__class__._api_token}",
                headers=e2e_csrf(e2e_session),
            )
