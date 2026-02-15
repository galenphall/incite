"""End-to-end tests for the inCite cloud processing service.

Tests the full pipeline: health → create job → upload PDFs → process → download.

Usage:
    # Against local server (start server first, see below)
    pytest tests/test_cloud_e2e.py -v -m e2e

    # Against remote server
    pytest tests/test_cloud_e2e.py -v -m e2e \
        --api-url https://inciteref.com --api-key mc_xxx

    # Standalone (no pytest needed)
    python tests/test_cloud_e2e.py --api-url http://localhost:9100

Local server setup:
    JOBS_DIR=/tmp/incite_e2e GROBID_URL=http://localhost:8070 API_KEYS="" \
      PYTHONPATH=. uvicorn cloud.server:app --port 9100
"""

from __future__ import annotations

import json
import tarfile
import tempfile
import time
from pathlib import Path

import pytest
import requests

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ZOTERO_CORPUS = Path.home() / ".incite" / "zotero_corpus.jsonl"
MAX_PDF_SIZE = 2 * 1024 * 1024  # 2 MB


def _find_test_pdfs(n: int = 3) -> list[dict]:
    """Find small real PDFs from the Zotero corpus for testing.

    Returns a list of dicts with keys: id, title, abstract, pdf_path.
    Falls back to empty list if corpus or PDFs aren't available.
    """
    if not ZOTERO_CORPUS.exists():
        return []

    zotero_storage = Path.home() / "Zotero" / "storage"
    if not zotero_storage.exists():
        return []

    candidates = []
    with open(ZOTERO_CORPUS) as f:
        for line in f:
            if len(candidates) >= n:
                break
            paper = json.loads(line)
            paper_id = paper.get("id", "")
            # Look for PDF in Zotero storage
            for storage_dir in zotero_storage.iterdir():
                if not storage_dir.is_dir():
                    continue
                for pdf in storage_dir.glob("*.pdf"):
                    if pdf.stat().st_size <= MAX_PDF_SIZE:
                        candidates.append(
                            {
                                "id": paper_id,
                                "title": paper.get("title", "Unknown"),
                                "abstract": paper.get("abstract", ""),
                                "pdf_path": str(pdf),
                            }
                        )
                        break
                if len(candidates) >= n:
                    break

    return candidates[:n]


# ---------------------------------------------------------------------------
# Shared state across sequential tests (class-based)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestCloudE2E:
    """Sequential e2e tests for the cloud processing pipeline.

    Tests share state via class attributes (job_id, test_papers, etc.).
    They MUST run in order — pytest-ordering or alphabetical naming ensures this.
    """

    job_id: str = ""
    test_papers: list[dict] = []
    result_dir: str = ""

    def test_01_health(self, api_url):
        """GET /health returns healthy status with GROBID alive."""
        resp = requests.get(f"{api_url}/health", timeout=10)
        assert resp.status_code == 200, f"Health check failed: {resp.text}"
        data = resp.json()
        assert data["status"] in ("healthy", "degraded"), f"Unexpected status: {data['status']}"
        assert data["grobid_alive"] is True, "GROBID is not running"

    def test_02_create_job(self, api_url, api_headers):
        """POST /api/jobs creates a job and returns a job_id."""
        test_papers = _find_test_pdfs(3)
        if not test_papers:
            # Fallback: create a job with abstract-only papers
            test_papers = [
                {
                    "id": f"test_paper_{i}",
                    "title": f"Test Paper {i}",
                    "abstract": f"This is test abstract number {i} for e2e testing.",
                    "pdf_path": None,
                }
                for i in range(3)
            ]
        TestCloudE2E.test_papers = test_papers

        papers_payload = [
            {
                "id": p["id"],
                "title": p["title"],
                "abstract": p.get("abstract", ""),
                "has_pdf": p.get("pdf_path") is not None,
            }
            for p in test_papers
        ]

        resp = requests.post(
            f"{api_url}/api/jobs",
            json={"papers": papers_payload, "embedder": "minilm-ft"},
            headers=api_headers,
            timeout=10,
        )
        assert resp.status_code == 200, f"Create job failed: {resp.text}"
        data = resp.json()
        assert "job_id" in data, f"No job_id in response: {data}"
        assert data["num_papers"] == len(test_papers)

        TestCloudE2E.job_id = data["job_id"]

    def test_03_upload_pdfs(self, api_url, api_headers):
        """POST /api/jobs/{id}/upload sends real PDF files."""
        if not TestCloudE2E.job_id:
            pytest.skip("No job_id from previous test")

        pdfs_to_upload = [p for p in TestCloudE2E.test_papers if p.get("pdf_path") is not None]
        if not pdfs_to_upload:
            pytest.skip("No PDFs available for upload (abstract-only mode)")

        files = []
        for p in pdfs_to_upload:
            pdf_path = Path(p["pdf_path"])
            files.append(("files", (f"{p['id']}.pdf", open(pdf_path, "rb"), "application/pdf")))

        try:
            resp = requests.post(
                f"{api_url}/api/jobs/{TestCloudE2E.job_id}/upload",
                files=files,
                headers=api_headers,
                timeout=60,
            )
        finally:
            for _, (_, fobj, _) in files:
                fobj.close()

        assert resp.status_code == 200, f"Upload failed: {resp.text}"
        data = resp.json()
        assert data["received"] == len(pdfs_to_upload), (
            f"Expected {len(pdfs_to_upload)} received, got {data['received']}"
        )

    def test_04_start_processing(self, api_url, api_headers):
        """POST /api/jobs/{id}/start enqueues the job."""
        if not TestCloudE2E.job_id:
            pytest.skip("No job_id from previous test")

        resp = requests.post(
            f"{api_url}/api/jobs/{TestCloudE2E.job_id}/start",
            headers=api_headers,
            timeout=10,
        )
        assert resp.status_code == 200, f"Start failed: {resp.text}"
        data = resp.json()
        assert data["state"] == "queued", f"Expected state=queued, got {data['state']}"

    def test_05_poll_until_complete(self, api_url, api_headers):
        """GET /api/jobs/{id}/status polls until completed (timeout 5 min)."""
        if not TestCloudE2E.job_id:
            pytest.skip("No job_id from previous test")

        timeout = 300  # 5 minutes
        poll_interval = 3
        start = time.time()

        while time.time() - start < timeout:
            resp = requests.get(
                f"{api_url}/api/jobs/{TestCloudE2E.job_id}/status",
                headers=api_headers,
                timeout=10,
            )
            assert resp.status_code == 200, f"Status check failed: {resp.text}"
            data = resp.json()
            state = data["state"]

            if state == "completed":
                return
            if state == "failed":
                pytest.fail(f"Job failed: {data.get('error', 'unknown error')}")
            if state == "cancelled":
                pytest.fail("Job was cancelled")

            time.sleep(poll_interval)

        pytest.fail(f"Job did not complete within {timeout}s (last state: {state})")

    def test_06_download_result(self, api_url, api_headers):
        """GET /api/jobs/{id}/result downloads the tarball."""
        if not TestCloudE2E.job_id:
            pytest.skip("No job_id from previous test")

        resp = requests.get(
            f"{api_url}/api/jobs/{TestCloudE2E.job_id}/result",
            headers=api_headers,
            timeout=60,
            stream=True,
        )
        assert resp.status_code == 200, f"Download failed: {resp.text}"

        # Save to temp dir
        tmpdir = tempfile.mkdtemp(prefix="incite_e2e_")
        tarball_path = Path(tmpdir) / "result.tar.gz"
        with open(tarball_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        assert tarball_path.stat().st_size > 0, "Tarball is empty"
        TestCloudE2E.result_dir = tmpdir

    def test_07_tarball_contents(self):
        """Tarball contains expected files: chunks.jsonl, FAISS index, metadata."""
        if not TestCloudE2E.result_dir:
            pytest.skip("No result tarball from previous test")

        tarball_path = Path(TestCloudE2E.result_dir) / "result.tar.gz"
        extract_dir = Path(TestCloudE2E.result_dir) / "extracted"
        extract_dir.mkdir(exist_ok=True)

        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)

        # Check for expected files (may be nested in a subdirectory)
        all_files = {p.name for p in extract_dir.rglob("*") if p.is_file()}

        expected = {"chunks.jsonl", "index.faiss", "id_map.json", "chunk_to_paper.json"}
        missing = expected - all_files
        assert not missing, f"Missing files in tarball: {missing}. Found: {all_files}"

    def test_08_chunks_have_real_content(self):
        """chunks.jsonl contains chunks beyond just abstracts (proves GROBID worked)."""
        if not TestCloudE2E.result_dir:
            pytest.skip("No result tarball from previous test")

        extract_dir = Path(TestCloudE2E.result_dir) / "extracted"
        # Find chunks.jsonl wherever it is
        chunks_files = list(extract_dir.rglob("chunks.jsonl"))
        assert chunks_files, "chunks.jsonl not found in extracted tarball"

        chunks = []
        with open(chunks_files[0]) as f:
            for line in f:
                chunks.append(json.loads(line))

        assert len(chunks) > 0, "No chunks in chunks.jsonl"

        # If we uploaded PDFs, we should have chunks from non-abstract sections
        pdfs_uploaded = any(p.get("pdf_path") for p in TestCloudE2E.test_papers)
        if pdfs_uploaded:
            sections = {c.get("section", "") for c in chunks}
            non_abstract = sections - {"Abstract", "abstract", ""}
            assert non_abstract, (
                f"All chunks are abstract-only — GROBID may not have worked. "
                f"Sections found: {sections}"
            )

    def test_09_faiss_index_loadable(self):
        """FAISS index can be loaded and has the correct dimension."""
        if not TestCloudE2E.result_dir:
            pytest.skip("No result tarball from previous test")

        import faiss

        extract_dir = Path(TestCloudE2E.result_dir) / "extracted"
        index_files = list(extract_dir.rglob("index.faiss"))
        assert index_files, "index.faiss not found in extracted tarball"

        index = faiss.read_index(str(index_files[0]))
        assert index.ntotal > 0, "FAISS index is empty"
        # MiniLM-FT is 384-dim
        assert index.d == 384, f"Expected dimension 384, got {index.d}"

    def test_10_cleanup_job(self, api_url, api_headers):
        """DELETE /api/jobs/{id} cleans up the job."""
        if not TestCloudE2E.job_id:
            pytest.skip("No job_id from previous test")

        resp = requests.delete(
            f"{api_url}/api/jobs/{TestCloudE2E.job_id}",
            headers=api_headers,
            timeout=10,
        )
        assert resp.status_code == 200, f"Cleanup failed: {resp.text}"
        data = resp.json()
        assert data.get("deleted") is True

        # Verify it's gone
        resp = requests.get(
            f"{api_url}/api/jobs/{TestCloudE2E.job_id}/status",
            headers=api_headers,
            timeout=10,
        )
        assert resp.status_code == 404, "Job should be deleted but still exists"


# ---------------------------------------------------------------------------
# Standalone mode: run without pytest
# ---------------------------------------------------------------------------


def _run_standalone():
    """Run the e2e test sequence as a standalone script."""
    import argparse
    import sys
    import traceback

    parser = argparse.ArgumentParser(description="inCite cloud e2e test (standalone)")
    parser.add_argument("--api-url", default="http://localhost:9100", help="Server URL")
    parser.add_argument("--api-key", default="", help="API key (empty = no auth)")
    args = parser.parse_args()

    base_url = args.api_url.rstrip("/")
    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    passed = 0
    failed = 0
    job_id = None
    test_papers = []
    result_dir = None

    def run_step(name, fn):
        nonlocal passed, failed
        t0 = time.time()
        try:
            fn()
            elapsed = time.time() - t0
            print(f"  PASS  {name} ({elapsed:.1f}s)")
            passed += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAIL  {name} ({elapsed:.1f}s): {e}")
            traceback.print_exc()
            failed += 1
            return False
        return True

    # --- Step 1: Health ---
    def step_health():
        resp = requests.get(f"{base_url}/health", timeout=10)
        assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["grobid_alive"] is True, "GROBID not alive"

    # --- Step 2: Create job ---
    def step_create():
        nonlocal job_id, test_papers
        test_papers = _find_test_pdfs(3)
        if not test_papers:
            test_papers = [
                {
                    "id": f"test_{i}",
                    "title": f"Test {i}",
                    "abstract": f"Abstract {i}",
                    "pdf_path": None,
                }
                for i in range(3)
            ]
        payload = {
            "papers": [
                {
                    "id": p["id"],
                    "title": p["title"],
                    "abstract": p.get("abstract", ""),
                    "has_pdf": p.get("pdf_path") is not None,
                }
                for p in test_papers
            ],
            "embedder": "minilm-ft",
        }
        resp = requests.post(f"{base_url}/api/jobs", json=payload, headers=headers, timeout=10)
        assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
        job_id = resp.json()["job_id"]

    # --- Step 3: Upload PDFs ---
    def step_upload():
        pdfs = [p for p in test_papers if p.get("pdf_path")]
        if not pdfs:
            print("         (skipped — no PDFs available)")
            return
        files = []
        for p in pdfs:
            fname = f"{p['id']}.pdf"
            files.append(("files", (fname, open(p["pdf_path"], "rb"), "application/pdf")))
        try:
            resp = requests.post(
                f"{base_url}/api/jobs/{job_id}/upload", files=files, headers=headers, timeout=60
            )
        finally:
            for _, (_, fobj, _) in files:
                fobj.close()
        assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
        assert resp.json()["received"] == len(pdfs)

    # --- Step 4: Start ---
    def step_start():
        resp = requests.post(f"{base_url}/api/jobs/{job_id}/start", headers=headers, timeout=10)
        assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
        assert resp.json()["state"] == "queued"

    # --- Step 5: Poll ---
    def step_poll():
        timeout_s = 300
        start = time.time()
        while time.time() - start < timeout_s:
            resp = requests.get(f"{base_url}/api/jobs/{job_id}/status", headers=headers, timeout=10)
            assert resp.status_code == 200
            state = resp.json()["state"]
            if state == "completed":
                return
            if state == "failed":
                raise AssertionError(f"Job failed: {resp.json().get('error')}")
            time.sleep(3)
        raise AssertionError(f"Timed out after {timeout_s}s (state={state})")

    # --- Step 6: Download ---
    def step_download():
        nonlocal result_dir
        resp = requests.get(
            f"{base_url}/api/jobs/{job_id}/result", headers=headers, timeout=60, stream=True
        )
        assert resp.status_code == 200, f"HTTP {resp.status_code}"
        result_dir = tempfile.mkdtemp(prefix="incite_e2e_")
        tarball = Path(result_dir) / "result.tar.gz"
        with open(tarball, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        assert tarball.stat().st_size > 0

    # --- Step 7: Verify tarball ---
    def step_verify_tarball():
        tarball = Path(result_dir) / "result.tar.gz"
        extract = Path(result_dir) / "extracted"
        extract.mkdir(exist_ok=True)
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(path=extract)
        all_files = {p.name for p in extract.rglob("*") if p.is_file()}
        expected = {"chunks.jsonl", "index.faiss", "id_map.json", "chunk_to_paper.json"}
        missing = expected - all_files
        assert not missing, f"Missing: {missing}. Found: {all_files}"

    # --- Step 8: Verify chunks ---
    def step_verify_chunks():
        extract = Path(result_dir) / "extracted"
        chunks_file = next(extract.rglob("chunks.jsonl"))
        chunks = [json.loads(line) for line in open(chunks_file)]
        assert len(chunks) > 0, "No chunks"

    # --- Step 9: FAISS loadable ---
    def step_verify_faiss():
        import faiss

        extract = Path(result_dir) / "extracted"
        idx_file = next(extract.rglob("index.faiss"))
        index = faiss.read_index(str(idx_file))
        assert index.ntotal > 0, "Empty index"
        assert index.d == 384, f"Wrong dim: {index.d}"

    # --- Step 10: Cleanup ---
    def step_cleanup():
        resp = requests.delete(f"{base_url}/api/jobs/{job_id}", headers=headers, timeout=10)
        assert resp.status_code == 200

    # --- Run ---
    print("\ninCite Cloud E2E Test")
    print(f"Server: {base_url}")
    print(f"Auth:   {'API key set' if args.api_key else 'none (dev mode)'}\n")

    steps = [
        ("Health check", step_health),
        ("Create job", step_create),
        ("Upload PDFs", step_upload),
        ("Start processing", step_start),
        ("Poll until complete", step_poll),
        ("Download result", step_download),
        ("Verify tarball contents", step_verify_tarball),
        ("Verify chunks", step_verify_chunks),
        ("Verify FAISS index", step_verify_faiss),
        ("Cleanup job", step_cleanup),
    ]

    for name, fn in steps:
        if not run_step(name, fn):
            # Continue on failure to gather all results
            pass

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    _run_standalone()
