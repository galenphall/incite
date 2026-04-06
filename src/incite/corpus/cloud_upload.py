"""Web-based library upload client for the inCite cloud service.

Handles uploading papers and PDFs to the inciteref.com web service,
triggering server-side processing, and polling for completion.
Used by the CLI setup wizard when "cloud" source type is selected.

Related modules:
    - incite.corpus.cloud_client: CLI batch processing client (CloudProcessingClient).
    - incite.cli.setup: Setup wizard that invokes upload_library().
    - cloud/upload_worker.py: Server-side processing (not in src/).
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# Upload PDFs in batches of this size (5 ≈ 10MB per batch, avoids timeouts)
UPLOAD_BATCH_SIZE = 5

# Number of parallel upload workers
UPLOAD_WORKERS = 5

# Poll interval in seconds
POLL_INTERVAL = 5

# Maximum wait time for processing (2 hours)
MAX_WAIT_SECONDS = 7200


class CloudProcessingError(Exception):
    """Error during cloud processing."""


class WebUploadClient:
    """Client for uploading a local Zotero library to the web tier.

    Uploads paper metadata and PDFs, then triggers server-side processing
    (GROBID extraction, chunking, FAISS indexing).

    Usage:
        client = WebUploadClient(server_url, token)
        client.upload_library(papers, progress_callback=print)
    """

    def __init__(self, server_url: str, token: str) -> None:
        self.server_url = server_url.rstrip("/")
        self.token = token

    def _headers(self) -> dict:
        """Return authorization headers for API requests."""
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }

    def upload_library(
        self,
        papers: list,
        progress_callback=None,
    ) -> None:
        """Upload a full library and wait for processing to complete.

        Orchestrates the full upload workflow:
        1. Upload paper metadata as JSON
        2. Upload PDFs (skipping any already on the server)
        3. Trigger server-side processing (GROBID, chunking, indexing)
        4. Poll until processing completes

        Args:
            papers: List of Paper objects (with source_file for PDFs)
            progress_callback: Optional callback(message: str) for progress
        """

        def _log(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)

        # Step 1: Upload metadata
        _log(f"Uploading metadata for {len(papers)} papers...")
        resp_meta = self._upload_metadata(papers)
        _log(f"Library created (id={resp_meta['library_id']}, {resp_meta['num_papers']} papers)")

        # Step 2: Upload PDFs (skip already-uploaded ones)
        papers_with_pdf = [p for p in papers if p.source_file and Path(p.source_file).exists()]
        if papers_with_pdf:
            # Check what's already on the server
            already_uploaded = self._get_uploaded_pdfs()
            papers_to_upload = [p for p in papers_with_pdf if f"{p.id}.pdf" not in already_uploaded]
            if papers_to_upload:
                _log(
                    f"Uploading {len(papers_to_upload)} PDFs "
                    f"({len(already_uploaded)} already on server)..."
                )
                self._upload_pdfs(papers_to_upload, progress_callback=_log)
            else:
                _log(f"All {len(papers_with_pdf)} PDFs already on server")
        else:
            _log("No PDFs found locally (will use abstract-only chunks)")

        # Step 3: Start processing
        _log("Starting server-side processing...")
        resp_proc = self._start_processing()
        _log(f"Processing started (job_id={resp_proc['job_id']})")

        # Step 4: Poll until done
        _log("Waiting for processing to complete...")
        self._wait_for_completion(progress_callback=_log)
        _log("Library upload complete!")

    def _get_uploaded_pdfs(self) -> set[str]:
        """Query the server for PDFs already uploaded.

        Returns:
            Set of filenames (e.g. ``{"abc123.pdf", ...}``) already present
            on the server, so they can be skipped during upload.
        """
        try:
            resp = requests.get(
                f"{self.server_url}/api/v1/upload-library/pdfs",
                headers=self._headers(),
                timeout=30,
            )
            if resp.status_code == 200:
                return set(resp.json().get("pdfs", []))
        except Exception as e:
            logger.warning("Failed to query uploaded PDFs: %s", e)
        return set()

    def _upload_metadata(self, papers: list) -> dict:
        """Upload paper metadata as JSON.

        Args:
            papers: List of Paper objects whose metadata will be serialized.

        Returns:
            Server response dict containing ``library_id`` and ``num_papers``.

        Raises:
            CloudProcessingError: If the server returns a non-200 status.
        """
        paper_data = []
        for p in papers:
            paper_data.append(
                {
                    "id": p.id,
                    "title": p.title,
                    "abstract": p.abstract or "",
                    "authors": p.authors or [],
                    "year": p.year,
                    "doi": p.doi,
                    "journal": p.journal,
                    "bibtex_key": getattr(p, "bibtex_key", None),
                }
            )

        resp = requests.post(
            f"{self.server_url}/api/v1/upload-library",
            json={"papers": paper_data},
            headers=self._headers(),
            timeout=60,
        )
        if resp.status_code != 200:
            raise CloudProcessingError(
                f"Failed to upload metadata: {resp.status_code} {resp.text[:200]}"
            )
        return resp.json()

    def _upload_pdfs(self, papers: list, progress_callback=None) -> None:
        """Upload PDFs in batches of UPLOAD_BATCH_SIZE with parallel workers.

        Strategy:
        - Files are grouped into batches of ``UPLOAD_BATCH_SIZE`` (~10 MB each)
          to stay under typical reverse-proxy body-size limits.
        - Batches are submitted to a ``ThreadPoolExecutor`` with
          ``UPLOAD_WORKERS`` threads for concurrent I/O.
        - Fallback on 413 (payload too large): if a whole batch is rejected,
          each PDF in that batch is retried individually. Files that are *still*
          too large after the single-file retry are skipped and counted as
          ``skipped`` rather than raising an error.

        Args:
            papers: Papers whose PDFs should be uploaded (source_file must
                exist on disk).
            progress_callback: Optional callable(message: str) for progress.
        """
        total = len(papers)
        uploaded = 0
        skipped = 0

        # Build list of batches
        batches: list[list] = []
        for batch_start in range(0, total, UPLOAD_BATCH_SIZE):
            batch = papers[batch_start : batch_start + UPLOAD_BATCH_SIZE]
            batches.append(batch)

        def _upload_batch(batch: list) -> tuple[int, int]:
            """Upload a single batch, return (uploaded_count, skipped_count)."""
            batch_uploaded = 0
            batch_skipped = 0
            files = []
            for paper in batch:
                pdf_path = Path(paper.source_file)
                if pdf_path.exists():
                    files.append(
                        ("files", (f"{paper.id}.pdf", open(pdf_path, "rb"), "application/pdf"))
                    )

            if not files:
                return 0, 0

            try:
                resp = requests.post(
                    f"{self.server_url}/api/v1/upload-library/pdfs",
                    files=files,
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout=300,
                )
                if resp.status_code == 413 and len(files) > 1:
                    # Batch too large — close handles and retry one-at-a-time
                    for _, (_, fobj, _) in files:
                        fobj.close()
                    for paper in batch:
                        pdf_path = Path(paper.source_file)
                        if not pdf_path.exists():
                            continue
                        with open(pdf_path, "rb") as fobj:
                            single = [("files", (f"{paper.id}.pdf", fobj, "application/pdf"))]
                            r = requests.post(
                                f"{self.server_url}/api/v1/upload-library/pdfs",
                                files=single,
                                headers={"Authorization": f"Bearer {self.token}"},
                                timeout=300,
                            )
                        if r.status_code == 413:
                            # Individual file still too large — skip it
                            batch_skipped += 1
                            continue
                        r.raise_for_status()
                        batch_uploaded += 1
                    return batch_uploaded, batch_skipped

                resp.raise_for_status()
                batch_uploaded = len(files)
            finally:
                for _, (_, fobj, _) in files:
                    if not fobj.closed:
                        fobj.close()

            return batch_uploaded, batch_skipped

        # Upload batches in parallel using thread pool
        with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as executor:
            futures = {executor.submit(_upload_batch, batch): batch for batch in batches}
            for future in as_completed(futures):
                batch_uploaded, batch_skipped = future.result()
                uploaded += batch_uploaded
                skipped += batch_skipped
                if progress_callback:
                    msg = f"Uploaded {uploaded}/{total} PDFs"
                    if skipped:
                        msg += f" ({skipped} skipped — too large)"
                    progress_callback(msg)

    def _start_processing(self) -> dict:
        """Trigger server-side processing (GROBID, chunking, indexing).

        Returns:
            Server response dict containing ``job_id``.

        Raises:
            CloudProcessingError: If the server returns a non-200 status.
        """
        resp = requests.post(
            f"{self.server_url}/api/v1/upload-library/process",
            headers=self._headers(),
            timeout=30,
        )
        if resp.status_code != 200:
            raise CloudProcessingError(
                f"Failed to start processing: {resp.status_code} {resp.text[:200]}"
            )
        return resp.json()

    def _wait_for_completion(self, progress_callback=None) -> None:
        """Poll upload-library/status until processing completes or fails.

        Polling intervals:
        - Normal: every ``POLL_INTERVAL`` seconds (5 s).
        - On transient network errors (Timeout, ConnectionError): retry after
          ``POLL_INTERVAL`` without raising, because the server may be busy
          with CPU-intensive embedding work and briefly unresponsive.
        - Hard deadline: ``MAX_WAIT_SECONDS`` (2 hours) after which
          ``CloudProcessingError`` is raised unconditionally.

        Args:
            progress_callback: Optional callable(message: str) for progress.

        Raises:
            CloudProcessingError: On server-reported failure or timeout.
        """
        deadline = time.time() + MAX_WAIT_SECONDS
        start_time = time.monotonic()
        last_message = ""

        while time.time() < deadline:
            try:
                resp = requests.get(
                    f"{self.server_url}/api/v1/upload-library/status",
                    headers=self._headers(),
                    timeout=30,
                )
                resp.raise_for_status()
                status = resp.json()
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                # Server may be busy with CPU-intensive embedding; retry
                time.sleep(POLL_INTERVAL)
                continue

            lib_status = status.get("library_status", "unknown")
            job_status = status.get("job_status")

            if lib_status == "ready":
                if progress_callback:
                    num_papers = status.get("num_papers", 0)
                    num_chunks = status.get("num_chunks", 0)
                    fulltext_papers = status.get("grobid_fulltext_papers", 0)
                    fulltext_chunks = status.get("grobid_fulltext_chunks", 0)
                    abstract_papers = status.get("abstract_only_papers", 0)
                    msg = f"Done! {num_papers} papers, {num_chunks} chunks indexed"
                    if fulltext_papers or abstract_papers:
                        abstract_chunks = num_chunks - fulltext_chunks
                        msg += (
                            f" ({fulltext_chunks} from full-text, {abstract_chunks} from abstracts)"
                        )
                    progress_callback(msg)
                return

            if lib_status == "error" or job_status == "failed":
                error = status.get("error", "Unknown error")
                raise CloudProcessingError(f"Server processing failed: {error}")

            # Report progress with elapsed time
            if progress_callback:
                stage = status.get("stage", "")
                current = status.get("current", 0)
                total = status.get("total", 0)
                elapsed = time.monotonic() - start_time
                elapsed_str = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"

                if stage and total:
                    msg = f"  {stage}: {current}/{total} [{elapsed_str} elapsed]"
                elif stage:
                    msg = f"  {stage}... [{elapsed_str} elapsed]"
                else:
                    msg = f"  Processing... [{elapsed_str} elapsed]"

                if msg != last_message:
                    progress_callback(msg)
                    last_message = msg

            time.sleep(POLL_INTERVAL)

        raise CloudProcessingError(f"Processing timed out after {MAX_WAIT_SECONDS // 60} minutes")

    def get_diagnostics(self) -> dict:
        """Fetch library diagnostics from the server.

        Returns:
            Diagnostics dict (paper counts, chunk counts, job status, etc.)

        Raises:
            requests.HTTPError: If the server returns an error status.
        """
        resp = requests.get(
            f"{self.server_url}/api/v1/upload-library/diagnostics",
            headers=self._headers(),
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
