"""Client for the inCite cloud processing service.

Handles the full batch workflow:
1. Create job with paper metadata
2. Upload PDFs in batches
3. Start processing
4. Poll for completion
5. Download result tarball
6. Unpack into ~/.incite/
"""

import logging
import shutil
import tarfile
import time
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


class CloudProcessingClient:
    """Client for batch cloud processing of PDF libraries.

    Usage:
        client = CloudProcessingClient(api_url, api_key)
        result_dir = client.process_library(papers, embedder="minilm-ft")
    """

    def __init__(self, api_url: str, api_key: str = ""):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> dict:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def check_health(self) -> dict:
        """Check if the cloud service is healthy.

        Returns:
            Health response dict with status, grobid_alive, etc.

        Raises:
            CloudProcessingError: If service is unreachable
        """
        try:
            resp = requests.get(f"{self.api_url}/health", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            raise CloudProcessingError(f"Cloud service unreachable: {e}") from e

    def process_library(
        self,
        papers: list,
        embedder: str = "minilm-ft",
        progress_callback=None,
    ) -> Path:
        """Process a full library through the cloud service.

        Args:
            papers: List of Paper objects (with source_file for PDFs)
            embedder: Embedder type to use for index building
            progress_callback: Optional callback(message: str) for progress updates

        Returns:
            Path to the unpacked result directory

        Raises:
            CloudProcessingError: On any processing failure
        """

        def _log(msg):
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)

        # Step 1: Create job
        _log("Creating cloud processing job...")
        job_id = self._create_job(papers, embedder)
        _log(f"Job created: {job_id}")

        try:
            # Step 2: Upload PDFs
            papers_with_pdf = [p for p in papers if p.source_file and Path(p.source_file).exists()]
            if papers_with_pdf:
                _log(f"Uploading {len(papers_with_pdf)} PDFs...")
                self._upload_pdfs(job_id, papers_with_pdf, progress_callback=_log)

            # Step 3: Start processing
            _log("Starting cloud processing...")
            self._start_job(job_id)

            # Step 4: Poll for completion
            _log("Processing (this may take several minutes)...")
            self._wait_for_completion(job_id, progress_callback=_log)

            # Step 5: Download result
            _log("Downloading result...")
            tarball_path = self._download_result(job_id)

            # Step 6: Unpack
            _log("Unpacking result...")
            result_dir = self._unpack_result(tarball_path, embedder)

            _log(f"Cloud processing complete. Results in {result_dir}")
            return result_dir

        except Exception:
            # Try to clean up the job on failure
            try:
                self._delete_job(job_id)
            except Exception as e:
                logger.warning("Failed to clean up job %s: %s", job_id, e)
            raise

    def _create_job(self, papers: list, embedder: str) -> str:
        """Create a processing job and return the job ID."""
        paper_data = []
        for p in papers:
            has_pdf = bool(p.source_file and Path(p.source_file).exists())
            paper_data.append(
                {
                    "id": p.id,
                    "title": p.title,
                    "abstract": p.abstract or "",
                    "authors": p.authors or [],
                    "year": p.year,
                    "doi": p.doi,
                    "journal": p.journal,
                    "has_pdf": has_pdf,
                }
            )

        resp = requests.post(
            f"{self.api_url}/api/jobs",
            json={"papers": paper_data, "embedder": embedder},
            headers=self._headers(),
            timeout=30,
        )
        if resp.status_code != 200:
            raise CloudProcessingError(
                f"Failed to create job: {resp.status_code} {resp.text[:200]}"
            )

        return resp.json()["job_id"]

    def _upload_pdfs(self, job_id: str, papers: list, progress_callback=None):
        """Upload PDFs in batches."""
        total = len(papers)
        uploaded = 0

        for batch_start in range(0, total, UPLOAD_BATCH_SIZE):
            batch = papers[batch_start : batch_start + UPLOAD_BATCH_SIZE]
            files = []
            for paper in batch:
                pdf_path = Path(paper.source_file)
                if pdf_path.exists():
                    files.append(
                        ("files", (f"{paper.id}.pdf", open(pdf_path, "rb"), "application/pdf"))
                    )

            if not files:
                continue

            try:
                resp = requests.post(
                    f"{self.api_url}/api/jobs/{job_id}/upload",
                    files=files,
                    headers=self._headers(),
                    timeout=120,
                )
                resp.raise_for_status()
                uploaded += len(files)

                if progress_callback:
                    progress_callback(f"Uploaded {uploaded}/{total} PDFs")
            finally:
                # Close file handles
                for _, (_, fobj, _) in files:
                    fobj.close()

    def _start_job(self, job_id: str):
        """Start processing a job."""
        resp = requests.post(
            f"{self.api_url}/api/jobs/{job_id}/start",
            headers=self._headers(),
            timeout=30,
        )
        if resp.status_code != 200:
            raise CloudProcessingError(f"Failed to start job: {resp.status_code} {resp.text[:200]}")

    def _wait_for_completion(
        self,
        job_id: str,
        progress_callback=None,
    ):
        """Poll job status until completion or timeout."""
        deadline = time.time() + MAX_WAIT_SECONDS
        start_time = time.monotonic()
        last_message = ""

        while time.time() < deadline:
            resp = requests.get(
                f"{self.api_url}/api/jobs/{job_id}/status",
                headers=self._headers(),
                timeout=15,
            )
            resp.raise_for_status()
            status = resp.json()

            state = status.get("state", "unknown")

            if state == "completed":
                return

            if state == "failed":
                error = status.get("error", "Unknown error")
                raise CloudProcessingError(f"Cloud processing failed: {error}")

            if state == "cancelled":
                raise CloudProcessingError("Job was cancelled")

            # Report progress with elapsed time and ETA
            progress = status.get("progress")
            if progress and progress_callback:
                msg = progress.get("message", "")
                processed = progress.get("papers_processed", 0)
                total = progress.get("total_papers", 0)

                elapsed = time.monotonic() - start_time
                elapsed_str = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"

                # Compute ETA if we have progress data
                eta_str = ""
                if processed and total and processed > 0:
                    rate = elapsed / processed
                    remaining = rate * (total - processed)
                    eta_str = f", ~{int(remaining // 60)}:{int(remaining % 60):02d} remaining"
                    msg = f"{msg} ({processed}/{total} papers{eta_str})"

                msg_with_time = f"{msg} [{elapsed_str} elapsed]"
                if msg_with_time != last_message:
                    progress_callback(msg_with_time)
                    last_message = msg_with_time
            elif progress_callback:
                # No progress info from server, just show elapsed time
                elapsed = time.monotonic() - start_time
                if elapsed > 10:
                    elapsed_str = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"
                    tick_msg = f"Processing... [{elapsed_str} elapsed]"
                    if tick_msg != last_message:
                        progress_callback(tick_msg)
                        last_message = tick_msg

            time.sleep(POLL_INTERVAL)

        raise CloudProcessingError(f"Processing timed out after {MAX_WAIT_SECONDS // 60} minutes")

    def _download_result(self, job_id: str) -> Path:
        """Download the result tarball to a temp location."""
        from incite.webapp.state import get_cache_dir

        cache_dir = get_cache_dir()
        tarball_path = cache_dir / f"cloud_result_{job_id}.tar.gz"

        resp = requests.get(
            f"{self.api_url}/api/jobs/{job_id}/result",
            headers=self._headers(),
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()

        with open(tarball_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        return tarball_path

    def _unpack_result(self, tarball_path: Path, embedder: str) -> Path:
        """Unpack result tarball into ~/.incite/.

        Installs:
        - chunks.jsonl -> ~/.incite/zotero_chunks_{embedder}.jsonl
        - FAISS files -> ~/.incite/zotero_chunks_{embedder}/
        """
        from incite.webapp.state import get_cache_dir

        cache_dir = get_cache_dir()

        # Target paths
        chunks_dest = cache_dir / f"zotero_chunks_{embedder}.jsonl"
        index_dest = cache_dir / f"zotero_chunks_{embedder}"
        index_dest.mkdir(exist_ok=True)

        # Backup existing files before overwriting
        if chunks_dest.exists():
            backup = chunks_dest.with_suffix(".jsonl.bak")
            shutil.copy2(chunks_dest, backup)
            logger.info("Backed up existing chunks to %s", backup)

        # Extract tarball
        with tarfile.open(tarball_path, "r:gz") as tar:
            for member in tar.getmembers():
                # Security: prevent path traversal
                if member.name.startswith("/") or ".." in member.name:
                    continue

                if member.name == "chunks.jsonl":
                    tar.extract(member, path=str(cache_dir))
                    # Move to the correct filename
                    extracted = cache_dir / "chunks.jsonl"
                    if extracted.exists():
                        shutil.move(str(extracted), str(chunks_dest))
                elif member.name in (
                    "index.faiss",
                    "id_map.json",
                    "chunk_to_paper.json",
                    "job_info.json",
                ):
                    tar.extract(member, path=str(index_dest))

        # Clean up tarball
        tarball_path.unlink(missing_ok=True)

        return index_dest

    def _delete_job(self, job_id: str):
        """Delete a job from the server."""
        try:
            requests.delete(
                f"{self.api_url}/api/jobs/{job_id}",
                headers=self._headers(),
                timeout=10,
            )
        except Exception as e:
            logger.warning("Failed to delete job %s: %s", job_id, e)


class WebUploadClient:
    """Client for uploading a local Zotero library to the web tier.

    Uploads paper metadata and PDFs, then triggers server-side processing
    (GROBID extraction, chunking, FAISS indexing).

    Usage:
        client = WebUploadClient(server_url, token)
        client.upload_library(papers, progress_callback=print)
    """

    def __init__(self, server_url: str, token: str):
        self.server_url = server_url.rstrip("/")
        self.token = token

    def _headers(self) -> dict:
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
        """Query the server for PDFs already uploaded. Returns a set of filenames."""
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
        """Upload paper metadata as JSON."""
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

        Uses a thread pool (UPLOAD_WORKERS) to upload multiple batches
        concurrently. Falls back to one-at-a-time uploads on 413 (payload
        too large) errors, and skips individual files that still exceed the
        server limit.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

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
        """Trigger server-side processing."""
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
        """Poll upload-library/status until processing completes or fails."""
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

            # Report progress
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
        """Fetch library diagnostics from the server."""
        resp = requests.get(
            f"{self.server_url}/api/v1/upload-library/diagnostics",
            headers=self._headers(),
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
