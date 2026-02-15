"""Processing pipeline abstraction for PDF → chunks → embeddings.

Provides a common interface for local and cloud processing, enabling
progressive migration from fully-local to cloud-hosted PDF processing.

Local pipeline: PyMuPDF extract → paragraph chunking → local embedding
Cloud pipeline: upload PDF → remote GROBID → download chunks → local embedding

Usage:
    pipeline = get_pipeline()  # auto-selects based on config
    chunks = pipeline.process_paper(paper)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from incite.models import Chunk, Paper

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStatus:
    """Status of a processing job."""

    paper_id: str
    state: str  # "pending", "processing", "completed", "failed"
    progress: float = 0.0  # 0.0 to 1.0
    message: str = ""
    num_chunks: int = 0
    error: Optional[str] = None


class ProcessingPipeline(ABC):
    """Abstract interface for PDF processing pipelines.

    Implementations handle the full path from PDF to indexed chunks,
    regardless of whether processing happens locally or remotely.
    """

    @abstractmethod
    def process_paper(
        self,
        paper: Paper,
        pdf_path: Optional[Path] = None,
    ) -> list[Chunk]:
        """Process a single paper into chunks.

        Args:
            paper: Paper with metadata (title, abstract, etc.)
            pdf_path: Path to PDF file. If None, uses paper.source_file.

        Returns:
            List of Chunk objects extracted from the paper.
            Returns empty list if PDF cannot be processed.
        """
        ...

    @abstractmethod
    def process_batch(
        self,
        papers: list[Paper],
        progress_callback=None,
    ) -> list[Chunk]:
        """Process multiple papers into chunks.

        Args:
            papers: List of papers to process
            progress_callback: Optional callback(current, total, message)

        Returns:
            Combined list of Chunk objects from all papers
        """
        ...

    @abstractmethod
    def get_status(self, paper_id: str) -> Optional[ProcessingStatus]:
        """Get processing status for a paper.

        Returns None if the paper has not been submitted for processing.
        """
        ...


class LocalPipeline(ProcessingPipeline):
    """Local processing pipeline using PyMuPDF and paragraph chunking.

    This is the existing processing path, wrapped in the Pipeline interface.
    """

    def __init__(self, chunking_strategy: str = "paragraph"):
        self.chunking_strategy = chunking_strategy

    def process_paper(
        self,
        paper: Paper,
        pdf_path: Optional[Path] = None,
    ) -> list[Chunk]:
        from incite.corpus.pdf_extractor import extract_pdf_text
        from incite.retrieval.factory import get_chunker

        path = pdf_path or (Path(paper.source_file) if paper.source_file else None)
        if path is None or not path.exists():
            # No PDF — create abstract-only chunk
            if paper.abstract:
                return [
                    Chunk(
                        id=f"{paper.id}::chunk_0",
                        paper_id=paper.id,
                        text=paper.abstract,
                        section="Abstract",
                    )
                ]
            return []

        # Extract text from PDF
        result = extract_pdf_text(path)
        if not result.full_text:
            if paper.abstract:
                return [
                    Chunk(
                        id=f"{paper.id}::chunk_0",
                        paper_id=paper.id,
                        text=paper.abstract,
                        section="Abstract",
                    )
                ]
            return []

        # Update paper with extracted text for chunking
        paper.full_text = result.full_text
        paper.paragraphs = result.paragraphs

        # Chunk the paper
        chunker = get_chunker(self.chunking_strategy)
        return chunker([paper], show_progress=False)

    def process_batch(
        self,
        papers: list[Paper],
        progress_callback=None,
    ) -> list[Chunk]:
        all_chunks = []
        for i, paper in enumerate(papers):
            chunks = self.process_paper(paper)
            all_chunks.extend(chunks)
            if progress_callback:
                progress_callback(i + 1, len(papers), f"Processed {paper.title[:40]}...")
        return all_chunks

    def get_status(self, paper_id: str) -> Optional[ProcessingStatus]:
        # Local processing is synchronous — no status tracking
        return None


class CloudPipeline(ProcessingPipeline):
    """Cloud processing pipeline that sends PDFs to a remote service.

    The remote service runs GROBID for high-quality extraction, chunks the
    result, and returns structured chunks. This pipeline handles upload,
    status polling, and chunk download.

    Designed for the hosted inCite service where users upload their
    library and get a cloud-processed, page-tagged vector DB.
    """

    def __init__(
        self,
        api_url: str,
        api_key: Optional[str] = None,
        timeout: int = 300,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _upload_pdf(self, paper: Paper, pdf_path: Path) -> Optional[str]:
        """Upload a PDF for processing. Returns job ID or None on failure."""
        url = f"{self.api_url}/api/process"
        try:
            # Use requests if available (for multipart upload)
            import requests

            with open(pdf_path, "rb") as f:
                resp = requests.post(
                    url,
                    files={"pdf": (pdf_path.name, f, "application/pdf")},
                    data={
                        "paper_id": paper.id,
                        "title": paper.title,
                        "abstract": paper.abstract or "",
                    },
                    headers=self._headers(),
                    timeout=self.timeout,
                )
            if resp.status_code == 200:
                return resp.json().get("job_id")
            logger.warning("Cloud upload failed: %s %s", resp.status_code, resp.text[:200])
            return None
        except Exception as e:
            logger.warning("Cloud upload error: %s", e)
            return None

    def _poll_status(self, job_id: str) -> ProcessingStatus:
        """Poll for job completion status."""
        import json
        import urllib.request

        url = f"{self.api_url}/api/status/{job_id}"
        try:
            req = urllib.request.Request(url, headers=self._headers())
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                return ProcessingStatus(
                    paper_id=data.get("paper_id", ""),
                    state=data.get("state", "unknown"),
                    progress=data.get("progress", 0.0),
                    message=data.get("message", ""),
                    num_chunks=data.get("num_chunks", 0),
                    error=data.get("error"),
                )
        except Exception as e:
            return ProcessingStatus(
                paper_id="",
                state="error",
                error=str(e),
            )

    def _download_chunks(self, job_id: str) -> list[Chunk]:
        """Download processed chunks for a completed job."""
        import json
        import urllib.request

        url = f"{self.api_url}/api/chunks/{job_id}"
        try:
            req = urllib.request.Request(url, headers=self._headers())
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                chunks = []
                for item in data.get("chunks", []):
                    chunks.append(
                        Chunk(
                            id=item["id"],
                            paper_id=item["paper_id"],
                            text=item["text"],
                            section=item.get("section"),
                            char_offset=item.get("char_offset", 0),
                            page_number=item.get("page_number"),
                            context_text=item.get("context_text"),
                            parent_text=item.get("parent_text"),
                        )
                    )
                return chunks
        except Exception as e:
            logger.warning("Failed to download chunks for job %s: %s", job_id, e)
            return []

    def process_paper(
        self,
        paper: Paper,
        pdf_path: Optional[Path] = None,
    ) -> list[Chunk]:
        import time

        path = pdf_path or (Path(paper.source_file) if paper.source_file else None)
        if path is None or not path.exists():
            return []

        # Upload PDF
        job_id = self._upload_pdf(paper, path)
        if job_id is None:
            logger.warning("Failed to upload %s to cloud pipeline", paper.id)
            return []

        # Poll until complete (with timeout)
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            status = self._poll_status(job_id)
            if status.state == "completed":
                return self._download_chunks(job_id)
            elif status.state == "failed":
                logger.warning("Cloud processing failed for %s: %s", paper.id, status.error)
                return []
            time.sleep(2)

        logger.warning("Cloud processing timed out for %s", paper.id)
        return []

    def process_batch(
        self,
        papers: list[Paper],
        progress_callback=None,
    ) -> list[Chunk]:
        """Upload all PDFs, then poll for all results."""
        import time

        # Phase 1: Upload all PDFs
        jobs: dict[str, str] = {}  # paper_id -> job_id
        for i, paper in enumerate(papers):
            path = Path(paper.source_file) if paper.source_file else None
            if path and path.exists():
                job_id = self._upload_pdf(paper, path)
                if job_id:
                    jobs[paper.id] = job_id
            if progress_callback:
                progress_callback(i + 1, len(papers), f"Uploading PDFs... ({i + 1}/{len(papers)})")

        # Phase 2: Poll for all completions
        all_chunks: list[Chunk] = []
        completed = set()
        deadline = time.time() + self.timeout

        while len(completed) < len(jobs) and time.time() < deadline:
            for paper_id, job_id in jobs.items():
                if paper_id in completed:
                    continue
                status = self._poll_status(job_id)
                if status.state in ("completed", "failed"):
                    completed.add(paper_id)
                    if status.state == "completed":
                        chunks = self._download_chunks(job_id)
                        all_chunks.extend(chunks)
                    if progress_callback:
                        progress_callback(
                            len(completed),
                            len(jobs),
                            f"Processing... ({len(completed)}/{len(jobs)})",
                        )
            if len(completed) < len(jobs):
                time.sleep(3)

        return all_chunks

    def get_status(self, paper_id: str) -> Optional[ProcessingStatus]:
        # Would need a paper_id -> job_id mapping (stored in config/DB)
        return None

    def check_health(self) -> bool:
        """Check if the cloud processing service is available."""
        import urllib.request

        try:
            req = urllib.request.Request(f"{self.api_url}/health", headers=self._headers())
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False


def get_pipeline(
    mode: Optional[str] = None,
    chunking_strategy: str = "paragraph",
) -> ProcessingPipeline:
    """Get the appropriate processing pipeline based on config.

    Args:
        mode: Override processing mode ("local", "cloud", or None for auto).
            If None, reads from ~/.incite/config.json.
        chunking_strategy: Chunking strategy for local pipeline.

    Returns:
        A ProcessingPipeline instance.
    """
    if mode is None:
        # Read from config
        try:
            from incite.webapp.state import get_config

            config = get_config()
            mode = config.get("processing", {}).get("mode", "local")
        except Exception:
            mode = "local"

    if mode == "cloud":
        try:
            from incite.webapp.state import get_config

            config = get_config()
            cloud_config = config.get("cloud", {})
            api_url = cloud_config.get("api_url", "")
            api_key = cloud_config.get("api_key", "")
            if api_url:
                return CloudPipeline(api_url=api_url, api_key=api_key)
        except Exception:
            pass
        logger.warning("Cloud pipeline not configured, falling back to local")

    return LocalPipeline(chunking_strategy=chunking_strategy)
