"""PDF acquisition pipeline orchestrator.

Tries sources in priority order: existing PDF -> Unpaywall OA -> arXiv ->
proxy + publisher URL -> proxy + DOI landing page (meta tag / page fallback).
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from incite.acquire.config import ProxyConfig
from incite.acquire.proxy import LibraryProxy, _is_pdf, create_proxy
from incite.acquire.resolver import doi_slug, resolve_doi
from incite.acquire.unpaywall import UnpaywallClient
from incite.models import Paper

logger = logging.getLogger(__name__)


@dataclass
class AcquisitionResult:
    """Result for a single paper acquisition attempt."""

    doi: str
    paper_id: str = ""
    title: str = ""
    status: str = ""  # "acquired", "skipped", "failed"
    source: str = ""  # "existing", "unpaywall_oa", "arxiv", "proxy", "proxy_doi"
    pdf_path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class AcquisitionSummary:
    """Summary of a batch acquisition run."""

    results: list[AcquisitionResult] = field(default_factory=list)

    @property
    def acquired(self) -> int:
        return sum(1 for r in self.results if r.status == "acquired")

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == "skipped")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == "failed")

    @property
    def by_source(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            if r.status == "acquired":
                counts[r.source] = counts.get(r.source, 0) + 1
        return counts


def _download_url(url: str, dest: Path) -> Optional[Path]:
    """Download a file from a URL, validating it's a PDF.

    Args:
        url: URL to download from.
        dest: Destination file path.

    Returns:
        Path to downloaded file, or None if failed or not a valid PDF.
    """
    import requests

    try:
        resp = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": "inCite/0.1 (mailto:research@umich.edu)"},
            stream=True,
        )
        resp.raise_for_status()

        content = resp.content
        if not _is_pdf(content):
            return None

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        return dest

    except Exception as e:
        logger.warning(f"Download failed for {url}: {e}")
        return None


class AcquisitionPipeline:
    """Orchestrates PDF acquisition from multiple sources.

    Sources are tried in priority order:
    1. Already have PDF? (check dest_dir and paper.source_file) -> skip
    2. Unpaywall OA -> direct download
    3. arXiv preprint (stub for v1, Unpaywall covers most arXiv)
    4. Proxy + Unpaywall publisher URL -> proxy download
    5. Proxy + DOI landing page -> session cascade (direct PDF / meta tag / page fallback)

    Args:
        dest_dir: Directory to save acquired PDFs.
        email: Email for Unpaywall API (required by their TOS).
        proxy_config: Proxy configuration (optional, skips proxy if None).
        free_only: If True, only use free sources (no proxy).
        dry_run: If True, print plan but don't download.
    """

    def __init__(
        self,
        dest_dir: Path,
        email: str,
        proxy_config: Optional[ProxyConfig] = None,
        free_only: bool = False,
        dry_run: bool = False,
    ):
        self.dest_dir = Path(dest_dir)
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self.email = email
        self.proxy_config = proxy_config
        self.free_only = free_only
        self.dry_run = dry_run

        self._unpaywall = UnpaywallClient(email=email)
        self._proxy: Optional[LibraryProxy] = None
        self._proxy_authenticated = False

    def _ensure_proxy(self) -> Optional[LibraryProxy]:
        """Lazy-initialize and authenticate proxy."""
        if self.free_only:
            return None

        if self._proxy is not None:
            return self._proxy if self._proxy_authenticated else None

        if self.proxy_config is None or not self.proxy_config.is_configured:
            return None

        try:
            self._proxy = create_proxy(self.proxy_config)
            self._proxy_authenticated = self._proxy.ensure_authenticated(interactive=True)
            if not self._proxy_authenticated:
                print("  Proxy authentication failed. Falling back to free sources only.")
                return None
            return self._proxy
        except Exception as e:
            logger.warning(f"Proxy initialization failed: {e}")
            return None

    def _pdf_path(self, doi: str) -> Path:
        """Get the destination path for a paper's PDF."""
        return self.dest_dir / f"{doi_slug(doi)}.pdf"

    def acquire_paper(self, paper: Paper) -> AcquisitionResult:
        """Acquire a PDF for a single paper, trying sources in order.

        Args:
            paper: Paper to acquire.

        Returns:
            AcquisitionResult with status and source information.
        """
        doi = paper.doi
        if not doi:
            return AcquisitionResult(
                doi="",
                paper_id=paper.id,
                title=paper.title,
                status="failed",
                error="No DOI",
            )

        dest = self._pdf_path(doi)

        # 1. Already have PDF?
        if dest.exists():
            return AcquisitionResult(
                doi=doi,
                paper_id=paper.id,
                title=paper.title,
                status="skipped",
                source="existing",
                pdf_path=str(dest),
            )

        if paper.source_file and Path(paper.source_file).exists():
            return AcquisitionResult(
                doi=doi,
                paper_id=paper.id,
                title=paper.title,
                status="skipped",
                source="existing",
                pdf_path=paper.source_file,
            )

        if self.dry_run:
            return AcquisitionResult(
                doi=doi,
                paper_id=paper.id,
                title=paper.title,
                status="acquired",
                source="dry_run",
            )

        # 2. Unpaywall lookup
        uw_result = self._unpaywall.lookup(doi)

        if uw_result and uw_result.is_oa and uw_result.best_oa_url:
            # Direct OA download
            downloaded = _download_url(uw_result.best_oa_url, dest)
            if downloaded:
                return AcquisitionResult(
                    doi=doi,
                    paper_id=paper.id,
                    title=paper.title,
                    status="acquired",
                    source="unpaywall_oa",
                    pdf_path=str(downloaded),
                )

        # 3. arXiv (stub for v1 -- Unpaywall covers most arXiv papers)
        # TODO: Add arXiv lookup via Semantic Scholar API

        # Steps 4-6 require proxy
        proxy = self._ensure_proxy()
        if proxy is None:
            # No proxy available -- can only report failure
            if uw_result and not uw_result.is_oa:
                return AcquisitionResult(
                    doi=doi,
                    paper_id=paper.id,
                    title=paper.title,
                    status="failed",
                    error="Paywalled (no proxy configured)",
                )
            return AcquisitionResult(
                doi=doi,
                paper_id=paper.id,
                title=paper.title,
                status="failed",
                error="Not found via free sources",
            )

        # 4. Proxy + Unpaywall publisher URL
        if uw_result and uw_result.publisher_pdf_url:
            downloaded = proxy.download_pdf(uw_result.publisher_pdf_url, dest)
            if downloaded:
                return AcquisitionResult(
                    doi=doi,
                    paper_id=paper.id,
                    title=paper.title,
                    status="acquired",
                    source="proxy",
                    pdf_path=str(downloaded),
                )

        # 5. Proxy + DOI resolution -> landing URL -> session cascade
        #    (session internally handles: direct PDF / meta tag / page fallback)
        landing_url = resolve_doi(doi)
        if landing_url:
            downloaded = proxy.download_pdf(landing_url, dest)
            if downloaded:
                return AcquisitionResult(
                    doi=doi,
                    paper_id=paper.id,
                    title=paper.title,
                    status="acquired",
                    source="proxy_doi",
                    pdf_path=str(downloaded),
                )

        return AcquisitionResult(
            doi=doi,
            paper_id=paper.id,
            title=paper.title,
            status="failed",
            error="All sources exhausted",
        )

    def acquire_batch(
        self,
        papers: list[Paper],
        limit: Optional[int] = None,
    ) -> AcquisitionSummary:
        """Acquire PDFs for a batch of papers.

        Args:
            papers: List of papers to acquire.
            limit: Maximum number of papers to process (None = all).

        Returns:
            AcquisitionSummary with per-paper results.
        """
        if limit is not None:
            papers = papers[:limit]

        if not papers:
            print("No papers to acquire.")
            return AcquisitionSummary()

        # Pre-check proxy auth before starting batch (if proxy is configured)
        if not self.free_only and self.proxy_config and self.proxy_config.is_configured:
            print("Checking proxy authentication...")
            self._ensure_proxy()

        summary = AcquisitionSummary()

        for i, paper in enumerate(papers, 1):
            # Print per-paper status line
            authors = ""
            if paper.authors:
                first = paper.authors[0].split()[-1] if paper.authors[0] else ""
                if len(paper.authors) > 1:
                    authors = f"{first} et al."
                else:
                    authors = first
            year_str = f" ({paper.year})" if paper.year else ""
            title_short = paper.title[:50] + ("..." if len(paper.title) > 50 else "")
            print(f'\n  {i}/{len(papers)}  {authors}{year_str} "{title_short}"')

            if paper.doi:
                print(f"        DOI: {paper.doi}")

            result = self.acquire_paper(paper)
            summary.results.append(result)

            # Print result
            if result.status == "acquired":
                print(f"        -> {result.source}: acquired")
            elif result.status == "skipped":
                print("        -> Already have PDF")
            else:
                print(f"        -> Failed: {result.error}")

        # Save manifest
        if not self.dry_run:
            self._save_manifest(summary)

        return summary

    def _save_manifest(self, summary: AcquisitionSummary) -> None:
        """Save/merge acquisition manifest to dest_dir/manifest.json.

        The manifest maps DOIs to file paths and Zotero keys, enabling
        future attachment back to Zotero items.
        """
        manifest_path = self.dest_dir / "manifest.json"

        # Load existing manifest if present
        existing: dict = {}
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text())
            except (json.JSONDecodeError, OSError):
                existing = {}

        acquired = existing.get("acquired", [])
        failed = existing.get("failed", [])

        # Existing DOIs for dedup
        existing_dois = {entry.get("doi") for entry in acquired}
        existing_failed_dois = {entry.get("doi") for entry in failed}

        for result in summary.results:
            if result.status == "acquired" and result.doi not in existing_dois:
                acquired.append(
                    {
                        "doi": result.doi,
                        "paper_id": result.paper_id,
                        "title": result.title,
                        "pdf_path": result.pdf_path,
                        "source": result.source,
                        "acquired_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
            elif (
                result.status == "failed" and result.doi and result.doi not in existing_failed_dois
            ):
                failed.append(
                    {
                        "doi": result.doi,
                        "paper_id": result.paper_id,
                        "title": result.title,
                        "reason": result.error or "unknown",
                    }
                )

        manifest = {"acquired": acquired, "failed": failed}

        # Atomic write
        fd, tmp_path = tempfile.mkstemp(dir=self.dest_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(manifest, f, indent=2)
            Path(tmp_path).replace(manifest_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def close(self):
        """Clean up resources."""
        if self._proxy is not None:
            self._proxy.close()
