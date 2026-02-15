"""Cloud processing CLI commands for inCite.

Provides:
- `incite cloud-process` — upload library to cloud for GROBID extraction
- `incite upload-library` — upload local Zotero library to the web tier
"""

import sys

from incite.cli._shared import EMBEDDER_CHOICES


def register(subparsers):
    """Register cloud subcommands."""
    # Existing: cloud-process (batch job API)
    p = subparsers.add_parser(
        "cloud-process",
        help="Process library PDFs via the inCite cloud service (GROBID + embedding)",
    )
    p.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="Cloud service URL (default: from config or https://inciteref.com)",
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (default: from config)",
    )
    p.add_argument(
        "--embedder",
        type=str,
        choices=EMBEDDER_CHOICES,
        default="minilm-ft",
        help="Embedder model for index building (default: minilm-ft)",
    )
    p.set_defaults(func=cmd_cloud_process)

    # New: upload-library (web tier upload)
    p2 = subparsers.add_parser(
        "upload-library",
        help="Upload your local Zotero library to the inCite web service",
    )
    p2.add_argument(
        "--server",
        type=str,
        default="https://inciteref.com",
        help="Server URL (default: https://inciteref.com)",
    )
    p2.add_argument(
        "--token",
        type=str,
        required=True,
        help="API token (generate from your account page at inciteref.com)",
    )
    p2.add_argument(
        "--diagnostics",
        action="store_true",
        default=False,
        help="Show diagnostics about your library's processing results instead of uploading",
    )
    p2.set_defaults(func=cmd_upload_library)


def _load_zotero_papers() -> list:
    """Load papers from local Zotero, auto-detecting the data directory."""
    from pathlib import Path

    from incite.corpus.zotero_reader import find_zotero_data_dir
    from incite.webapp.state import get_config, load_zotero_direct

    # Try config first, then auto-detect
    config = get_config()
    zotero_config = config.get("zotero", {})
    zotero_dir = zotero_config.get("data_dir", "")

    if not zotero_dir:
        detected = find_zotero_data_dir()
        if detected is None:
            raise FileNotFoundError(
                "Could not find Zotero data directory. Run 'incite setup' to configure it."
            )
        zotero_dir = str(detected)

    return load_zotero_direct(Path(zotero_dir))


def cmd_cloud_process(args):
    """Upload library to cloud service for GROBID processing."""
    from incite.corpus.cloud_client import CloudProcessingClient, CloudProcessingError
    from incite.webapp.state import get_config, save_config

    # Resolve API URL and key from args > config > default
    config = get_config()
    cloud_config = config.get("cloud", {})

    api_url = args.api_url or cloud_config.get("api_url", "https://inciteref.com")
    api_key = args.api_key or cloud_config.get("api_key", "")

    if not api_key:
        print("Error: No API key provided.")
        print("Use --api-key or set it via 'incite setup' with cloud processing.")
        sys.exit(1)

    # Load papers
    print("Loading papers from Zotero corpus...")
    try:
        papers = _load_zotero_papers()
    except Exception as e:
        print(f"Error loading papers: {e}")
        print("Run 'incite setup' first to configure your library.")
        sys.exit(1)

    n_pdfs = sum(1 for p in papers if p.source_file)
    print(f"Found {len(papers)} papers ({n_pdfs} with PDFs)")

    if not papers:
        print("No papers found. Run 'incite setup' first.")
        sys.exit(1)

    # Check service health
    client = CloudProcessingClient(api_url=api_url, api_key=api_key)
    try:
        health = client.check_health()
        if health.get("status") == "unhealthy":
            print(f"Warning: Cloud service reports unhealthy status: {health}")
        elif not health.get("grobid_alive"):
            print("Warning: GROBID is not running on the cloud server. Processing may fail.")
        else:
            print(f"Cloud service healthy (GROBID alive, {health.get('disk_free_gb', '?')}GB free)")
    except CloudProcessingError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Process
    def _progress(msg):
        print(f"  {msg}")

    try:
        result_dir = client.process_library(
            papers=papers,
            embedder=args.embedder,
            progress_callback=_progress,
        )
    except CloudProcessingError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    # Update config to reflect cloud processing
    config["processing"] = {"mode": "cloud"}
    config["cloud"] = {"api_url": api_url, "api_key": api_key}
    config["embedder"] = args.embedder
    save_config(config)

    print()
    print("Cloud processing complete!")
    print(f"  Chunks + FAISS index installed at: {result_dir}")
    print()
    print("Next steps:")
    print(f"  incite serve --embedder {args.embedder}   # start API server")
    print('  incite recommend "your text" -k 5       # test recommendations')


def cmd_upload_library(args):
    """Upload local Zotero library to the inCite web tier."""
    from incite.corpus.cloud_client import CloudProcessingError, WebUploadClient

    client = WebUploadClient(server_url=args.server, token=args.token)

    if args.diagnostics:
        try:
            diag = client.get_diagnostics()
        except Exception as e:
            print(f"Error fetching diagnostics: {e}")
            sys.exit(1)

        print("Library Diagnostics")
        print("=" * 40)
        print(f"  Status:              {diag.get('library_status', 'unknown')}")
        print(f"  Last job status:     {diag.get('last_job_status', 'N/A')}")
        if diag.get("last_job_error"):
            print(f"  Last job error:      {diag['last_job_error']}")
        print()
        print(f"  Total papers:        {diag.get('total_papers', 0)}")
        print(f"  Total chunks:        {diag.get('total_chunks', 0)}")
        print(f"    Abstract chunks:   {diag.get('abstract_chunks', 0)}")
        print(f"    Full-text chunks:  {diag.get('fulltext_chunks', 0)}")
        print(f"  PDFs on disk:        {diag.get('pdfs_on_disk', 0)}")
        print(f"  GROBID cache entries:{diag.get('grobid_cache_entries', 0)}")
        sections = diag.get("section_names", {})
        if sections:
            print()
            print("  Section breakdown:")
            for section, count in sorted(sections.items(), key=lambda x: -x[1]):
                label = section if section else "(empty)"
                print(f"    {label}: {count}")
        return

    # Load papers from local Zotero
    print("Reading local Zotero library...")
    try:
        papers = _load_zotero_papers()
    except Exception as e:
        print(f"Error loading Zotero library: {e}")
        print("Make sure Zotero is installed and has papers in it.")
        print("Run 'incite setup' first if you haven't configured your library.")
        sys.exit(1)

    if not papers:
        print("No papers found in your Zotero library.")
        sys.exit(1)

    n_pdfs = sum(1 for p in papers if p.source_file)
    print(f"Found {len(papers)} papers ({n_pdfs} with local PDFs)")

    # Upload
    def _progress(msg):
        print(f"  {msg}")

    try:
        client.upload_library(papers=papers, progress_callback=_progress)
    except CloudProcessingError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    print()
    print("Your library is now available on the web!")
    print(f"  Visit {args.server} to get citation recommendations.")
