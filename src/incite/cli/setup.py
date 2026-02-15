"""Setup wizard: guided first-run experience for inCite."""

import json
import sys
from pathlib import Path

from incite.cli._shared import EMBEDDER_CHOICES


def register(subparsers):
    """Register the setup command."""
    p = subparsers.add_parser(
        "setup",
        help="Interactive setup wizard for first-run configuration",
    )
    p.add_argument(
        "--zotero",
        action="store_true",
        help="Non-interactive: use auto-detected Zotero library",
    )
    p.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Non-interactive: use a folder of PDFs at this path",
    )
    p.add_argument(
        "--embedder",
        type=str,
        choices=EMBEDDER_CHOICES,
        default="minilm",
        help="Embedder model (default: minilm)",
    )
    p.set_defaults(func=cmd_setup)


def cmd_setup(args):
    """Run the setup wizard."""
    print()
    print("Welcome to inCite!")
    print("=" * 40)
    print()

    # Determine source: non-interactive flags take priority
    if args.folder:
        source_config = _setup_folder(args.folder)
    elif args.zotero:
        source_config = _setup_zotero_noninteractive()
    else:
        source_config = _setup_interactive()

    if source_config is None:
        sys.exit(1)

    # Check if cloud processing was selected
    is_cloud = source_config.pop("processing", None) == "cloud"
    cloud_api_url = source_config.pop("cloud_api_url", "")
    cloud_api_key = source_config.pop("cloud_api_key", "")

    # Step 2: Load papers
    print()
    print("Step 2: Loading papers...")
    papers = _load_papers(source_config)
    if papers is None:
        sys.exit(1)

    n_abstracts = sum(1 for p in papers if p.abstract)
    n_pdfs = sum(1 for p in papers if p.source_file)
    print(f"  Found {len(papers)} papers ({n_abstracts} with abstracts, {n_pdfs} with PDFs)")

    embedder_type = args.embedder

    if is_cloud:
        # Cloud processing path
        print()
        print("Step 3: Cloud processing...")
        _run_cloud_processing(papers, cloud_api_url, cloud_api_key, embedder_type)

        config = {
            "source": source_config,
            "embedder": embedder_type,
            "method": "hybrid",
            "processing": {"mode": "cloud"},
            "cloud": {"api_url": cloud_api_url, "api_key": cloud_api_key},
        }
        _save_setup_config(config)
    else:
        # Local processing path
        print()
        print("Step 3: Building search index...")
        retriever = _build_index(papers, embedder_type)
        if retriever is None:
            sys.exit(1)

        # Step 4: Test query
        print()
        _test_query(retriever, papers)

        config = {
            "source": source_config,
            "embedder": embedder_type,
            "method": "hybrid",
            "processing": {"mode": "local"},
        }
        _save_setup_config(config)

    # Done
    print()
    print("Setup complete! Next steps:")
    print("  incite serve    -- start the API server")
    print("  incite tray     -- start the menu bar app (macOS)")
    print('  incite recommend "your text" -- test from the command line')
    print(f"  (saved embedder={embedder_type}, method=hybrid to config)")
    print()


# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------


def _setup_interactive():
    """Interactive source selection (Step 1)."""
    from incite.corpus.zotero_reader import find_zotero_data_dir, get_library_stats

    print("Step 1: Where are your papers?")

    zotero_dir = find_zotero_data_dir()
    if zotero_dir:
        stats = get_library_stats(zotero_dir)
        if "error" not in stats:
            total = stats["total_papers"]
            pdfs = stats["with_pdfs"]
            print(
                f"  [Auto-detect] Found Zotero at {zotero_dir} ({total} papers, {pdfs} with PDFs)"
            )
        else:
            print(f"  [Auto-detect] Found Zotero at {zotero_dir} (could not read stats)")

    print()
    print("Options:")
    if zotero_dir:
        print("  1. Use detected Zotero library (local processing)")
        print("  2. Enter path to a folder of PDFs (local processing)")
        print("  3. Use Paperpile library (BibTeX sync)")
        print("  4. Use cloud processing (recommended for large libraries)")
    else:
        print("  Zotero not detected.")
        print("  1. Enter path to a folder of PDFs (local processing)")
        print("  2. Use Paperpile library (BibTeX sync)")
        print("  3. Use cloud processing (recommended for large libraries)")

    print()
    try:
        choice = input("Choose [1]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        print("\nSetup cancelled.")
        return None

    # Map choice to action based on whether Zotero was detected
    if zotero_dir:
        if choice == "1":
            return {"type": "zotero", "path": str(zotero_dir)}
        elif choice == "2":
            try:
                folder = input("Enter folder path: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSetup cancelled.")
                return None
            return _setup_folder(folder)
        elif choice == "3":
            return _setup_paperpile()
        elif choice == "4":
            return _setup_cloud(zotero_dir)
        else:
            return {"type": "zotero", "path": str(zotero_dir)}
    else:
        if choice == "1":
            try:
                folder = input("Enter folder path: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSetup cancelled.")
                return None
            return _setup_folder(folder)
        elif choice == "2":
            return _setup_paperpile()
        elif choice == "3":
            return _setup_cloud(None)
        else:
            try:
                folder = input("Enter folder path: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSetup cancelled.")
                return None
            return _setup_folder(folder)


def _setup_zotero_noninteractive():
    """Non-interactive Zotero setup."""
    from incite.corpus.zotero_reader import find_zotero_data_dir

    zotero_dir = find_zotero_data_dir()
    if zotero_dir is None:
        print("Error: Could not auto-detect Zotero directory.")
        print("Make sure Zotero is installed, or use --folder instead.")
        return None

    print(f"Step 1: Using Zotero library at {zotero_dir}")
    return {"type": "zotero", "path": str(zotero_dir)}


def _setup_folder(folder_path):
    """Set up a folder source, validating the path."""
    folder = Path(folder_path).expanduser().resolve()
    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        return None
    if not folder.is_dir():
        print(f"Error: Not a directory: {folder}")
        return None

    # Check for PDFs
    pdfs = list(folder.rglob("*.pdf"))
    if not pdfs:
        print(f"Error: No PDF files found in {folder}")
        return None

    print(f"Step 1: Using folder at {folder} ({len(pdfs)} PDFs)")
    return {"type": "folder", "path": str(folder)}


def _setup_paperpile():
    """Set up Paperpile source, prompting for BibTeX URL and optional PDF folder."""
    print()
    print("Paperpile syncs your library as a BibTeX file.")
    print("In Paperpile: Settings > Workflows > BibTeX export > Copy URL")
    print()

    try:
        bibtex_url = input("BibTeX URL: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nSetup cancelled.")
        return None

    if not bibtex_url:
        print("Error: BibTeX URL is required.")
        return None

    # Optional PDF folder
    try:
        pdf_folder = input("PDF folder path (press Enter to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        pdf_folder = ""

    # Save paperpile-specific config
    from incite.webapp.state import get_config, save_config

    config = get_config()
    config["paperpile"] = {
        "bibtex_url": bibtex_url,
        "bibtex_path": "",
        "pdf_folder": pdf_folder,
    }
    save_config(config)

    print("Step 1: Using Paperpile library (BibTeX sync)")
    return {"type": "paperpile", "bibtex_url": bibtex_url, "pdf_folder": pdf_folder}


def _setup_cloud(zotero_dir):
    """Set up cloud processing, prompting for API URL and key."""
    print()
    print("Cloud processing uses a remote server with GROBID for")
    print("high-quality PDF extraction. You need an API key.")
    print()

    try:
        api_url = input("API URL [https://inciteref.com]: ").strip()
        api_url = api_url or "https://inciteref.com"
        api_key = input("API key: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nSetup cancelled.")
        return None

    if not api_key:
        print("Error: API key is required for cloud processing.")
        return None

    source_config = {"type": "zotero", "path": str(zotero_dir)} if zotero_dir else None
    if source_config is None:
        try:
            folder = input("Enter path to your PDFs (for paper metadata): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSetup cancelled.")
            return None
        source_config = _setup_folder(folder)
        if source_config is None:
            return None

    # Mark as cloud processing
    source_config["processing"] = "cloud"
    source_config["cloud_api_url"] = api_url
    source_config["cloud_api_key"] = api_key
    return source_config


# ---------------------------------------------------------------------------
# Paper loading
# ---------------------------------------------------------------------------


def _load_papers(source_config):
    """Load papers from the configured source."""
    import sqlite3

    source_type = source_config["type"]

    try:
        if source_type == "zotero":
            from incite.webapp.state import load_zotero_direct

            return load_zotero_direct(Path(source_config["path"]))
        elif source_type == "paperpile":
            from incite.corpus.paperpile_source import PaperpileSource

            bibtex_url = source_config.get("bibtex_url")
            pdf_folder = source_config.get("pdf_folder")
            source = PaperpileSource(
                bibtex_url=bibtex_url or None,
                pdf_folder=Path(pdf_folder) if pdf_folder else None,
            )
            return source.load_papers()
        else:
            from incite.corpus.folder_source import FolderCorpusSource

            source = FolderCorpusSource(source_config["path"])
            return source.load_papers()
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower():
            print("Error: Zotero database is locked. Close Zotero and try again.")
        else:
            print(f"Error reading Zotero database: {e}")
        return None
    except PermissionError:
        print(
            "Error: Permission denied reading your library. "
            "Check file permissions on your Zotero data directory."
        )
        return None
    except FileNotFoundError as e:
        print(f"Error: File not found — {e}")
        return None
    except Exception as e:
        print(f"Error loading papers: {_user_friendly_error('load papers', e)}")
        return None


# ---------------------------------------------------------------------------
# Cloud processing
# ---------------------------------------------------------------------------


def _run_cloud_processing(papers, api_url, api_key, embedder_type):
    """Run cloud processing for the paper library."""
    import requests

    from incite.corpus.cloud_client import CloudProcessingClient, CloudProcessingError

    client = CloudProcessingClient(api_url=api_url, api_key=api_key)

    # Check health first
    try:
        health = client.check_health()
        if not health.get("grobid_alive"):
            print("  Warning: GROBID is not available on the cloud server.")
    except (ConnectionError, requests.exceptions.ConnectionError):
        print(f"  Error: Cannot connect to cloud service at {api_url}")
        print("  Check your internet connection and verify the API URL is correct.")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"  Error: Connection to {api_url} timed out.")
        print("  The server may be down. Try again later.")
        sys.exit(1)
    except CloudProcessingError as e:
        print(f"  Error connecting to cloud service: {e}")
        sys.exit(1)

    import time

    _cloud_start = time.monotonic()

    def _progress(msg):
        elapsed = time.monotonic() - _cloud_start
        if elapsed > 5:
            mins, secs = divmod(int(elapsed), 60)
            print(f"  {msg} [{mins}:{secs:02d} elapsed]")
        else:
            print(f"  {msg}")

    try:
        result_dir = client.process_library(
            papers=papers,
            embedder=embedder_type,
            progress_callback=_progress,
        )
        print(f"  Results installed at: {result_dir}")
    except (ConnectionError, requests.exceptions.ConnectionError):
        print("  Cloud processing failed: lost connection to server.")
        print("  Check your internet connection and try again.")
        sys.exit(1)
    except CloudProcessingError as e:
        print(f"  Cloud processing failed: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------


def _build_index(papers, embedder_type):
    """Build the search index, skipping if already fresh."""
    from incite.webapp.state import get_cache_dir, get_retriever

    cache_dir = get_cache_dir()
    index_path = cache_dir / f"zotero_index_{embedder_type}"

    # Check if index already exists and has the right paper count
    if (index_path / "index.faiss").exists():
        id_map_path = index_path / "id_map.json"
        if id_map_path.exists():
            with open(id_map_path) as f:
                id_map = json.load(f)
            cached_count = len(id_map.get("id_to_idx", {}))
            cached_embedder = id_map.get("embedder_type")
            if cached_embedder and cached_embedder != embedder_type:
                print(
                    f"  Note: existing index uses {cached_embedder}, "
                    f"rebuilding for {embedder_type}..."
                )
            elif cached_count == len(papers):
                print("  Index already up to date, skipping rebuild.")
                return get_retriever(
                    papers,
                    method="hybrid",
                    embedder_type=embedder_type,
                )

    import time

    _start_time = time.monotonic()

    def _progress(msg):
        elapsed = time.monotonic() - _start_time
        if elapsed > 5:
            mins, secs = divmod(int(elapsed), 60)
            print(f"  {msg} [{mins}:{secs:02d} elapsed]")
        else:
            print(f"  {msg}")

    try:
        retriever = get_retriever(
            papers,
            method="hybrid",
            embedder_type=embedder_type,
            progress_callback=_progress,
        )
        print("  Done!")
        return retriever
    except (OSError, RuntimeError) as e:
        err_str = str(e).lower()
        if "model" in err_str or "download" in err_str or "huggingface" in err_str:
            print(
                f"Error downloading embedding model: {e}\n"
                "  Check your internet connection and try again."
            )
        else:
            print(f"Error building index: {_user_friendly_error('build index', e)}")
        return None
    except Exception as e:
        print(f"Error building index: {_user_friendly_error('build index', e)}")
        return None


# ---------------------------------------------------------------------------
# Test query
# ---------------------------------------------------------------------------


def _test_query(retriever, papers):
    """Optional test query (Step 4)."""
    print("Step 4: Test it out!")
    print("  Enter a sentence from your writing (or press Enter to skip):")

    try:
        query = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if not query:
        return

    paper_dict = {p.id: p for p in papers}
    try:
        results = retriever.retrieve(query, k=3, papers=paper_dict, deduplicate=True)
        print()
        print("  Top 3 results:")
        for i, result in enumerate(results, 1):
            paper = paper_dict.get(result.paper_id)
            if paper:
                authors = ""
                if paper.authors:
                    first_author = paper.authors[0].split(",")[0]
                    authors = f"{first_author}, " if paper.year else first_author
                year = paper.year or "n.d."
                print(f"    {i}. [{result.score:.2f}] {paper.title} ({authors}{year})")
        print()
    except Exception as e:
        print(f"  Error running query: {e}")


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------


def _save_setup_config(config):
    """Save setup configuration to ~/.incite/config.json."""
    from incite.webapp.state import get_config, save_config

    # Merge with any existing config rather than overwriting
    existing = get_config()
    existing["source"] = config["source"]
    existing["embedder"] = config["embedder"]
    existing["method"] = config["method"]
    existing["processing"] = config["processing"]
    save_config(existing)


def _user_friendly_error(action: str, exc: Exception) -> str:
    """Map common exceptions to plain-English messages.

    Args:
        action: What the user was trying to do (e.g., "load papers")
        exc: The exception that occurred

    Returns:
        A user-friendly error string
    """
    import sqlite3

    msg = str(exc)
    if isinstance(exc, sqlite3.OperationalError) and "locked" in msg.lower():
        return "Zotero database is locked. Close Zotero and try again."
    if isinstance(exc, PermissionError):
        return "Permission denied. Check file permissions."
    if isinstance(exc, FileNotFoundError):
        return f"File not found: {msg}"
    if isinstance(exc, (OSError, RuntimeError)):
        lower = msg.lower()
        if "model" in lower or "download" in lower or "huggingface" in lower:
            return f"Could not download model. Check your internet connection. ({msg})"
        if "no space" in lower or "disk" in lower:
            return f"Disk space issue: {msg}"
    if isinstance(exc, (ConnectionError,)):
        return "Network error. Check your internet connection."
    if isinstance(exc, ValueError) and "auto-detect" in msg.lower():
        return "Could not auto-detect configuration. Run 'incite setup' first."
    return msg
