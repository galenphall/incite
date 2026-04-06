"""Source configuration steps for the inCite setup wizard.

Each function handles interactive configuration of a specific corpus source:
Zotero, local PDF folder, Paperpile BibTeX, or cloud processing service.
Called by the main setup wizard in cli/setup.py.

Related modules:
    - incite.cli.setup: Main setup wizard flow.
    - incite.webapp.state: Config persistence (get_config, save_config).
"""

from pathlib import Path


def _setup_interactive():
    """Interactively prompt the user to select a corpus source (Step 1).

    Detects whether Zotero is installed and presents appropriate options for
    Zotero, local PDF folder, Paperpile BibTeX sync, or cloud processing.

    Returns:
        dict with source configuration keys, or None if the user cancelled.
    """
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
    """Non-interactively configure Zotero as the corpus source.

    Auto-detects the Zotero data directory. Prints an error and returns None
    if Zotero cannot be found.

    Returns:
        dict with ``{"type": "zotero", "path": ...}``, or None on failure.
    """
    from incite.corpus.zotero_reader import find_zotero_data_dir

    zotero_dir = find_zotero_data_dir()
    if zotero_dir is None:
        print("Error: Could not auto-detect Zotero directory.")
        print("Make sure Zotero is installed, or use --folder instead.")
        return None

    print(f"Step 1: Using Zotero library at {zotero_dir}")
    return {"type": "zotero", "path": str(zotero_dir)}


def _setup_folder(folder_path: str):
    """Configure a local PDF folder as the corpus source.

    Validates that the path exists, is a directory, and contains at least
    one PDF file.

    Args:
        folder_path: Path string to the PDF folder (may include ``~``).

    Returns:
        dict with ``{"type": "folder", "path": ...}``, or None on failure.
    """
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
    """Configure Paperpile as the corpus source via BibTeX sync URL.

    Prompts for the Paperpile BibTeX export URL and an optional PDF folder
    path, then persists the Paperpile-specific settings to the incite config.

    Returns:
        dict with Paperpile source configuration keys, or None if cancelled.
    """
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
    """Configure cloud processing, prompting for API URL and key.

    If a Zotero directory was already detected, it is used as the source.
    Otherwise the user is prompted for a local PDF folder path.  The returned
    dict includes ``"processing": "cloud"`` so the main wizard can route to
    the cloud processing path.

    Args:
        zotero_dir: Path to the auto-detected Zotero data directory, or None.

    Returns:
        dict with source and cloud credential keys, or None if cancelled.
    """
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
