"""Paperpile integration commands: setup, sync, status."""

import sys
from pathlib import Path

from incite.cli._shared import EMBEDDER_CHOICES


def register(subparsers):
    """Register the paperpile command group."""
    p = subparsers.add_parser(
        "paperpile",
        help="Manage Paperpile library integration",
    )
    sub = p.add_subparsers(dest="paperpile_command", help="Paperpile subcommands")

    # paperpile setup
    setup_p = sub.add_parser("setup", help="Configure Paperpile BibTeX URL and PDF folder")
    setup_p.add_argument(
        "--bibtex-url",
        type=str,
        default=None,
        help="Paperpile BibTeX auto-sync URL (non-interactive)",
    )
    setup_p.add_argument(
        "--bibtex-path",
        type=str,
        default=None,
        help="Path to a local .bib file (alternative to URL)",
    )
    setup_p.add_argument(
        "--pdf-folder",
        type=str,
        default=None,
        help="Path to Paperpile's Google Drive PDF folder",
    )
    setup_p.set_defaults(func=cmd_paperpile_setup)

    # paperpile sync
    sync_p = sub.add_parser("sync", help="Fetch latest BibTeX and update corpus")
    sync_p.add_argument(
        "--force",
        action="store_true",
        help="Force full re-sync even if BibTeX unchanged",
    )
    sync_p.set_defaults(func=cmd_paperpile_sync)

    # paperpile status
    status_p = sub.add_parser("status", help="Show Paperpile configuration and sync status")
    status_p.set_defaults(func=cmd_paperpile_status)

    p.set_defaults(func=lambda args: p.print_help())


def cmd_paperpile_setup(args):
    """Configure Paperpile integration."""
    from incite.webapp.state import get_config, save_config

    print()
    print("Paperpile Setup")
    print("=" * 40)

    bibtex_url = args.bibtex_url
    bibtex_path = args.bibtex_path
    pdf_folder = args.pdf_folder

    # Interactive mode if no flags provided
    if not bibtex_url and not bibtex_path:
        print()
        print("Paperpile can sync your library as a BibTeX file.")
        print("In Paperpile: Settings > Workflows > BibTeX export > Copy URL")
        print()
        try:
            choice = input("Do you have a BibTeX sync URL? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nSetup cancelled.")
            sys.exit(1)

        if choice in ("", "y", "yes"):
            try:
                bibtex_url = input("BibTeX URL: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSetup cancelled.")
                sys.exit(1)
            if not bibtex_url:
                print("Error: URL is required.")
                sys.exit(1)
        else:
            try:
                bibtex_path = input("Path to local .bib file: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSetup cancelled.")
                sys.exit(1)
            if not bibtex_path:
                print("Error: Path is required.")
                sys.exit(1)

    # Validate BibTeX path if provided
    if bibtex_path:
        bib_file = Path(bibtex_path).expanduser().resolve()
        if not bib_file.exists():
            print(f"Error: File not found: {bib_file}")
            sys.exit(1)
        bibtex_path = str(bib_file)

    # Validate URL if provided
    if bibtex_url:
        print("  Testing BibTeX URL...")
        try:
            import requests

            resp = requests.head(bibtex_url, timeout=10)
            resp.raise_for_status()
            print(f"  URL is reachable (HTTP {resp.status_code})")
        except Exception as e:
            print(f"  Warning: Could not reach URL: {e}")
            print("  (Will save anyway — you can test later with `incite paperpile sync`)")

    # PDF folder (interactive if not provided)
    if not pdf_folder:
        print()
        print("Optional: Point to your Paperpile PDF folder for paragraph-level evidence.")
        print("(Typically ~/Google Drive/My Drive/Paperpile or similar)")
        try:
            pdf_folder = input("PDF folder path (press Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            pdf_folder = ""

    if pdf_folder:
        folder = Path(pdf_folder).expanduser().resolve()
        if folder.exists():
            pdfs = list(folder.rglob("*.pdf"))
            print(f"  Found {len(pdfs)} PDFs in {folder}")
            pdf_folder = str(folder)
        else:
            print(f"  Warning: Folder not found: {folder}")
            pdf_folder = str(folder)

    # Save config
    config = get_config()
    config["paperpile"] = {
        "bibtex_url": bibtex_url or "",
        "bibtex_path": bibtex_path or "",
        "pdf_folder": pdf_folder or "",
    }
    # Set source type for other commands to discover
    config["source"] = {"type": "paperpile"}
    save_config(config)

    print()
    print("Configuration saved to ~/.incite/config.json")
    print()
    print("Next steps:")
    print("  incite paperpile sync    -- fetch BibTeX and build corpus")
    print("  incite paperpile status  -- check configuration")
    print()


def cmd_paperpile_sync(args):
    """Fetch latest BibTeX and update corpus."""
    from incite.webapp.state import get_config

    config = get_config()
    pp_config = config.get("paperpile", {})

    bibtex_url = pp_config.get("bibtex_url", "")
    bibtex_path = pp_config.get("bibtex_path", "")
    pdf_folder = pp_config.get("pdf_folder", "")

    if not bibtex_url and not bibtex_path:
        print("Error: Paperpile not configured. Run `incite paperpile setup` first.")
        sys.exit(1)

    print("Syncing Paperpile library...")

    from incite.corpus.paperpile_source import PaperpileSource

    source = PaperpileSource(
        bibtex_url=bibtex_url or None,
        bibtex_path=Path(bibtex_path) if bibtex_path else None,
        pdf_folder=Path(pdf_folder) if pdf_folder else None,
    )

    try:
        papers = source.load_papers()
    except Exception as e:
        print(f"Error syncing: {e}")
        sys.exit(1)

    n_abstracts = sum(1 for p in papers if p.abstract)
    n_dois = sum(1 for p in papers if p.doi)
    n_pdfs = sum(1 for p in papers if p.source_file)

    print()
    print(f"Sync complete: {len(papers)} papers")
    print(f"  With abstracts: {n_abstracts}")
    print(f"  With DOIs: {n_dois}")
    print(f"  Matched to PDFs: {n_pdfs}")
    print()

    # Save last sync time
    import datetime

    from incite.webapp.state import get_config, save_config

    config = get_config()
    config.setdefault("paperpile", {})["last_sync"] = datetime.datetime.now().isoformat()
    save_config(config)


def cmd_paperpile_status(args):
    """Show Paperpile configuration and sync status."""
    from incite.webapp.state import get_cache_dir, get_config

    config = get_config()
    pp_config = config.get("paperpile", {})

    print()
    print("Paperpile Configuration")
    print("=" * 40)

    bibtex_url = pp_config.get("bibtex_url", "")
    bibtex_path = pp_config.get("bibtex_path", "")
    pdf_folder = pp_config.get("pdf_folder", "")
    last_sync = pp_config.get("last_sync", "never")

    if bibtex_url:
        # Truncate URL for display
        display_url = bibtex_url[:60] + "..." if len(bibtex_url) > 60 else bibtex_url
        print(f"  BibTeX URL: {display_url}")
    elif bibtex_path:
        print(f"  BibTeX file: {bibtex_path}")
    else:
        print("  Not configured. Run `incite paperpile setup`.")
        return

    if pdf_folder:
        folder = Path(pdf_folder)
        if folder.exists():
            pdfs = list(folder.rglob("*.pdf"))
            print(f"  PDF folder: {pdf_folder} ({len(pdfs)} PDFs)")
        else:
            print(f"  PDF folder: {pdf_folder} (not found)")
    else:
        print("  PDF folder: not configured")

    print(f"  Last sync: {last_sync}")

    # Check cached corpus
    cache_dir = get_cache_dir()
    corpus_path = cache_dir / "paperpile_corpus.jsonl"
    if corpus_path.exists():
        try:
            from incite.corpus.loader import load_corpus

            papers = load_corpus(corpus_path)
            n_abstracts = sum(1 for p in papers if p.abstract)
            n_pdfs = sum(1 for p in papers if p.source_file)
            print(
                f"  Cached corpus: {len(papers)} papers"
                f" ({n_abstracts} with abstracts, {n_pdfs} with PDFs)"
            )
        except Exception:
            print("  Cached corpus: exists but could not read")
    else:
        print("  Cached corpus: not yet synced")

    # Check for indexes
    for embedder in EMBEDDER_CHOICES:
        index_dir = cache_dir / f"paperpile_index_{embedder}"
        if (index_dir / "index.faiss").exists():
            print(f"  Index ({embedder}): built")

    print()
