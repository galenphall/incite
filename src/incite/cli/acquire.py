"""PDF acquisition commands: acquire, proxy {setup, test, login, status}."""

import os
import sys
from pathlib import Path


def register(subparsers):
    """Register acquire and proxy commands."""
    _register_acquire(subparsers)
    _register_proxy(subparsers)


def _register_acquire(subparsers):
    p = subparsers.add_parser(
        "acquire", help="Acquire PDFs for papers (free sources + library proxy)"
    )
    p.add_argument("--doi", type=str, help="Acquire PDF for a single DOI")
    p.add_argument("--dois", type=str, help="Path to file with one DOI per line")
    p.add_argument(
        "--backfill", action="store_true", help="Scan Zotero library for items missing PDFs"
    )
    p.add_argument(
        "--from-last-recommendations",
        action="store_true",
        help="Acquire PDFs for papers from the last recommendation query",
    )
    p.add_argument("--corpus", type=str, help="Acquire PDFs for papers in a corpus JSONL file")
    p.add_argument("--dry-run", action="store_true", help="Show plan without downloading")
    p.add_argument(
        "--free-only",
        action="store_true",
        help="Only use free sources (Unpaywall/arXiv), skip proxy",
    )
    p.add_argument("--limit", type=int, default=None, help="Stop after acquiring N papers")
    p.add_argument(
        "--dest",
        type=str,
        default=None,
        help="Output directory for PDFs (default: ~/.incite/acquired/)",
    )
    p.set_defaults(func=cmd_acquire)


def _register_proxy(subparsers):
    proxy_parser = subparsers.add_parser("proxy", help="Configure and manage library proxy access")
    proxy_subparsers = proxy_parser.add_subparsers(dest="proxy_command", help="Proxy subcommands")

    proxy_subparsers.add_parser("setup", help="Interactive proxy configuration wizard")
    proxy_subparsers.add_parser("test", help="Test proxy connectivity and authentication")
    proxy_subparsers.add_parser("login", help="Force re-authentication with proxy")
    proxy_subparsers.add_parser("status", help="Show proxy configuration and session status")

    proxy_parser.set_defaults(func=cmd_proxy)


# --- Command handlers ---


def cmd_acquire(args):
    """Acquire PDFs for papers from various sources."""
    import json

    from incite.acquire.config import load_proxy_config
    from incite.acquire.pipeline import AcquisitionPipeline
    from incite.webapp.state import get_cache_dir

    email = os.getenv("OPENALEX_EMAIL")
    if not email:
        print("Error: OPENALEX_EMAIL not set (required by Unpaywall TOS).")
        print("Add OPENALEX_EMAIL=you@example.com to your .env file.")
        sys.exit(1)

    dest_dir = Path(args.dest) if args.dest else get_cache_dir() / "acquired"

    proxy_config = None
    if not args.free_only:
        proxy_config = load_proxy_config()
        if not proxy_config.is_configured:
            proxy_config = None
            if not args.free_only:
                print("No proxy configured. Using free sources only.")
                print("Run 'incite proxy setup' to configure library proxy.\n")

    papers = []

    if args.doi:
        from incite.models import Paper

        papers = [Paper(id=args.doi, title=args.doi, doi=args.doi)]
    elif args.dois:
        from incite.models import Paper

        dois_path = Path(args.dois)
        if not dois_path.exists():
            print(f"Error: File not found: {args.dois}")
            sys.exit(1)
        with open(dois_path) as f:
            dois = [line.strip() for line in f if line.strip()]
        papers = [Paper(id=doi, title=doi, doi=doi) for doi in dois]
        print(f"Loaded {len(papers)} DOIs from {args.dois}")
    elif args.backfill:
        from incite.corpus.zotero_reader import find_zotero_data_dir, read_zotero_library

        zotero_dir = find_zotero_data_dir()
        if not zotero_dir:
            print("Error: Could not find Zotero data directory.")
            sys.exit(1)
        print(f"Scanning Zotero library at {zotero_dir}...")
        all_papers = read_zotero_library(zotero_dir, show_progress=True)
        papers = [
            p for p in all_papers if p.doi and not (p.source_file and Path(p.source_file).exists())
        ]
        print(f"Found {len(all_papers)} items, {len(papers)} missing PDFs (with DOI).")
    elif args.from_last_recommendations:
        last_recs_path = get_cache_dir() / "last_recommendations.json"
        if not last_recs_path.exists():
            print("Error: No last recommendations found.")
            print("Run 'incite agent recommend' or use the API first.")
            sys.exit(1)
        with open(last_recs_path) as f:
            data = json.load(f)
        from incite.models import Paper

        papers = []
        for entry in data.get("recommendations", []):
            doi = entry.get("doi")
            if doi:
                papers.append(
                    Paper(
                        id=entry.get("paper_id", doi),
                        title=entry.get("title", doi),
                        doi=doi,
                    )
                )
        print(f"Loaded {len(papers)} papers from last recommendations")
        if not papers:
            print("No papers with DOIs in last recommendations.")
            sys.exit(0)
    elif args.corpus:
        from incite.corpus.loader import load_corpus

        all_papers = load_corpus(args.corpus)
        papers = [p for p in all_papers if p.doi]
        print(f"Loaded {len(all_papers)} papers, {len(papers)} with DOIs")
    else:
        print(
            "Error: Specify one of --doi, --dois, --backfill,"
            " --from-last-recommendations, or --corpus"
        )
        sys.exit(1)

    if not papers:
        print("No papers to acquire.")
        sys.exit(0)

    pipeline = AcquisitionPipeline(
        dest_dir=dest_dir,
        email=email,
        proxy_config=proxy_config,
        free_only=args.free_only,
        dry_run=args.dry_run,
    )

    try:
        summary = pipeline.acquire_batch(papers, limit=args.limit)
    finally:
        pipeline.close()

    print("\nSummary:")
    print(f"  Acquired:  {summary.acquired}")
    print(f"  Skipped:   {summary.skipped}")
    print(f"  Failed:    {summary.failed}")
    if summary.by_source:
        sources = ", ".join(f"{count} {src}" for src, count in summary.by_source.items())
        print(f"  Sources:   {sources}")
    print(f"\nPDFs saved to: {dest_dir}")


def cmd_proxy(args):
    """Dispatch proxy subcommands."""
    if args.proxy_command == "setup":
        cmd_proxy_setup(args)
    elif args.proxy_command == "test":
        cmd_proxy_test(args)
    elif args.proxy_command == "login":
        cmd_proxy_login(args)
    elif args.proxy_command == "status":
        cmd_proxy_status(args)
    else:
        print("Usage: incite proxy {setup|test|login|status}")


def cmd_proxy_setup(args):
    """Interactive proxy configuration wizard."""
    from incite.acquire.config import INSTITUTION_PRESETS, ProxyConfig, save_proxy_config

    print("\nLibrary Proxy Configuration")
    print("===========================\n")
    print("How does your institution provide access to paywalled papers?\n")
    print("  1. EZproxy (URL prefix) -- e.g., proxy.lib.umich.edu/login?url=...")
    print("  2. EZproxy (URL suffix) -- e.g., www-nature-com.proxy.lib.umich.edu")
    print("  3. VPN -- I connect to my university VPN and everything works")
    print()

    presets = list(INSTITUTION_PRESETS.keys())
    if presets:
        print(f"  Or use a preset: {', '.join(presets)}")
        print()

    choice = input("Enter choice (1/2/3 or preset name): ").strip().lower()

    if choice in INSTITUTION_PRESETS:
        config = INSTITUTION_PRESETS[choice]
        print(f"\nUsing preset: {config.institution_name}")
    elif choice == "1":
        url = input(
            "\nEnter your proxy URL prefix (e.g., https://proxy.lib.umich.edu/login?url=): "
        ).strip()
        name = input("Institution name (for your reference): ").strip()
        config = ProxyConfig(
            proxy_type="ezproxy_prefix",
            proxy_url=url,
            institution_name=name,
        )
    elif choice == "2":
        suffix = input("\nEnter your proxy suffix (e.g., .proxy.lib.umich.edu): ").strip()
        name = input("Institution name (for your reference): ").strip()
        config = ProxyConfig(
            proxy_type="ezproxy_suffix",
            proxy_suffix=suffix,
            institution_name=name,
        )
    elif choice == "3":
        name = input("\nInstitution name (for your reference): ").strip()
        config = ProxyConfig(
            proxy_type="vpn",
            institution_name=name,
        )
    else:
        print("Invalid choice.")
        sys.exit(1)

    save_proxy_config(config)
    print("\nProxy configuration saved.")
    print(f"  Type: {config.proxy_type}")
    print(f"  Institution: {config.institution_name}")
    if config.proxy_url:
        print(f"  URL: {config.proxy_url}")
    if config.proxy_suffix:
        print(f"  Suffix: {config.proxy_suffix}")
    print("\nRun 'incite proxy test' to verify connectivity.")


def cmd_proxy_test(args):
    """Test proxy connectivity and authentication."""
    from incite.acquire.config import load_proxy_config
    from incite.acquire.proxy import create_proxy

    config = load_proxy_config()
    if not config.is_configured:
        print("No proxy configured. Run 'incite proxy setup' first.")
        sys.exit(1)

    print(f"Testing proxy: {config.institution_name} ({config.proxy_type})")
    proxy = create_proxy(config)

    try:
        result = proxy.ensure_authenticated(interactive=False)
        if result:
            print("Proxy session is valid.")
        else:
            print("Proxy session has expired or is not set up.")
            print("Run 'incite proxy login' to authenticate.")
    finally:
        proxy.close()


def cmd_proxy_login(args):
    """Force re-authentication with proxy."""
    from incite.acquire.config import load_proxy_config
    from incite.acquire.proxy import create_proxy

    config = load_proxy_config()
    if not config.is_configured:
        print("No proxy configured. Run 'incite proxy setup' first.")
        sys.exit(1)

    print(f"Authenticating with: {config.institution_name}")
    proxy = create_proxy(config)

    try:
        result = proxy.ensure_authenticated(interactive=True)
        if result:
            print("Authentication successful!")
        else:
            print("Authentication failed.")
            sys.exit(1)
    finally:
        proxy.close()


def cmd_proxy_status(args):
    """Show proxy configuration and session status."""
    from incite.acquire.config import load_proxy_config

    config = load_proxy_config()
    if not config.is_configured:
        print("No proxy configured.")
        print("Run 'incite proxy setup' to configure.")
        return

    print("\nProxy Configuration:")
    print(f"  Type:        {config.proxy_type}")
    print(f"  Institution: {config.institution_name}")
    if config.proxy_url:
        print(f"  URL:         {config.proxy_url}")
    if config.proxy_suffix:
        print(f"  Suffix:      {config.proxy_suffix}")
    print(f"  Test DOI:    {config.test_doi}")
    print(f"  Session dir: {config.session_dir}")

    storage_state = config.session_dir / "storage_state.json"
    if storage_state.exists():
        from datetime import datetime

        mtime = os.path.getmtime(storage_state)
        saved_at = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n  Session saved: {saved_at}")
        print("  Run 'incite proxy test' to verify it's still valid.")
    else:
        print("\n  No saved session. Run 'incite proxy login' to authenticate.")
