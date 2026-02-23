"""Server commands: serve, webapp."""

import os
import sys
from pathlib import Path

from incite.cli._shared import EMBEDDER_CHOICES


def _config_default(key: str, fallback):
    """Read a default value from saved config, falling back if missing."""
    try:
        from incite.webapp.state import get_config

        config = get_config()
        value = config.get(key, fallback)
        # Validate embedder choice
        if key == "embedder" and value not in EMBEDDER_CHOICES:
            return fallback
        return value
    except Exception:
        return fallback


def register(subparsers):
    """Register server commands."""
    _register_serve(subparsers)
    _register_webapp(subparsers)
    _register_tray(subparsers)


def _register_serve(subparsers):
    saved_embedder = _config_default("embedder", "minilm")
    saved_method = _config_default("method", "hybrid")
    saved_port = _config_default("port", 8230)

    p = subparsers.add_parser("serve", help="Start the REST API server for editor integration")
    p.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)"
    )
    p.add_argument(
        "--port", type=int, default=saved_port, help=f"Port to listen on (default: {saved_port})"
    )
    p.add_argument(
        "--method",
        type=str,
        choices=["neural", "bm25", "hybrid"],
        default=saved_method,
        help=f"Retrieval method (default: {saved_method})",
    )
    p.add_argument(
        "--embedder",
        type=str,
        choices=EMBEDDER_CHOICES,
        default=saved_embedder,
        help=f"Embedder model (default: {saved_embedder})",
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["paper", "paragraph", "sentence", "grobid"],
        default="paper",
        help="Retrieval mode: paper (rankings only), paragraph/sentence/grobid (with evidence)",
    )
    p.add_argument(
        "--source",
        type=str,
        choices=["zotero", "paperpile", "folder", "file"],
        default="zotero",
        help="Corpus source (default: zotero)",
    )
    p.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    p.add_argument(
        "--ssl-cert",
        type=str,
        default=None,
        help="Path to SSL certificate (for HTTPS, required by Word add-ins in production)",
    )
    p.add_argument("--ssl-key", type=str, default=None, help="Path to SSL private key")
    p.set_defaults(func=cmd_serve)


def _register_webapp(subparsers):
    p = subparsers.add_parser("webapp", help="Launch the testing webapp")
    p.add_argument(
        "--port", type=int, default=8501, help="Port to run the webapp on (default: 8501)"
    )
    p.set_defaults(func=cmd_webapp)


def _register_tray(subparsers):
    saved_embedder = _config_default("embedder", "minilm")
    saved_method = _config_default("method", "hybrid")
    saved_port = _config_default("port", 8230)

    p = subparsers.add_parser("tray", help="Start the menu bar app (macOS)")
    p.add_argument(
        "--no-autostart",
        action="store_true",
        help="Don't auto-start the API server",
    )
    p.add_argument(
        "--embedder",
        type=str,
        choices=EMBEDDER_CHOICES,
        default=saved_embedder,
        help=f"Embedder model (default: {saved_embedder})",
    )
    p.add_argument(
        "--method",
        type=str,
        choices=["neural", "bm25", "hybrid"],
        default=saved_method,
        help=f"Retrieval method (default: {saved_method})",
    )
    p.add_argument(
        "--port",
        type=int,
        default=saved_port,
        help=f"API server port (default: {saved_port})",
    )
    p.set_defaults(func=cmd_tray)


# --- Command handlers ---


def cmd_serve(args):
    """Start the REST API server."""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Install with: pip install incite[api]")
        sys.exit(1)

    mode_map = {
        "paper": ("paper", "paragraph"),
        "paragraph": ("paragraph", "paragraph"),
        "sentence": ("paragraph", "sentence"),
        "grobid": ("paragraph", "grobid"),
    }
    internal_mode, chunking = mode_map[args.mode]

    os.environ["INCITE_METHOD"] = args.method
    os.environ["INCITE_EMBEDDER"] = args.embedder
    os.environ["INCITE_MODE"] = internal_mode
    os.environ["INCITE_CHUNKING"] = chunking
    os.environ["INCITE_SOURCE"] = args.source

    # Save effective settings so next run remembers them
    try:
        from incite.webapp.state import get_config, save_config

        config = get_config()
        config["embedder"] = args.embedder
        config["method"] = args.method
        config["port"] = args.port
        save_config(config)
    except Exception:
        pass  # Don't fail startup over config save

    ssl_kwargs = {}
    if args.ssl_cert:
        ssl_kwargs["ssl_certfile"] = args.ssl_cert
    if args.ssl_key:
        ssl_kwargs["ssl_keyfile"] = args.ssl_key

    scheme = "https" if args.ssl_cert else "http"
    print(f"Starting inCite API server on {args.host}:{args.port}")
    print(f"  Method: {args.method}, Embedder: {args.embedder}, Mode: {args.mode}")
    print(f"  Docs: {scheme}://{args.host}:{args.port}/docs")
    if args.ssl_cert:
        print(f"  Word add-in: {scheme}://{args.host}:{args.port}/word-addin/taskpane.html")
    print("Press Ctrl+C to stop\n")

    uvicorn.run(
        "incite.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
        **ssl_kwargs,
    )


def cmd_webapp(args):
    """Launch the testing webapp."""
    import subprocess

    app_path = Path(__file__).parent.parent / "webapp" / "app.py"

    if not app_path.exists():
        print(f"Error: Webapp not found at {app_path}")
        sys.exit(1)

    print(f"Launching inCite webapp on port {args.port}...")
    print("Press Ctrl+C to stop\n")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.port",
                str(args.port),
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("\nWebapp stopped.")
    except subprocess.CalledProcessError as e:
        if "streamlit" in str(e):
            print("\nError: Streamlit not installed. Install with: pip install incite[webapp]")
            sys.exit(1)
        raise


def cmd_tray(args):
    """Start the macOS menu bar app."""
    try:
        from incite.tray import InCiteTray
    except ImportError:
        print("Error: rumps not installed. Install with: pip install incite[tray]")
        sys.exit(1)

    app = InCiteTray(
        auto_start=not args.no_autostart,
        embedder=args.embedder,
        method=args.method,
        port=args.port,
    )
    app.run()
