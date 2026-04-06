"""Configuration management for the inCite local application.

Handles reading/writing ~/.incite/config.json, cache directory management,
and one-time migration from the old ~/.mayacite/ path.

The config file stores:
    - source_type: "zotero" | "paperpile" | "folder" | "cloud"
    - zotero_data_dir: Path to Zotero data directory
    - embedder: Selected embedding model name
    - method: Retrieval method ("hybrid", "neural", "bm25")
    - Various source-specific paths and settings

Related modules:
    - incite.webapp.state: Retriever and corpus state management.
    - incite.cli.setup: Interactive setup wizard writes this config.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _migrate_cache_dir() -> None:
    """Migrate ~/.mayacite/ to ~/.incite/ if needed (one-time rename)."""
    old_dir = Path.home() / ".mayacite"
    new_dir = Path.home() / ".incite"
    if old_dir.is_dir() and not new_dir.exists():
        logger.info("Migrating cache directory: %s -> %s", old_dir, new_dir)
        old_dir.rename(new_dir)


def get_cache_dir() -> Path:
    """Get the cache directory, creating if needed."""
    _migrate_cache_dir()
    cache_dir = Path.home() / ".incite"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_config_path() -> Path:
    """Get path to config file."""
    return get_cache_dir() / "config.json"


def get_config() -> dict:
    """Load configuration from JSON file.

    Config is stored at ~/.incite/config.json. On first run (no JSON config),
    falls back to:
      1. TOML migration: if ~/.incite/config.toml exists, reads it with tomllib
         (Python 3.11+) or tomli (older), saves as JSON, and returns it.
      2. Auto-detection: tries to find the Zotero data directory automatically
         and returns a default config dict (not yet saved to disk).

    Returns:
        Configuration dict with top-level keys: "zotero", "paperpile", "webapp".
    """
    config_path = get_config_path()

    # Migrate from old TOML config if it exists.
    # tomllib is in stdlib from Python 3.11; fall back to the tomli backport.
    old_toml_path = get_cache_dir() / "config.toml"
    if not config_path.exists() and old_toml_path.exists():
        config = _load_toml_config(old_toml_path)
        if config:
            save_config(config)
            return config

    if not config_path.exists():
        # Auto-detect Zotero directory so first-run works without manual setup.
        from incite.corpus.zotero_reader import find_zotero_data_dir

        detected_dir = find_zotero_data_dir()
        return {
            "zotero": {
                "data_dir": str(detected_dir) if detected_dir else "",
            },
            "paperpile": {
                "bibtex_url": "",
                "bibtex_path": "",
                "pdf_folder": "",
            },
            "webapp": {
                "default_method": "hybrid",
                "default_k": 5,
            },
        }

    with open(config_path) as f:
        return json.loads(f.read())


def _load_toml_config(toml_path: Path) -> dict | None:
    """Attempt to load a TOML config file using tomllib or tomli.

    Returns None if neither library is available.
    """
    try:
        import tomllib

        with open(toml_path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        pass

    try:
        import tomli

        with open(toml_path, "rb") as f:
            return tomli.load(f)
    except ImportError:
        return None


def save_config(config: dict) -> None:
    """Save configuration to JSON file."""
    config_path = get_config_path()
    config_path.write_text(json.dumps(config, indent=2))
