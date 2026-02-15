"""Diagnostic command: check system health and configuration."""

import json
import shutil
import sys
from pathlib import Path


def register(subparsers):
    """Register the doctor command."""
    p = subparsers.add_parser(
        "doctor",
        help="Check system health and diagnose common issues",
    )
    p.set_defaults(func=cmd_doctor)


def cmd_doctor(args):
    """Run diagnostic checks and report results."""
    print()
    print("inCite Doctor")
    print("=" * 40)
    print()

    checks = [
        _check_python_version,
        _check_faiss,
        _check_torch_device,
        _check_zotero_db,
        _check_config_embedder,
        _check_embedding_model,
        _check_faiss_index,
        _check_chunk_cache,
        _check_api_server,
        _check_cloud_api,
        _check_disk_space,
    ]

    passed = 0
    failed = 0
    for check_fn in checks:
        ok, message = check_fn()
        prefix = "[+]" if ok else "[!]"
        print(f"  {prefix} {message}")
        if ok:
            passed += 1
        else:
            failed += 1

    print()
    if failed == 0:
        print(f"All {passed} checks passed.")
    else:
        print(f"{passed} passed, {failed} issue(s) found.")
    print()


# ---------------------------------------------------------------------------
# Individual checks — each returns (ok: bool, message: str)
# ---------------------------------------------------------------------------


def _check_python_version() -> tuple[bool, str]:
    v = sys.version_info
    ok = v >= (3, 10)
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if ok:
        return True, f"Python {version_str}"
    return False, f"Python {version_str} (need >= 3.10)"


def _check_faiss() -> tuple[bool, str]:
    try:
        import faiss  # noqa: F401

        return True, "FAISS importable"
    except ImportError:
        return False, "FAISS not installed. Run: pip install faiss-cpu"


def _check_torch_device() -> tuple[bool, str]:
    try:
        from incite.utils import get_best_device

        device = get_best_device()
        return True, f"PyTorch device: {device}"
    except ImportError:
        return False, "PyTorch not installed"
    except Exception as e:
        return False, f"PyTorch error: {e}"


def _check_zotero_db() -> tuple[bool, str]:
    try:
        from incite.webapp.state import get_config

        config = get_config()
        source = config.get("source", {})

        # Check setup config path first
        if source.get("type") == "zotero" and source.get("path"):
            zotero_dir = Path(source["path"])
        else:
            # Fall back to auto-detection
            from incite.corpus.zotero_reader import find_zotero_data_dir

            zotero_dir = find_zotero_data_dir()

        if zotero_dir is None:
            return False, "Zotero not found. Run 'incite setup' to configure."

        db_path = zotero_dir / "zotero.sqlite"
        if not db_path.exists():
            return False, f"Zotero DB not found at {db_path}"

        # Check if readable (not locked)
        import sqlite3

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)
            conn.execute("SELECT COUNT(*) FROM items")
            conn.close()
            return True, f"Zotero DB readable ({zotero_dir})"
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                return False, "Zotero DB locked — close Zotero and try again"
            return False, f"Zotero DB error: {e}"
    except Exception as e:
        return False, f"Zotero check failed: {e}"


def _check_config_embedder() -> tuple[bool, str]:
    try:
        from incite.webapp.state import get_config

        config = get_config()
        embedder = config.get("embedder")
        if embedder:
            return True, f"Config embedder: {embedder}"
        return False, "No embedder in config. Run 'incite setup' to configure."
    except Exception as e:
        return False, f"Config check failed: {e}"


def _check_embedding_model() -> tuple[bool, str]:
    try:
        from incite.webapp.state import get_config

        config = get_config()
        embedder = config.get("embedder", "minilm")

        # Check if model files are cached locally
        import importlib.util

        if importlib.util.find_spec("sentence_transformers") is None:
            return False, "sentence-transformers not installed"

        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        # For fine-tuned models, check the models directory
        if embedder in ("minilm-ft", "granite-ft"):
            from incite.retrieval.factory import EMBEDDERS

            model_path = EMBEDDERS.get(embedder, {}).get("model")
            if model_path and Path(model_path).exists():
                return True, f"Embedding model cached: {embedder}"
            return False, f"Fine-tuned model '{embedder}' not found locally"

        # For HuggingFace models, just try to instantiate
        # This is a lightweight check — it checks cache, not downloads
        if cache_dir.exists() and any(cache_dir.iterdir()):
            return True, f"Embedding model: {embedder} (HF cache exists)"
        return True, f"Embedding model: {embedder} (will download on first use)"
    except ImportError:
        return False, "sentence-transformers not installed"
    except Exception as e:
        return False, f"Model check failed: {e}"


def _check_faiss_index() -> tuple[bool, str]:
    try:
        from incite.webapp.state import get_cache_dir, get_config

        config = get_config()
        embedder = config.get("embedder", "minilm")
        cache_dir = get_cache_dir()

        index_path = cache_dir / f"zotero_index_{embedder}"
        faiss_file = index_path / "index.faiss"
        id_map_file = index_path / "id_map.json"

        if not faiss_file.exists():
            return False, f"No FAISS index for {embedder}. Run 'incite setup'."

        if id_map_file.exists():
            with open(id_map_file) as f:
                id_map = json.load(f)
            n_papers = len(id_map.get("id_to_idx", {}))
            cached_embedder = id_map.get("embedder_type", "unknown")
            if cached_embedder != embedder:
                return (
                    False,
                    f"Index embedder mismatch: index={cached_embedder}, "
                    f"config={embedder}. Will rebuild on serve.",
                )
            return True, f"FAISS index: {n_papers} papers ({embedder})"

        return True, f"FAISS index exists for {embedder}"
    except Exception as e:
        return False, f"Index check failed: {e}"


def _check_chunk_cache() -> tuple[bool, str]:
    try:
        from incite.webapp.state import get_cache_dir

        cache_dir = get_cache_dir()

        # Check for any chunk cache file
        chunk_files = list(cache_dir.glob("zotero_chunks_*.jsonl"))
        if not chunk_files:
            # Not an error — chunk cache is optional (paper-mode works without it)
            return True, "No chunk cache (paper-mode only — OK)"

        # Count lines in the largest chunk file
        largest = max(chunk_files, key=lambda p: p.stat().st_size)
        n_chunks = sum(1 for _ in open(largest))
        strategy = largest.stem.replace("zotero_chunks_", "")
        return True, f"Chunk cache: {n_chunks} chunks ({strategy})"
    except Exception as e:
        return False, f"Chunk cache check failed: {e}"


def _check_api_server() -> tuple[bool, str]:
    try:
        import requests

        from incite.webapp.state import get_config

        config = get_config()
        port = config.get("port", 8230)

        resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=3)
        data = resp.json()
        if data.get("ready"):
            n = data.get("corpus_size", "?")
            return True, f"API server running on port {port} ({n} papers)"
        return True, f"API server on port {port} (loading...)"
    except Exception:
        return True, "API server not running (start with 'incite serve')"


def _check_cloud_api() -> tuple[bool, str]:
    try:
        from incite.webapp.state import get_config

        config = get_config()
        cloud = config.get("cloud", {})
        api_url = cloud.get("api_url")

        if not api_url:
            return True, "Cloud processing: not configured (OK)"

        import requests

        resp = requests.get(f"{api_url}/health", timeout=5)
        data = resp.json()
        if data.get("status") == "healthy":
            return True, f"Cloud API healthy ({api_url})"
        return False, f"Cloud API unhealthy: {data}"
    except Exception:
        return False, f"Cloud API unreachable at {api_url}"


def _check_disk_space() -> tuple[bool, str]:
    try:
        from incite.webapp.state import get_cache_dir

        cache_dir = get_cache_dir()
        usage = shutil.disk_usage(cache_dir)
        free_gb = usage.free / (1024**3)
        if free_gb >= 1.0:
            return True, f"Disk space: {free_gb:.1f} GB free"
        return False, f"Low disk space: {free_gb:.1f} GB free (need >= 1 GB)"
    except Exception as e:
        return False, f"Disk space check failed: {e}"
