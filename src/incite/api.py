"""FastAPI server for editor integration (VS Code, Obsidian, etc.).

Thin HTTP wrapper around InCiteAgent. Start with:
    incite serve --port 8230
"""

import logging
import os
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Module-level agent, loaded at startup via lifespan
_agent = None


class RecommendRequest(BaseModel):
    query: str
    k: int = Field(default=10, ge=1, le=100)
    author_boost: float = Field(default=1.0, ge=0.0, le=5.0)
    cursor_sentence_index: Optional[int] = Field(
        default=None,
        description="Index of the focal sentence in the query (for position-weighted embedding)",
    )
    focus_decay: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Decay rate per sentence of distance from cursor (0=only cursor, 1=uniform)",
    )
    alpha: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Paper vs chunk score blend weight for two-stage retrieval. "
            "1.0 = paper only, 0.0 = chunk only. "
            "Only effective when two-stage retrieval is active."
        ),
    )


class BatchRequest(BaseModel):
    queries: list[str]
    k: int = Field(default=10, ge=1, le=100)
    author_boost: float = Field(default=1.0, ge=0.0, le=5.0)


def get_agent():
    """Get the loaded agent or raise 503 if not ready."""
    if _agent is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not loaded. Server is still starting up.",
        )
    return _agent


def _create_source(source_type: str):
    """Create a CorpusSource based on the INCITE_SOURCE env var.

    Args:
        source_type: Source type ("zotero", "file", or a JSONL path)

    Returns:
        A CorpusSource instance
    """
    if source_type == "zotero":
        from incite.corpus.zotero_reader import ZoteroSource

        return ZoteroSource()
    elif source_type == "paperpile":
        from incite.corpus.paperpile_source import PaperpileSource
        from incite.webapp.state import get_config

        config = get_config()
        pp = config.get("paperpile", {})
        return PaperpileSource(
            bibtex_url=pp.get("bibtex_url") or None,
            bibtex_path=Path(pp["bibtex_path"]) if pp.get("bibtex_path") else None,
            pdf_folder=Path(pp["pdf_folder"]) if pp.get("pdf_folder") else None,
        )
    elif source_type == "folder":
        from incite.corpus.folder_source import FolderCorpusSource

        folder_path = os.environ.get("INCITE_FOLDER_PATH")
        if not folder_path:
            raise ValueError("INCITE_SOURCE=folder requires INCITE_FOLDER_PATH to be set")
        return FolderCorpusSource(folder_path)
    elif source_type == "file" or source_type.endswith(".jsonl"):
        from incite.corpus.loader import CorpusFileSource

        path = os.environ.get("INCITE_CORPUS_PATH", source_type)
        if path == "file":
            raise ValueError("INCITE_SOURCE=file requires INCITE_CORPUS_PATH to be set")
        return CorpusFileSource(path)
    else:
        raise ValueError(
            f"Unknown source type: {source_type}. "
            f"Use 'zotero', 'folder', 'file', or a path to a .jsonl file."
        )


def _diagnose_startup_error(exc: Exception) -> str:
    """Map startup exceptions to actionable messages for the console."""
    import sqlite3

    msg = str(exc)
    if isinstance(exc, sqlite3.OperationalError) and "locked" in msg.lower():
        return "Zotero database is locked — close Zotero and restart the server."
    if isinstance(exc, ValueError) and "auto-detect" in msg.lower():
        return "Could not auto-detect Zotero library. Run 'incite setup' first."
    if isinstance(exc, FileNotFoundError):
        return f"File not found: {msg}. Run 'incite setup' to configure your library."
    if isinstance(exc, (OSError, RuntimeError)):
        lower = msg.lower()
        if "model" in lower or "download" in lower or "huggingface" in lower:
            return f"Model download failed. Check your internet connection. ({msg})"
    if isinstance(exc, PermissionError):
        return "Permission denied reading library files. Check file permissions."
    return msg


def _warn_embedder_mismatch(embedder: str) -> None:
    """Check if cached FAISS index was built with a different embedder.

    Logs a warning if so — the rebuild will take 15-20 minutes and users
    should know it's coming rather than thinking the app is frozen.
    """
    try:
        from incite.webapp.state import get_cache_dir

        cache_dir = get_cache_dir()
        index_path = cache_dir / f"zotero_index_{embedder}"
        id_map_path = index_path / "id_map.json"

        if not id_map_path.exists():
            return

        import json

        with open(id_map_path) as f:
            id_map = json.load(f)
        cached_embedder = id_map.get("embedder_type")

        if cached_embedder and cached_embedder != embedder:
            logger.warning(
                "Embedder mismatch: cached index was built with '%s' but starting "
                "with '%s'. The index will be rebuilt (~15 min for large libraries). "
                "To avoid this, run: incite serve --embedder %s",
                cached_embedder,
                embedder,
                cached_embedder,
            )
    except Exception:
        pass  # Don't block startup over a warning check


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load InCiteAgent at startup."""
    global _agent

    from incite.agent import InCiteAgent

    method = os.environ.get("INCITE_METHOD", "hybrid")
    embedder = os.environ.get("INCITE_EMBEDDER", "minilm")
    mode = os.environ.get("INCITE_MODE", "paper")
    chunking = os.environ.get("INCITE_CHUNKING", "paragraph")
    source_type = os.environ.get("INCITE_SOURCE", "zotero")

    logger.info(
        "Loading InCiteAgent (source=%s, method=%s, embedder=%s, mode=%s, chunking=%s)...",
        source_type,
        method,
        embedder,
        mode,
        chunking,
    )

    # Check for embedder mismatch before loading (avoids surprise 18-min rebuild)
    _warn_embedder_mismatch(embedder)

    try:
        if source_type == "zotero":
            # Use the optimized Zotero path with full caching
            _agent = InCiteAgent.from_zotero(
                method=method,
                embedder_type=embedder,
                mode=mode,
                chunking_strategy=chunking,
            )
        elif source_type == "paperpile":
            _agent = InCiteAgent.from_paperpile(
                method=method,
                embedder_type=embedder,
                mode=mode,
                chunking_strategy=chunking,
            )
        else:
            source = _create_source(source_type)
            _agent = InCiteAgent.from_source(
                source,
                method=method,
                embedder_type=embedder,
                mode=mode,
                chunking_strategy=chunking,
            )
        chunking_str = f", chunking={chunking}" if mode == "paragraph" else ""
        logger.info(
            "Agent ready: %d papers, method=%s, embedder=%s, mode=%s%s",
            _agent.corpus_size,
            method,
            embedder,
            mode,
            chunking_str,
        )
    except Exception as exc:
        friendly = _diagnose_startup_error(exc)
        logger.error("Failed to load agent: %s\n%s", friendly, traceback.format_exc())
        # _agent stays None; /health will report not ready

    yield

    _agent = None


app = FastAPI(
    title="inCite",
    description="Citation recommendation API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$|^https://[a-z0-9-]+\.googleusercontent\.com$",
    allow_methods=["*"],
    allow_headers=["*"],
)


class PrivateNetworkAccessMiddleware(BaseHTTPMiddleware):
    """Handle Chrome Private Network Access preflight for Google Docs sidebar.

    When a public website (*.googleusercontent.com) fetches from localhost,
    Chrome sends Access-Control-Request-Private-Network: true in the preflight.
    The server must respond with Access-Control-Allow-Private-Network: true.
    """

    async def dispatch(self, request: Request, call_next):
        if (
            request.method == "OPTIONS"
            and request.headers.get("access-control-request-private-network") == "true"
        ):
            response = Response(status_code=200)
            response.headers["access-control-allow-private-network"] = "true"
            # Copy CORS headers so the preflight is complete
            origin = request.headers.get("origin", "")
            response.headers["access-control-allow-origin"] = origin
            response.headers["access-control-allow-methods"] = "*"
            response.headers["access-control-allow-headers"] = "*"
            return response
        response = await call_next(request)
        return response


app.add_middleware(PrivateNetworkAccessMiddleware)


# Mount Word add-in static files if available
try:
    from starlette.staticfiles import StaticFiles

    # Try dev path (source tree)
    _word_addin_dist = (
        Path(__file__).parent.parent.parent / "editor-plugins" / "word-incite" / "dist"
    )
    if _word_addin_dist.exists():
        app.mount(
            "/word-addin",
            StaticFiles(directory=str(_word_addin_dist), html=True),
            name="word-addin",
        )
        logger.info("Mounted Word add-in from %s", _word_addin_dist)
except Exception:
    pass  # Static files not available


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    from fastapi.responses import JSONResponse

    logger.error("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


def _get_display_mode(agent) -> str:
    """Get user-facing mode name from internal mode/chunking."""
    if agent.mode == "paper":
        return "paper"
    # For paragraph mode, return the chunking strategy as the mode
    return agent.chunking_strategy  # "paragraph", "sentence", or "grobid"


@app.get("/health")
def health():
    ready = _agent is not None
    result = {"status": "ready" if ready else "loading", "ready": ready}
    if ready:
        result["corpus_size"] = _agent.corpus_size
        result["mode"] = _get_display_mode(_agent)
    return result


@app.get("/stats")
def stats():
    agent = get_agent()
    return agent.get_stats()


@app.get("/config")
def config():
    agent = get_agent()
    from incite.retrieval.factory import get_available_embedders

    result = {
        "method": agent.method,
        "embedder": agent._embedder_type,
        "mode": _get_display_mode(agent),
        "two_stage": agent._two_stage,
        "available_embedders": list(get_available_embedders().keys()),
        "available_methods": ["neural", "bm25", "hybrid"],
        "available_modes": ["paper", "paragraph", "sentence", "grobid"],
    }

    # Report alpha if two-stage is active
    from incite.retrieval.two_stage import TwoStageRetriever

    if isinstance(agent._retriever, TwoStageRetriever):
        result["alpha"] = agent._retriever.alpha

    return result


@app.post("/recommend")
def recommend(req: RecommendRequest):
    agent = get_agent()
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be empty.")

    # Apply alpha override to TwoStageRetriever if provided
    from incite.retrieval.two_stage import TwoStageRetriever

    if req.alpha is not None and isinstance(agent._retriever, TwoStageRetriever):
        agent._retriever.alpha = req.alpha

    response = agent.recommend(
        query=req.query,
        k=req.k,
        author_boost=req.author_boost,
        cursor_sentence_index=req.cursor_sentence_index,
        focus_decay=req.focus_decay,
    )
    return response.to_dict()


@app.post("/batch")
def batch(req: BatchRequest):
    agent = get_agent()
    if not req.queries:
        raise HTTPException(status_code=422, detail="Queries list must not be empty.")
    responses = agent.batch_recommend(queries=req.queries, k=req.k, author_boost=req.author_boost)
    return [r.to_dict() for r in responses]


# --- Cloud processing pipeline endpoints (groundwork for hosted service) ---


@app.get("/pipeline/status")
def pipeline_status():
    """Get processing pipeline configuration and status.

    Returns the current processing mode (local/cloud) and whether
    a cloud pipeline is configured. Used by the setup wizard and
    tray app to show processing state.
    """
    try:
        from incite.webapp.state import get_config

        config = get_config()
        processing = config.get("processing", {})
        cloud = config.get("cloud", {})

        result = {
            "mode": processing.get("mode", "local"),
            "cloud_configured": bool(cloud.get("api_url")),
        }

        # If cloud is configured, check its health
        if cloud.get("api_url"):
            from incite.corpus.pipeline import CloudPipeline

            pipeline = CloudPipeline(
                api_url=cloud["api_url"],
                api_key=cloud.get("api_key"),
            )
            result["cloud_healthy"] = pipeline.check_health()

        return result
    except Exception as e:
        return {"mode": "local", "cloud_configured": False, "error": str(e)}
