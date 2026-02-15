"""Manage the inCite API server as a subprocess."""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

logger = logging.getLogger(__name__)

INCITE_DIR = Path.home() / ".incite"
PID_FILE = INCITE_DIR / "server.pid"
LOG_FILE = INCITE_DIR / "server.log"


class ServerManager:
    """Start/stop the uvicorn API server and monitor its health."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8230,
        embedder: str = "minilm",
        method: str = "hybrid",
        mode: str = "paper",
    ):
        self.host = host
        self.port = port
        self.embedder = embedder
        self.method = method
        self.mode = mode
        self._process: subprocess.Popen | None = None

        # Recover from a previous run if a PID file exists
        self._recover_from_pid_file()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Launch the uvicorn subprocess.

        Returns True if the server was started, False if already running.
        """
        if self.is_running():
            logger.info("Server already running (PID %s)", self._process.pid)
            return False

        INCITE_DIR.mkdir(parents=True, exist_ok=True)

        env = {**os.environ}
        env["INCITE_METHOD"] = self.method
        env["INCITE_EMBEDDER"] = self.embedder
        env["INCITE_MODE"] = self.mode
        # For paper mode the chunking is "paragraph" (evidence lookup)
        env["INCITE_CHUNKING"] = "paragraph"

        log_fh = open(LOG_FILE, "a")  # noqa: SIM115
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "incite.api:app",
                "--host",
                self.host,
                "--port",
                str(self.port),
            ],
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )

        PID_FILE.write_text(str(self._process.pid))
        logger.info("Server started (PID %s)", self._process.pid)
        return True

    def stop(self) -> bool:
        """Stop the server subprocess. Returns True if it was stopped."""
        if self._process is None:
            return False

        if self._process.poll() is not None:
            # Already exited
            self._cleanup()
            return False

        # Graceful shutdown
        self._process.send_signal(signal.SIGTERM)
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Server did not stop after SIGTERM, sending SIGKILL")
            self._process.kill()
            self._process.wait(timeout=3)

        self._cleanup()
        logger.info("Server stopped")
        return True

    def is_running(self) -> bool:
        """Return True if the subprocess is alive."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def check_health(self) -> dict | None:
        """GET /health and return parsed JSON, or None on failure."""
        url = f"http://{self.host}:{self.port}/health"
        try:
            with urlopen(url, timeout=2) as resp:  # noqa: S310
                return json.loads(resp.read())
        except (URLError, OSError, json.JSONDecodeError, ValueError):
            return None

    def get_log_tail(self, lines: int = 20) -> str:
        """Return the last *lines* lines of server.log."""
        if not LOG_FILE.exists():
            return "(no log file)"
        text = LOG_FILE.read_text(encoding="utf-8", errors="replace")
        all_lines = text.splitlines()
        return "\n".join(all_lines[-lines:])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recover_from_pid_file(self):
        """If a PID file exists, check whether that process is still alive."""
        if not PID_FILE.exists():
            return
        try:
            pid = int(PID_FILE.read_text().strip())
        except (ValueError, OSError):
            PID_FILE.unlink(missing_ok=True)
            return

        # Check whether the PID is alive
        try:
            os.kill(pid, 0)  # signal 0 = existence check
        except ProcessLookupError:
            # Stale PID file
            PID_FILE.unlink(missing_ok=True)
            return
        except PermissionError:
            pass  # process exists but we don't own it — treat as running

        # Process is alive. We cannot attach a Popen object to an existing
        # PID, but we can still stop it via os.kill.
        logger.info("Recovered running server (PID %s) from PID file", pid)
        self._process = _ExternalProcess(pid)

    def _cleanup(self):
        self._process = None
        PID_FILE.unlink(missing_ok=True)


class _ExternalProcess:
    """Minimal stand-in for subprocess.Popen when we recover a PID from disk.

    Only implements the subset of the Popen API that ServerManager uses.
    """

    def __init__(self, pid: int):
        self.pid = pid

    def poll(self) -> int | None:
        try:
            os.kill(self.pid, 0)
            return None  # still alive
        except ProcessLookupError:
            return 1  # exited

    def send_signal(self, sig: int):
        os.kill(self.pid, sig)

    def kill(self):
        os.kill(self.pid, signal.SIGKILL)

    def wait(self, timeout: float | None = None):
        """Block until the process exits or *timeout* seconds elapse."""
        deadline = time.monotonic() + (timeout or 30)
        while time.monotonic() < deadline:
            try:
                os.kill(self.pid, 0)
            except ProcessLookupError:
                return 0
            time.sleep(0.2)
        raise subprocess.TimeoutExpired(cmd="server", timeout=timeout)
