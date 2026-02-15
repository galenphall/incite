"""inCite menu bar (tray) application using rumps."""

import logging
import subprocess
import webbrowser

import rumps

from incite.tray.server_manager import LOG_FILE, ServerManager

logger = logging.getLogger(__name__)


class InCiteTray(rumps.App):
    """macOS menu bar app for controlling the inCite API server."""

    def __init__(
        self,
        auto_start: bool = True,
        embedder: str = "minilm",
        method: str = "hybrid",
        mode: str = "paper",
        port: int = 8230,
    ):
        super().__init__("inCite", quit_button=None)

        self.server = ServerManager(
            port=port,
            embedder=embedder,
            method=method,
            mode=mode,
        )
        self.auto_start = auto_start
        self._port = port

        # Read-only status items (no callback)
        self.status_item = rumps.MenuItem("Status: Checking...")
        self.status_item.set_callback(None)
        self.papers_item = rumps.MenuItem("Papers: --")
        self.papers_item.set_callback(None)

        self.menu = [
            self.status_item,
            self.papers_item,
            None,  # separator
            rumps.MenuItem("Start Server", callback=self.on_start),
            rumps.MenuItem("Stop Server", callback=self.on_stop),
            None,
            rumps.MenuItem("Open Google Docs", callback=self.on_open_google_docs),
            rumps.MenuItem("Open Webapp", callback=self.on_open_webapp),
            rumps.MenuItem("Open API Docs", callback=self.on_open_docs),
            None,
            rumps.MenuItem("Run Setup...", callback=self.on_run_setup),
            rumps.MenuItem("View Log", callback=self.on_view_log),
            rumps.MenuItem("Troubleshooting", callback=self.on_troubleshooting),
            None,
            rumps.MenuItem("Quit", callback=self.on_quit),
        ]

        if self.auto_start and not self.server.is_running():
            self.server.start()

    # ------------------------------------------------------------------
    # Menu callbacks
    # ------------------------------------------------------------------

    def on_start(self, _):
        if self.server.is_running():
            rumps.notification("inCite", "", "Server is already running.")
            return
        ok = self.server.start()
        if ok:
            rumps.notification("inCite", "", "Server starting...")
        else:
            rumps.notification("inCite", "", "Failed to start server. Check the log.")

    def on_stop(self, _):
        stopped = self.server.stop()
        if stopped:
            rumps.notification("inCite", "", "Server stopped.")
        else:
            rumps.notification("inCite", "", "Server is not running.")

    def on_open_google_docs(self, _):
        webbrowser.open("https://docs.google.com")

    def on_open_webapp(self, _):
        webbrowser.open("http://localhost:8501")

    def on_open_docs(self, _):
        webbrowser.open(f"http://localhost:{self._port}/docs")

    def on_run_setup(self, _):
        subprocess.Popen(  # noqa: S603, S607
            [
                "osascript",
                "-e",
                'tell application "Terminal" to do script "~/.incite/venv/bin/incite setup"',
            ]
        )

    def on_view_log(self, _):
        if LOG_FILE.exists():
            subprocess.run(["open", str(LOG_FILE)])  # noqa: S603, S607
        else:
            rumps.notification("inCite", "", "No log file found.")

    def on_troubleshooting(self, _):
        subprocess.Popen(  # noqa: S603, S607
            [
                "osascript",
                "-e",
                'tell application "Terminal" to do script "~/.incite/venv/bin/incite doctor"',
            ]
        )

    def on_quit(self, _):
        self.server.stop()
        rumps.quit_application()

    # ------------------------------------------------------------------
    # Health polling
    # ------------------------------------------------------------------

    @rumps.timer(5)
    def poll_health(self, _):
        """Poll /health every 5 seconds and update the menu bar."""
        health = self.server.check_health()
        if health and health.get("ready"):
            self.title = "inCite \u2713"
            self.status_item.title = "Status: Running"
            corpus_size = health.get("corpus_size", "?")
            self.papers_item.title = f"Papers: {corpus_size}"
        elif self.server.is_running():
            self.title = "inCite \u27f3"
            self.status_item.title = "Status: Loading..."
        else:
            self.title = "inCite"
            self.status_item.title = "Status: Stopped"
            self.papers_item.title = "Papers: --"
