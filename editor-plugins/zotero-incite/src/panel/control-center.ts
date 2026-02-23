/**
 * inCite Control Center — loaded from chrome:// in a Zotero window.
 * Two-mode panel: local server status or cloud library management.
 */
import type { ClientConfig, HealthResponse, ServerConfigResponse, LibraryStatusResponse } from "@incite/shared";
import { InCiteClient, FetchTransport } from "@incite/shared";

/** Base URL for the Zotero connector server (panel is loaded via chrome://, not HTTP). */
const CONNECTOR = "http://localhost:23119";

// --- Settings ---

interface PanelSettings {
	apiMode: string;
	localUrl: string;
	cloudUrl: string;
	apiToken: string;
}

const DEFAULT_SETTINGS: PanelSettings = {
	apiMode: "local",
	localUrl: "http://127.0.0.1:8230",
	cloudUrl: "https://inciteref.com",
	apiToken: "",
};

let settings: PanelSettings = { ...DEFAULT_SETTINGS };
let pollTimer: ReturnType<typeof setTimeout> | null = null;
/** Generation counter — stale in-flight polls check this to avoid overwriting fresh results. */
let pollGeneration = 0;

async function loadSettings(): Promise<void> {
	try {
		const resp = await fetch(CONNECTOR + "/incite/settings");
		if (resp.ok) {
			const data = await resp.json();
			settings = { ...DEFAULT_SETTINGS, ...data };
		}
	} catch {
		// Use defaults
	}
}

async function saveSettings(): Promise<void> {
	try {
		const resp = await fetch(CONNECTOR + "/incite/settings", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(settings),
		});
		if (resp.ok) {
			showToast("Settings saved");
		} else {
			showToast("Failed to save settings (HTTP " + resp.status + ")");
		}
	} catch {
		showToast("Failed to save settings");
	}
}

function makeClient(): InCiteClient {
	const config: ClientConfig = {
		apiMode: settings.apiMode as "local" | "cloud",
		localUrl: settings.localUrl,
		cloudUrl: settings.cloudUrl,
		apiToken: settings.apiToken,
	};
	return new InCiteClient(config, new FetchTransport());
}

// --- DOM references ---

const statusDot = document.getElementById("status-dot")!;
const statusLabel = document.getElementById("status-label")!;
const settingsPanel = document.getElementById("settings-panel")!;
const btnSettingsToggle = document.getElementById("btn-settings-toggle") as HTMLButtonElement;
const btnSettingsSave = document.getElementById("btn-settings-save") as HTMLButtonElement;

const localView = document.getElementById("local-view")!;
const localConnected = document.getElementById("local-connected")!;
const localDisconnected = document.getElementById("local-disconnected")!;

const cloudView = document.getElementById("cloud-view")!;
const cloudConnected = document.getElementById("cloud-connected")!;
const cloudDisconnected = document.getElementById("cloud-disconnected")!;
const cloudError = document.getElementById("cloud-error")!;
const cloudErrorMessage = document.getElementById("cloud-error-message")!;

const btnSync = document.getElementById("btn-sync") as HTMLButtonElement;
const btnUpload = document.getElementById("btn-upload") as HTMLButtonElement;
const btnInstallIncite = document.getElementById("btn-install-incite") as HTMLButtonElement;
const btnStartServer = document.getElementById("btn-start-server") as HTMLButtonElement;
const btnStopServer = document.getElementById("btn-stop-server") as HTMLButtonElement;
const btnProcessLibrary = document.getElementById("btn-process-library") as HTMLButtonElement;

// --- Event listeners ---

btnSettingsToggle.addEventListener("click", () => {
	const visible = settingsPanel.style.display !== "none";
	settingsPanel.style.display = visible ? "none" : "block";
	if (!visible) {
		populateSettingsUI();
	}
});

btnSettingsSave.addEventListener("click", async () => {
	readSettingsFromUI();
	await saveSettings();
	settingsPanel.style.display = "none";
	switchView();
	startPolling();
});

document.getElementById("setting-apiMode")?.addEventListener("change", () => {
	updateSettingsVisibility();
});

btnInstallIncite?.addEventListener("click", () => installIncite());
btnStartServer?.addEventListener("click", () => startLocalServer());
btnStopServer?.addEventListener("click", () => stopLocalServer());
btnProcessLibrary?.addEventListener("click", () => processLibrary());
btnUpload?.addEventListener("click", () => uploadLibrary());


btnSync.addEventListener("click", () => syncLibrary());

// --- System management (local mode) ---

async function checkSystem(): Promise<void> {
	const pyStatus = document.getElementById("setup-python-status")!;
	const inciteStatus = document.getElementById("setup-incite-status")!;
	const installRow = document.getElementById("setup-install-row")!;
	const pythonMissing = document.getElementById("setup-python-missing")!;

	// Check Python
	pyStatus.innerHTML = '<span class="setup-spinner"></span> Checking...';
	try {
		const resp = await fetch(CONNECTOR + "/incite/system/python");
		const data = await resp.json();
		if (data.found) {
			pyStatus.innerHTML = `<span class="setup-check">\u2713</span> ${data.version}`;
			pythonMissing.style.display = "none";

			// Check incite
			inciteStatus.innerHTML = '<span class="setup-spinner"></span> Checking...';
			const resp2 = await fetch(CONNECTOR + "/incite/system/incite");
			const data2 = await resp2.json();
			if (data2.installed) {
				inciteStatus.innerHTML = `<span class="setup-check">\u2713</span> ${data2.version}`;
				installRow.style.display = "none";
				btnStartServer.disabled = false;
				btnProcessLibrary.disabled = false;
			} else {
				inciteStatus.innerHTML = '<span class="setup-x">\u2717</span> Not installed';
				installRow.style.display = "flex";
				btnStartServer.disabled = true;
				btnProcessLibrary.disabled = true;
			}
		} else {
			pyStatus.innerHTML = '<span class="setup-x">\u2717</span> Not found';
			inciteStatus.innerHTML = '<span class="setup-x">\u2717</span> —';
			pythonMissing.style.display = "block";
			installRow.style.display = "none";
			btnStartServer.disabled = true;
			btnProcessLibrary.disabled = true;
		}
	} catch {
		pyStatus.innerHTML = '<span class="setup-x">\u2717</span> Error';
		inciteStatus.innerHTML = '<span class="setup-x">\u2717</span> —';
	}
}

async function installIncite(): Promise<void> {
	const installRow = document.getElementById("setup-install-row")!;
	const progressDiv = document.getElementById("setup-install-progress")!;
	const progressText = document.getElementById("install-progress-text")!;

	btnInstallIncite.disabled = true;
	installRow.style.display = "none";
	progressDiv.style.display = "block";
	progressText.textContent = "Setting up local server...";

	try {
		await fetch(CONNECTOR + "/incite/system/install", { method: "POST" });

		// Poll for completion
		let pollCount = 0;
		const poll = async () => {
			pollCount++;
			try {
				const resp = await fetch(CONNECTOR + "/incite/system/install/status");
				const state = await resp.json();
				if (state.status === "running" || state.status === "idle") {
					// "idle" can happen if poll fires before the async install sets state
					progressText.textContent = state.status === "idle" ? "Starting install..." : "Installing... (this may take a few minutes)";
					setTimeout(poll, 3000);
				} else if (state.status === "done") {
					progressDiv.style.display = "none";
					showToast("inCite installed successfully");
					checkSystem();
				} else if (state.status === "error") {
					const out = state.output ?? "";
					progressText.textContent = "Install failed: " + (out.length > 500 ? "..." + out.slice(-500) : out);
					btnInstallIncite.disabled = false;
					installRow.style.display = "flex";
				}
			} catch {
				// Network error polling — keep trying for a while
				if (pollCount < 60) {
					setTimeout(poll, 3000);
				} else {
					progressText.textContent = "Install status unknown — check Zotero console";
					btnInstallIncite.disabled = false;
					installRow.style.display = "flex";
				}
			}
		};
		setTimeout(poll, 3000);
	} catch (err) {
		progressText.textContent = "Install failed";
		btnInstallIncite.disabled = false;
		installRow.style.display = "flex";
	}
}

async function startLocalServer(): Promise<void> {
	btnStartServer.disabled = true;
	btnStartServer.textContent = "Starting...";
	const serverStatus = document.getElementById("setup-server-status")!;
	serverStatus.textContent = "Starting...";

	try {
		await fetch(CONNECTOR + "/incite/system/server/start", { method: "POST" });
		showToast("Server starting...");
		// Normal polling will pick up the running server
		setTimeout(() => startPolling(5_000), 3000);
	} catch (err) {
		showToast("Failed to start server");
		btnStartServer.disabled = false;
		btnStartServer.textContent = "Start Server";
		serverStatus.textContent = "Stopped";
	}
}

async function stopLocalServer(): Promise<void> {
	btnStopServer.disabled = true;
	try {
		await fetch(CONNECTOR + "/incite/system/server/stop", { method: "POST" });
		showToast("Server stopped");
		startPolling();
	} catch (err) {
		showToast("Failed to stop server");
	}
	btnStopServer.disabled = false;
}

/** Update the process progress bar UI from a status response. */
function updateProcessProgressBar(state: { status: string; output: string; stage?: string; progress?: number; detail?: string }): void {
	const progressDiv = document.getElementById("setup-process-progress")!;
	const stageText = document.getElementById("process-stage-text")!;
	const pctText = document.getElementById("process-progress-pct")!;
	const progressFill = document.getElementById("process-progress-fill")! as HTMLElement;
	const detailText = document.getElementById("process-detail-text")!;

	progressDiv.style.display = "block";

	if (state.status === "error") {
		stageText.textContent = state.stage || "Error";
		stageText.style.color = "var(--red, #e55)";
		pctText.textContent = "";
		progressFill.style.width = "0%";
		progressFill.style.background = "var(--red, #e55)";
		detailText.textContent = state.detail || state.output || "";
		detailText.style.color = "var(--red, #e55)";
		return;
	}

	// Reset error styling
	stageText.style.color = "";
	progressFill.style.background = "";
	detailText.style.color = "";

	stageText.textContent = state.stage || state.output || "Processing...";
	const pct = state.progress ?? 0;
	pctText.textContent = `${pct}%`;
	progressFill.style.width = `${pct}%`;
	detailText.textContent = state.detail || "";
}

async function processLibrary(): Promise<void> {
	const progressDiv = document.getElementById("setup-process-progress")!;
	const processStatus = document.getElementById("setup-process-status")!;

	btnProcessLibrary.disabled = true;
	updateProcessProgressBar({ status: "running", output: "Starting library processing...", stage: "Starting...", progress: 0 });
	processStatus.textContent = "Processing...";

	try {
		await fetch(CONNECTOR + "/incite/system/process", { method: "POST" });

		let pollCount = 0;
		const poll = async () => {
			pollCount++;
			try {
				const resp = await fetch(CONNECTOR + "/incite/system/process/status");
				const state = await resp.json() as { status: string; output: string; stage?: string; progress?: number; detail?: string };
				if (state.status === "running" || state.status === "idle") {
					updateProcessProgressBar(state);
					setTimeout(poll, 3000);
				} else if (state.status === "done") {
					updateProcessProgressBar({ status: "done", output: "Server ready", stage: "Server ready", progress: 100 });
					setTimeout(() => {
						progressDiv.style.display = "none";
						processStatus.textContent = "Ready";
					}, 1000);
					showToast("Library processed successfully");
					startPolling(5_000);
				} else if (state.status === "error") {
					const out = state.output ?? "";
					const stageText = document.getElementById("process-stage-text")!;
					stageText.textContent = "Failed: " + (out.length > 300 ? "..." + out.slice(-300) : out);
					const pctText = document.getElementById("process-progress-pct")!;
					pctText.textContent = "";
					processStatus.textContent = "Error";
					btnProcessLibrary.disabled = false;
				}
			} catch {
				if (pollCount < 60) {
					setTimeout(poll, 3000);
				} else {
					const stageText = document.getElementById("process-stage-text")!;
					stageText.textContent = "Status unknown — check Zotero console";
					btnProcessLibrary.disabled = false;
				}
			}
		};
		setTimeout(poll, 3000);
	} catch {
		const stageText = document.getElementById("process-stage-text")!;
		stageText.textContent = "Failed to start processing";
		processStatus.textContent = "Error";
		btnProcessLibrary.disabled = false;
	}
}

async function uploadLibrary(): Promise<void> {
	const uploadProgress = document.getElementById("upload-progress")!;
	const uploadStage = document.getElementById("upload-stage")!;
	const uploadCount = document.getElementById("upload-count")!;
	const uploadFill = document.getElementById("upload-fill")! as HTMLElement;

	btnUpload.disabled = true;
	btnUpload.textContent = "Uploading...";
	uploadProgress.style.display = "block";
	uploadStage.textContent = "Starting upload...";
	uploadCount.textContent = "";
	uploadFill.style.width = "0%";

	try {
		await fetch(CONNECTOR + "/incite/cloud/upload", { method: "POST" });

		let pollCount = 0;
		const poll = async () => {
			pollCount++;
			try {
				const resp = await fetch(CONNECTOR + "/incite/cloud/upload/status");
				const state = await resp.json() as { status: string; message: string; current?: number; total?: number };

				if (state.status === "done") {
					uploadProgress.style.display = "none";
					showToast("Library uploaded successfully");
					btnUpload.disabled = false;
					btnUpload.textContent = "Upload Library";
					// Start rapid polling to show server-side processing progress
					startPolling(10_000);
					return;
				}

				if (state.status === "error") {
					uploadStage.textContent = "Upload failed";
					uploadCount.textContent = "";
					uploadFill.style.width = "0%";
					showToast("Upload failed: " + state.message);
					btnUpload.disabled = false;
					btnUpload.textContent = "Upload Library";
					return;
				}

				// Still in progress
				uploadStage.textContent = state.message || capitalize(state.status);
				if (state.current != null && state.total != null && state.total > 0) {
					const pct = Math.round((state.current / state.total) * 100);
					uploadCount.textContent = `${state.current}/${state.total}`;
					uploadFill.style.width = `${pct}%`;
				} else {
					uploadCount.textContent = "";
					uploadFill.style.width = "0%";
				}
				setTimeout(poll, 2000);
			} catch {
				if (pollCount < 120) {
					setTimeout(poll, 3000);
				} else {
					uploadStage.textContent = "Upload status unknown";
					btnUpload.disabled = false;
					btnUpload.textContent = "Upload Library";
				}
			}
		};
		setTimeout(poll, 2000);
	} catch {
		showToast("Failed to start upload");
		btnUpload.disabled = false;
		btnUpload.textContent = "Upload Library";
		uploadProgress.style.display = "none";
	}
}

async function syncLibrary(): Promise<void> {
	const syncProgress = document.getElementById("sync-progress")!;
	const syncStage = document.getElementById("sync-stage")!;
	const syncCount = document.getElementById("sync-count")!;

	btnSync.disabled = true;
	btnSync.textContent = "Syncing...";
	syncProgress.style.display = "block";
	syncStage.textContent = "Starting sync...";
	syncCount.textContent = "";

	try {
		const resp = await fetch(CONNECTOR + "/incite/cloud/sync", { method: "POST" });
		if (!resp.ok) {
			throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
		}

		let pollCount = 0;
		const poll = async () => {
			pollCount++;
			try {
				const statusResp = await fetch(CONNECTOR + "/incite/cloud/sync/status");
				const state = await statusResp.json() as {
					status: string; message: string;
					created?: number; skipped?: number; tagsAdded?: number;
				};

				if (state.status === "done") {
					syncProgress.style.display = "none";
					showToast(state.message || "Sync complete");
					btnSync.disabled = false;
					btnSync.textContent = "Sync Library";
					startPolling(10_000);
					return;
				}

				if (state.status === "error") {
					syncStage.textContent = "Sync failed";
					syncCount.textContent = "";
					showToast("Sync failed: " + state.message);
					btnSync.disabled = false;
					btnSync.textContent = "Sync Library";
					return;
				}

				// Still in progress
				syncStage.textContent = state.message || capitalize(state.status);
				if (state.created != null || state.skipped != null) {
					syncCount.textContent = `${state.created ?? 0} new, ${state.skipped ?? 0} existing`;
				}
				setTimeout(poll, 2000);
			} catch {
				if (pollCount < 120) {
					setTimeout(poll, 3000);
				} else {
					syncStage.textContent = "Sync status unknown";
					btnSync.disabled = false;
					btnSync.textContent = "Sync Library";
				}
			}
		};
		setTimeout(poll, 2000);
	} catch (err) {
		showToast("Failed to start sync: " + (err instanceof Error ? err.message : String(err)));
		btnSync.disabled = false;
		btnSync.textContent = "Sync Library";
		syncProgress.style.display = "none";
	}
}

// --- Initialize ---

loadSettings().then(() => {
	populateSettingsUI();
	switchView();
	startPolling();
	if (settings.apiMode === "local") {
		checkSystem();
	}
});

// --- View switching ---

function switchView(): void {
	const isCloud = settings.apiMode === "cloud";
	localView.style.display = isCloud ? "none" : "block";
	cloudView.style.display = isCloud ? "block" : "none";
	// Don't call refreshCloudStatus/refreshLocalStatus here —
	// startPolling() is always called right after switchView() and handles the first poll.
}

// --- Polling ---

function startPolling(intervalMs?: number): void {
	if (pollTimer) {
		clearTimeout(pollTimer);
		pollTimer = null;
	}
	const gen = ++pollGeneration;

	const poll = () => {
		if (gen !== pollGeneration) return; // superseded
		if (settings.apiMode === "cloud") {
			refreshCloudStatus(gen).then((nextInterval) => {
				if (gen !== pollGeneration) return;
				pollTimer = setTimeout(poll, nextInterval ?? intervalMs ?? 60_000);
			});
		} else {
			refreshLocalStatus().then(() => {
				if (gen !== pollGeneration) return;
				pollTimer = setTimeout(poll, intervalMs ?? 30_000);
			});
		}
	};

	poll();
}

// --- Local mode ---

async function refreshLocalStatus(): Promise<void> {
	try {
		const client = makeClient();
		const health = await client.health();

		// Update setup controls for running state
		const serverStatus = document.getElementById("setup-server-status");
		if (serverStatus) serverStatus.textContent = "Running";
		if (btnStartServer) { btnStartServer.style.display = "none"; }
		if (btnStopServer) { btnStopServer.style.display = "inline-block"; }

		if (!health.ready) {
			// Server is running but not ready — could be loading or failed
			localConnected.style.display = "none";
			localDisconnected.style.display = "block";

			const processStatus = document.getElementById("setup-process-status");

			// Fetch structured status (auto-detects errors via log parsing)
			try {
				const resp = await fetch(CONNECTOR + "/incite/system/process/status");
				const state = await resp.json() as { status: string; output: string; stage?: string; progress?: number; detail?: string };

				if (state.status === "error") {
					setStatus("error", "Server error");
					if (processStatus) processStatus.textContent = "Error";
					updateProcessProgressBar(state);
					btnProcessLibrary.disabled = false;
				} else {
					setStatus("connected", "Loading...");
					if (processStatus) processStatus.textContent = "Loading...";
					btnProcessLibrary.disabled = true;
					if (state.stage || state.progress) {
						updateProcessProgressBar(state);
					}
				}
			} catch {
				setStatus("connected", "Loading...");
				if (processStatus) processStatus.textContent = "Loading...";
				btnProcessLibrary.disabled = true;
			}
			return;
		}

		setStatus("connected", `Connected — ${health.corpus_size ?? "?"} papers`);
		localConnected.style.display = "block";
		localDisconnected.style.display = "none";

		// Library is processed if server is running with papers
		const processStatus = document.getElementById("setup-process-status");
		if (processStatus) processStatus.textContent = `Ready (${health.corpus_size ?? 0} papers)`;

		// Hide progress bar if it was showing
		const progressDiv = document.getElementById("setup-process-progress");
		if (progressDiv) progressDiv.style.display = "none";
		btnProcessLibrary.disabled = false;

		document.getElementById("local-corpus")!.textContent = String(health.corpus_size ?? "—");

		// Try to get server config
		try {
			const config = await client.serverConfig();
			document.getElementById("local-embedder")!.textContent = config.embedder;
			document.getElementById("local-method")!.textContent = config.method;
			document.getElementById("local-mode")!.textContent = config.mode;
		} catch {
			document.getElementById("local-embedder")!.textContent = "—";
			document.getElementById("local-method")!.textContent = "—";
			document.getElementById("local-mode")!.textContent = "—";
		}
	} catch {
		setStatus("error", "Not connected");
		localConnected.style.display = "none";
		localDisconnected.style.display = "block";

		// Update setup controls for stopped state
		const serverStatus = document.getElementById("setup-server-status");
		if (serverStatus) serverStatus.textContent = "Stopped";
		if (btnStartServer) {
			btnStartServer.style.display = "inline-block";
			btnStartServer.textContent = "Start Server";
		}
		if (btnStopServer) { btnStopServer.style.display = "none"; }
	}
}

// --- Cloud mode ---

async function refreshCloudStatus(gen?: number): Promise<number> {
	try {
		// Use connector-server proxy to avoid CORS issues (panel is served from localhost)
		const resp = await fetch(CONNECTOR + "/incite/cloud/status");
		if (!resp.ok) {
			throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
		}
		// If this poll was superseded by a newer one, discard the result
		if (gen != null && gen !== pollGeneration) return 60_000;

		const status = await resp.json() as LibraryStatusResponse;

		// If token is missing, show disconnected state
		if (status.library_status === "no_token") {
			setStatus("error", "Not connected");
			cloudConnected.style.display = "none";
			cloudDisconnected.style.display = "block";
			cloudError.style.display = "none";
			return 60_000;
		}

		setStatus("connected", statusText(status.library_status));
		cloudConnected.style.display = "block";
		cloudDisconnected.style.display = "none";
		cloudError.style.display = "none";

		document.getElementById("cloud-status")!.textContent = status.library_status;
		document.getElementById("cloud-papers")!.textContent = String(status.num_papers);
		document.getElementById("cloud-chunks")!.textContent = String(status.num_chunks);
		document.getElementById("cloud-fulltext-papers")!.textContent = String(status.grobid_fulltext_papers);
		document.getElementById("cloud-fulltext-chunks")!.textContent = String(status.grobid_fulltext_chunks);
		document.getElementById("cloud-abstract-only")!.textContent = String(status.abstract_only_papers);

		// Progress bar
		const progressSection = document.getElementById("cloud-progress")!;
		const terminalStates = new Set(["idle", "completed", "failed"]);
		if (status.job_status && !terminalStates.has(status.job_status)) {
			progressSection.style.display = "block";
			document.getElementById("progress-stage")!.textContent = capitalize(status.stage ?? "Processing");
			if (status.current != null && status.total != null && status.total > 0) {
				const pct = Math.round((status.current / status.total) * 100);
				document.getElementById("progress-count")!.textContent = `${status.current}/${status.total}`;
				(document.getElementById("progress-fill")! as HTMLElement).style.width = `${pct}%`;
			} else {
				document.getElementById("progress-count")!.textContent = "";
				(document.getElementById("progress-fill")! as HTMLElement).style.width = "0%";
			}
			// Rapid polling during active job
			return 10_000;
		} else {
			progressSection.style.display = "none";
		}

		// Server-side error from a past job
		if (status.error) {
			cloudError.style.display = "block";
			cloudErrorMessage.textContent = status.error;
		}

		return 60_000;
	} catch (err) {
		// If this poll was superseded, discard
		if (gen != null && gen !== pollGeneration) return 60_000;

		const msg = err instanceof Error ? err.message : String(err);
		console.error("inCite cloud status error:", msg);

		// If we have a token, show the error details instead of generic "Not Connected"
		if (settings.apiToken) {
			setStatus("error", "Connection error");
			cloudConnected.style.display = "none";
			cloudDisconnected.style.display = "none";
			cloudError.style.display = "block";
			cloudErrorMessage.textContent = msg;
		} else {
			setStatus("error", "Not connected");
			cloudConnected.style.display = "none";
			cloudDisconnected.style.display = "block";
			cloudError.style.display = "none";
		}
		return 60_000;
	}
}

// --- Status helpers ---

function setStatus(state: "connected" | "error", label: string): void {
	statusDot.className = `status-dot ${state}`;
	statusDot.title = label;
	statusLabel.textContent = label;
}

function statusText(libraryStatus: string): string {
	switch (libraryStatus) {
		case "ready": return "Library ready";
		case "processing": return "Processing...";
		case "no_library": return "No library";
		default: return libraryStatus;
	}
}

function capitalize(s: string): string {
	return s.charAt(0).toUpperCase() + s.slice(1);
}

// --- Settings UI helpers ---

function populateSettingsUI(): void {
	(document.getElementById("setting-apiMode") as HTMLSelectElement).value = settings.apiMode;
	(document.getElementById("setting-localUrl") as HTMLInputElement).value = settings.localUrl;
	(document.getElementById("setting-cloudUrl") as HTMLInputElement).value = settings.cloudUrl;
	(document.getElementById("setting-apiToken") as HTMLInputElement).value = settings.apiToken;
	updateSettingsVisibility();
}

function readSettingsFromUI(): void {
	settings.apiMode = (document.getElementById("setting-apiMode") as HTMLSelectElement).value;
	settings.localUrl = (document.getElementById("setting-localUrl") as HTMLInputElement).value;
	settings.cloudUrl = (document.getElementById("setting-cloudUrl") as HTMLInputElement).value;
	settings.apiToken = (document.getElementById("setting-apiToken") as HTMLInputElement).value;
}

function updateSettingsVisibility(): void {
	const mode = (document.getElementById("setting-apiMode") as HTMLSelectElement).value;
	const isCloud = mode === "cloud";
	document.getElementById("label-localUrl")!.style.display = isCloud ? "none" : "flex";
	document.getElementById("label-cloudUrl")!.style.display = isCloud ? "flex" : "none";
	document.getElementById("label-apiToken")!.style.display = isCloud ? "flex" : "none";
}

// --- Toast ---

function showToast(message: string): void {
	const existing = document.querySelector(".toast");
	if (existing) existing.remove();

	const toast = document.createElement("div");
	toast.className = "toast";
	toast.textContent = message;
	document.body.appendChild(toast);
	setTimeout(() => toast.remove(), 2500);
}
