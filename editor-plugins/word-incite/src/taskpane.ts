/**
 * inCite Word Add-in Task Pane — Orchestrator
 *
 * Office.js task pane that provides citation recommendations from
 * the inCite cloud server or a local `incite serve` instance.
 *
 * This module is the entry point: it wires up UI, delegates to
 * focused modules for settings, API, context extraction, rendering,
 * bibliography, and citation storage.
 */

import {
	formatCitation,
	formatMultiCitation,
	stripCitations,
	CitationTracker,
	escapeHtml,
} from "@incite/shared";
import type { ApiMode, Recommendation } from "@incite/shared";

import { settings, loadSettings, saveSettings } from "./settings";
import { createClient } from "./api";
import { getContextFromWord } from "./context";
import { renderResults, renderSelectionBar } from "./results-renderer";
import { renderBibliography } from "./bibliography";
import { LocalStorageCitationStorage } from "./citation-storage";

// --- Office.js Declarations ---

/* global Office, Word */
declare const Office: {
	onReady: (callback: (info: { host: string }) => void) => void;
	HostType: { Word: string };
};
declare const Word: {
	run: <T>(callback: (context: WordContext) => Promise<T>) => Promise<T>;
	InsertLocation: { after: string };
};

interface WordContext {
	document: {
		getSelection: () => WordRange;
		properties: {
			customProperties: {
				getItemOrNullObject: (name: string) => CustomProperty;
				add: (name: string, value: string) => CustomProperty;
			};
		};
	};
	sync: () => Promise<void>;
}

interface CustomProperty {
	isNullObject: boolean;
	value: string;
	load: (options: { select: string }) => CustomProperty;
}

interface WordRange {
	text: string;
	insertText: (text: string, location: string) => void;
}

// --- State ---

let loading = false;
let lastResults: Recommendation[] = [];
const selectedRecs: Map<string, Recommendation> = new Map();

// --- Citation Tracking ---

let tracker: CitationTracker | null = null;

async function getOrCreateDocKey(): Promise<string> {
	return Word.run(async (context) => {
		const prop = context.document.properties.customProperties.getItemOrNullObject("incite_doc_id");
		prop.load({ select: "value" });
		await context.sync();

		if (!prop.isNullObject) {
			return prop.value;
		}

		const docId = crypto.randomUUID();
		context.document.properties.customProperties.add("incite_doc_id", docId);
		await context.sync();
		return docId;
	});
}

// --- DOM Helpers ---

function getEl(id: string): HTMLElement {
	return document.getElementById(id)!;
}

// --- Office.js Entry Point ---

Office.onReady((info) => {
	if (info.host === Office.HostType.Word) {
		initApp();
	}
});

// --- Initialization ---

function updateModeVisibility(): void {
	const cloudFields = getEl("cloudFields");
	const localFields = getEl("localFields");
	if (settings.apiMode === "cloud") {
		cloudFields.style.display = "";
		localFields.style.display = "none";
	} else {
		cloudFields.style.display = "none";
		localFields.style.display = "";
	}
}

async function initApp(): Promise<void> {
	// Load saved settings from localStorage
	loadSettings();

	// Initialize per-document citation tracker
	try {
		const docKey = await getOrCreateDocKey();
		tracker = new CitationTracker(new LocalStorageCitationStorage(), docKey);
		await tracker.load();
		doRenderBibliography();
	} catch (err) {
		console.error("Failed to initialize document tracker:", err);
		// Fallback to a session-level key
		tracker = new CitationTracker(new LocalStorageCitationStorage(), "word-fallback");
		await tracker.load();
		doRenderBibliography();
	}

	// Wire up buttons
	getEl("refreshBtn").addEventListener("click", onRefresh);
	getEl("settingsToggle").addEventListener("click", toggleSettings);

	// API mode select
	const modeSelect = getEl("apiMode") as HTMLSelectElement;
	modeSelect.value = settings.apiMode;
	modeSelect.addEventListener("change", () => {
		settings.apiMode = modeSelect.value as ApiMode;
		saveSettings();
		updateModeVisibility();
		checkHealth();
	});

	// API token
	const tokenInput = getEl("apiToken") as HTMLInputElement;
	tokenInput.value = settings.apiToken;
	tokenInput.addEventListener("change", () => {
		settings.apiToken = tokenInput.value.trim();
		saveSettings();
		checkHealth();
	});

	// Cloud URL
	const cloudUrlInput = getEl("cloudUrl") as HTMLInputElement;
	cloudUrlInput.value = settings.cloudUrl;
	cloudUrlInput.addEventListener("change", () => {
		settings.cloudUrl = cloudUrlInput.value.replace(/\/+$/, "");
		saveSettings();
		checkHealth();
	});

	// Local URL
	const localUrlInput = getEl("localUrl") as HTMLInputElement;
	localUrlInput.value = settings.localUrl;
	localUrlInput.addEventListener("change", () => {
		settings.localUrl = localUrlInput.value.replace(/\/+$/, "");
		saveSettings();
		checkHealth();
	});

	// Results count
	const kInput = getEl("kInput") as HTMLInputElement;
	kInput.value = String(settings.k);
	kInput.addEventListener("change", () => {
		const val = parseInt(kInput.value, 10);
		if (val >= 1 && val <= 50) {
			settings.k = val;
			saveSettings();
		}
	});

	// Citation format: dropdown + custom input
	const formatSelect = getEl("formatSelect") as HTMLSelectElement;
	const formatInput = getEl("insertFormat") as HTMLInputElement;

	// Set initial state: match saved format to dropdown or show custom
	const savedFormat = settings.insertFormat;
	const matchingOption = Array.from(formatSelect.options).find(
		(opt) => opt.value === savedFormat
	);
	if (matchingOption) {
		formatSelect.value = savedFormat;
		formatInput.style.display = "none";
	} else {
		formatSelect.value = "custom";
		formatInput.value = savedFormat;
		formatInput.style.display = "";
	}

	formatSelect.addEventListener("change", () => {
		if (formatSelect.value === "custom") {
			formatInput.style.display = "";
			formatInput.focus();
		} else {
			formatInput.style.display = "none";
			settings.insertFormat = formatSelect.value;
			saveSettings();
		}
	});

	formatInput.addEventListener("change", () => {
		if (formatSelect.value === "custom") {
			settings.insertFormat = formatInput.value;
			saveSettings();
		}
	});

	// Show paragraphs toggle
	const showParaCheckbox = getEl("showParagraphs") as HTMLInputElement;
	showParaCheckbox.checked = settings.showParagraphs;
	showParaCheckbox.addEventListener("change", () => {
		settings.showParagraphs = showParaCheckbox.checked;
		saveSettings();
		if (lastResults.length > 0) {
			doRenderResults(lastResults);
		}
	});

	// Set initial mode visibility
	updateModeVisibility();

	// Initial health check + polling
	checkHealth();
	setInterval(checkHealth, 30000);
}

// --- Settings Panel ---

function toggleSettings(): void {
	const panel = getEl("settingsPanel");
	panel.classList.toggle("visible");
}

// --- Server Health ---

async function checkHealth(): Promise<void> {
	const statusEl = getEl("status");
	try {
		const client = createClient();
		const health = await client.health();
		if (health.ready) {
			statusEl.textContent = `Connected (${health.corpus_size} papers, ${health.mode} mode)`;
			statusEl.className = "status connected";
			getEl("refreshBtn").removeAttribute("disabled");
		} else {
			statusEl.textContent = "Server loading...";
			statusEl.className = "status loading";
		}
	} catch {
		if (settings.apiMode === "cloud") {
			statusEl.textContent = "Cannot reach server";
		} else {
			statusEl.textContent = 'Server offline -- run "incite serve"';
		}
		statusEl.className = "status offline";
	}
}

// --- Recommendations ---

async function onRefresh(): Promise<void> {
	if (loading) return;
	setLoading(true);

	try {
		const ctx = await getContextFromWord();

		if (!ctx.text.trim()) {
			showError(
				"No text found. Place your cursor in a paragraph with text and try again.",
			);
			setLoading(false);
			return;
		}

		const cleaned = stripCitations(ctx.text);
		const client = createClient();
		const response = await client.recommend(
			cleaned,
			settings.k,
			settings.authorBoost,
		);

		lastResults = response.recommendations || [];

		if (lastResults.length === 0) {
			showEmpty();
		} else {
			doRenderResults(lastResults);
			renderTiming(response.timing);
		}
	} catch (err) {
		const modeHint =
			settings.apiMode === "cloud"
				? "Check your API token and server URL."
				: 'Is "incite serve" running?';
		showError(`Could not get recommendations: ${err}. ${modeHint}`);
	}

	setLoading(false);
}

// --- Citation Insertion ---

async function insertCitation(rec: Recommendation): Promise<void> {
	const citation = formatCitation(rec, settings.insertFormat);
	try {
		await Word.run(async (context) => {
			const selection = context.document.getSelection();
			selection.insertText(citation, Word.InsertLocation.after);
			await context.sync();
		});
		if (tracker) await tracker.track([rec]);
		doRenderBibliography();
		if (lastResults.length > 0) doRenderResults(lastResults);
		showStatus(`Inserted: ${citation}`);
		setTimeout(() => showStatus(""), 2000);
	} catch (err) {
		showError(`Could not insert citation: ${err}`);
	}
}

async function insertMultiCitation(recs: Recommendation[]): Promise<void> {
	if (recs.length === 0) return;
	const citation = formatMultiCitation(recs, settings.insertFormat);
	try {
		await Word.run(async (context) => {
			const selection = context.document.getSelection();
			selection.insertText(citation, Word.InsertLocation.after);
			await context.sync();
		});
		if (tracker) await tracker.track(recs);
		selectedRecs.clear();
		doRenderBibliography();
		if (lastResults.length > 0) doRenderResults(lastResults);
		showStatus(`Inserted ${recs.length} citations: ${citation}`);
		setTimeout(() => showStatus(""), 2000);
	} catch (err) {
		showError(`Could not insert citations: ${err}`);
	}
}

// --- Rendering Delegates ---

/** Render recommendation results into #results. */
function doRenderResults(results: Recommendation[]): void {
	const container = getEl("results");
	renderResults(results, container, {
		showParagraphs: settings.showParagraphs,
		selectedRecs,
		tracker: tracker ?? undefined,
	}, {
		onInsert: (rec) => insertCitation(rec),
		onCopy: (rec, btn) => {
			const citation = formatCitation(rec, settings.insertFormat);
			navigator.clipboard.writeText(citation).then(() => {
				btn.textContent = "Copied!";
				setTimeout(() => {
					btn.textContent = "Copy";
				}, 1500);
			});
		},
		onSelect: (rec, selected) => {
			if (selected) {
				selectedRecs.set(rec.paper_id, rec);
			} else {
				selectedRecs.delete(rec.paper_id);
			}
			renderSelectionBar(container, selectedRecs.size, {
				onInsertSelected: () => {
					const recs = Array.from(selectedRecs.values());
					insertMultiCitation(recs);
				},
				onClearSelected: () => {
					selectedRecs.clear();
					doRenderResults(lastResults);
				},
			});
		},
		onInsertSelected: () => {
			const recs = Array.from(selectedRecs.values());
			insertMultiCitation(recs);
		},
		onClearSelected: () => {
			selectedRecs.clear();
			doRenderResults(lastResults);
		},
	});
}

/** Render the bibliography section. */
function doRenderBibliography(): void {
	if (!tracker) return;
	const t = tracker;
	renderBibliography(t, async (paperId) => {
		await t.remove(paperId);
		doRenderBibliography();
		if (lastResults.length > 0) doRenderResults(lastResults);
	});
}

// --- UI State Helpers ---

function setLoading(on: boolean): void {
	loading = on;
	const btn = getEl("refreshBtn") as HTMLButtonElement;
	btn.disabled = on;
	btn.textContent = on ? "Searching..." : "Get Recommendations";

	if (on) {
		const container = getEl("results");
		container.innerHTML =
			'<div class="mc-state">' +
			'<div class="mc-spinner"></div>' +
			"<div>Searching for citations...</div>" +
			"</div>";
		getEl("timing").style.display = "none";
	}
}

function showError(message: string): void {
	const container = getEl("results");
	container.innerHTML = `<div class="mc-error">${escapeHtml(message)}</div>`;
	getEl("timing").style.display = "none";
}

function showEmpty(): void {
	const container = getEl("results");
	container.innerHTML =
		'<div class="mc-state">' +
		"<div>No results. Try placing your cursor in a paragraph with more context.</div>" +
		"</div>";
	getEl("timing").style.display = "none";
}

function showStatus(message: string): void {
	const bar = getEl("statusBar");
	if (message) {
		bar.textContent = message;
		bar.className = "mc-status-bar visible";
	} else {
		bar.className = "mc-status-bar";
	}
}

function renderTiming(timing: { total_ms: number } | null): void {
	const timingEl = getEl("timing");
	if (timing) {
		timingEl.textContent = `${timing.total_ms} ms`;
		timingEl.style.display = "block";
	} else {
		timingEl.style.display = "none";
	}
}
