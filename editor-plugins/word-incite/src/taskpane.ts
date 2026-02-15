/**
 * inCite Word Add-in Task Pane
 *
 * Office.js task pane that provides citation recommendations from
 * the inCite cloud server or a local `incite serve` instance.
 *
 * Usage:
 *   1. Sign in at inciteref.com and copy your API token, OR run `incite serve` locally
 *   2. Sideload this add-in in Word
 *   3. Click "Get Recommendations" to search based on cursor context
 */

import {
	extractContext,
	formatCitation,
	formatMultiCitation,
	stripCitations,
	CitationTracker,
	exportBibTeX,
	exportRIS,
	exportFormattedText,
} from "@incite/shared";
import type { Recommendation, CitationStorage, TrackedCitation } from "@incite/shared";

// --- Settings ---

type ApiMode = "cloud" | "local";

interface WordSettings {
	apiMode: ApiMode;
	apiToken: string;
	cloudUrl: string;
	localUrl: string;
	k: number;
	authorBoost: number;
	contextSentences: number;
	citationPatterns: string[];
	insertFormat: string;
	autoDetectEnabled: boolean;
	debounceMs: number;
	showParagraphs: boolean;
}

const DEFAULT_SETTINGS: WordSettings = {
	apiMode: "cloud",
	apiToken: "",
	cloudUrl: "https://inciteref.com",
	localUrl: "http://127.0.0.1:8230",
	k: 10,
	authorBoost: 1.0,
	contextSentences: 6,
	citationPatterns: [
		"\\[@[^\\]]*\\]",
		"\\[cite\\]",
		"\\\\cite\\{[^}]*\\}",
	],
	insertFormat: "({first_author}, {year})",
	autoDetectEnabled: false,
	debounceMs: 800,
	showParagraphs: true,
};

let settings: WordSettings = { ...DEFAULT_SETTINGS };

// --- State ---

let loading = false;
let lastResults: Recommendation[] = [];
const selectedRecs: Map<string, Recommendation> = new Map();

// --- Citation Tracking ---

const DOC_KEY = "word-document";

class LocalStorageCitationStorage implements CitationStorage {
	async load(docKey: string): Promise<TrackedCitation[]> {
		try {
			const raw = localStorage.getItem(`incite-citations-${docKey}`);
			return raw ? JSON.parse(raw) : [];
		} catch {
			return [];
		}
	}

	async save(docKey: string, citations: TrackedCitation[]): Promise<void> {
		try {
			localStorage.setItem(
				`incite-citations-${docKey}`,
				JSON.stringify(citations),
			);
		} catch {
			// Ignore storage errors
		}
	}
}

const tracker = new CitationTracker(new LocalStorageCitationStorage(), DOC_KEY);

// --- DOM Helpers ---

function getEl(id: string): HTMLElement {
	return document.getElementById(id)!;
}

function escapeHtml(str: string): string {
	const div = document.createElement("div");
	div.appendChild(document.createTextNode(str));
	return div.innerHTML;
}

// --- API Helpers ---

function getApiUrl(): string {
	return settings.apiMode === "cloud" ? settings.cloudUrl : settings.localUrl;
}

function getApiHeaders(): Record<string, string> {
	const headers: Record<string, string> = {
		"Content-Type": "application/json",
		Accept: "application/json",
	};
	if (settings.apiMode === "cloud" && settings.apiToken) {
		headers["Authorization"] = `Bearer ${settings.apiToken}`;
	}
	return headers;
}

async function apiRecommend(
	query: string,
	k: number,
	authorBoost: number,
): Promise<{ recommendations: Recommendation[]; timing: { total_ms: number } | null }> {
	const baseUrl = getApiUrl();
	const endpoint =
		settings.apiMode === "cloud" ? "/api/v1/recommend" : "/recommend";

	const response = await fetch(`${baseUrl}${endpoint}`, {
		method: "POST",
		headers: getApiHeaders(),
		body: JSON.stringify({
			query,
			k,
			author_boost: authorBoost,
		}),
	});

	if (!response.ok) {
		const text = await response.text().catch(() => "");
		throw new Error(
			`API error ${response.status}: ${text || response.statusText}`,
		);
	}

	return response.json();
}

async function apiHealth(): Promise<{
	ready: boolean;
	corpus_size?: number;
	mode?: string;
}> {
	const baseUrl = getApiUrl();
	const endpoint =
		settings.apiMode === "cloud" ? "/api/v1/health" : "/health";

	const response = await fetch(`${baseUrl}${endpoint}`, {
		method: "GET",
		headers: getApiHeaders(),
	});

	if (!response.ok) {
		throw new Error(`Health check failed: ${response.status}`);
	}

	const data = await response.json();

	// Cloud returns {status: "ready"|"processing", corpus_size, mode}
	// Local returns {ready: boolean, corpus_size, mode}
	if (settings.apiMode === "cloud") {
		return {
			ready: data.status === "ready",
			corpus_size: data.corpus_size,
			mode: data.mode,
		};
	}
	return data;
}

// --- Office.js Entry Point ---

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
		body: { paragraphs: WordParagraphCollection };
	};
	sync: () => Promise<void>;
}

interface WordRange {
	text: string;
	load: (props: string) => void;
	insertText: (text: string, location: string) => void;
}

interface WordParagraphCollection {
	items: Array<{ text: string }>;
	load: (props: string) => void;
}

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

function initApp(): void {
	// Load saved settings from localStorage
	loadSettings();

	// Load citation tracker
	tracker.load().then(() => renderBibliography());

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

	// Insert format
	const formatInput = getEl("insertFormat") as HTMLInputElement;
	formatInput.value = settings.insertFormat;
	formatInput.addEventListener("change", () => {
		settings.insertFormat = formatInput.value;
		saveSettings();
	});

	// Show paragraphs toggle
	const showParaCheckbox = getEl("showParagraphs") as HTMLInputElement;
	showParaCheckbox.checked = settings.showParagraphs;
	showParaCheckbox.addEventListener("change", () => {
		settings.showParagraphs = showParaCheckbox.checked;
		saveSettings();
		if (lastResults.length > 0) {
			renderResults(lastResults);
		}
	});

	// Set initial mode visibility
	updateModeVisibility();

	// Initial health check + polling
	checkHealth();
	setInterval(checkHealth, 30000);
}

// --- Settings Persistence ---

function loadSettings(): void {
	try {
		const saved = localStorage.getItem("incite-settings");
		if (saved) {
			const parsed = JSON.parse(saved);
			// Migrate old apiUrl to localUrl
			if (parsed.apiUrl && !parsed.localUrl) {
				parsed.localUrl = parsed.apiUrl;
				delete parsed.apiUrl;
			}
			settings = { ...DEFAULT_SETTINGS, ...parsed };
		}
	} catch {
		// Use defaults
	}
}

function saveSettings(): void {
	try {
		localStorage.setItem("incite-settings", JSON.stringify(settings));
	} catch {
		// Ignore storage errors
	}
}

function toggleSettings(): void {
	const panel = getEl("settingsPanel");
	panel.classList.toggle("visible");
}

// --- Server Health ---

async function checkHealth(): Promise<void> {
	const statusEl = getEl("status");
	try {
		const health = await apiHealth();
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

// --- Context Extraction from Word ---

async function getContextFromWord(): Promise<{
	text: string;
	cursorSentenceIndex: number;
}> {
	return Word.run(async (context) => {
		const selection = context.document.getSelection();
		selection.load("text");

		const paragraphs = context.document.body.paragraphs;
		paragraphs.load("text");

		await context.sync();

		const allText = paragraphs.items.map((p) => p.text).join("\n");
		const selectionText = selection.text;

		// Find cursor position: locate the selection text within the full document
		let cursorOffset: number;
		if (selectionText.trim().length > 0) {
			// Use the start of the selected text
			const idx = allText.indexOf(selectionText);
			cursorOffset = idx >= 0 ? idx : allText.length / 2;
		} else {
			// No selection -- approximate cursor position from paragraph order.
			// Word doesn't expose a character offset for the cursor, so we
			// find the first empty-text paragraph near the selection as a hint.
			// Fallback: use middle of document.
			cursorOffset = Math.floor(allText.length / 2);

			// Try to find the paragraph that contains/is adjacent to the cursor
			// by checking which paragraph the empty selection belongs to.
			let charPos = 0;
			for (const p of paragraphs.items) {
				charPos += p.text.length + 1;
				if (p.text.length === 0) {
					// Empty paragraphs are common cursor positions
					cursorOffset = charPos;
					break;
				}
			}
		}

		const extracted = extractContext(
			allText,
			cursorOffset,
			settings.contextSentences,
		);
		return {
			text: extracted.text,
			cursorSentenceIndex: extracted.cursorSentenceIndex,
		};
	});
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
		const response = await apiRecommend(
			cleaned,
			settings.k,
			settings.authorBoost,
		);

		lastResults = response.recommendations || [];

		if (lastResults.length === 0) {
			showEmpty();
		} else {
			renderResults(lastResults);
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
		await tracker.track([rec]);
		renderBibliography();
		if (lastResults.length > 0) renderResults(lastResults);
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
		await tracker.track(recs);
		selectedRecs.clear();
		renderBibliography();
		if (lastResults.length > 0) renderResults(lastResults);
		showStatus(`Inserted ${recs.length} citations: ${citation}`);
		setTimeout(() => showStatus(""), 2000);
	} catch (err) {
		showError(`Could not insert citations: ${err}`);
	}
}

// --- Rendering ---

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

function renderSelectionBar(): void {
	const existing = document.getElementById("selectionBar");
	if (existing) existing.remove();

	if (selectedRecs.size === 0) return;

	const bar = document.createElement("div");
	bar.id = "selectionBar";
	bar.className = "mc-selection-bar";

	const label = document.createElement("span");
	label.textContent = `${selectedRecs.size} selected`;
	bar.appendChild(label);

	const insertBtn = document.createElement("button");
	insertBtn.className = "mc-insert-btn";
	insertBtn.textContent = "Insert Selected";
	insertBtn.addEventListener("click", () => {
		const recs = Array.from(selectedRecs.values());
		insertMultiCitation(recs);
	});
	bar.appendChild(insertBtn);

	const clearBtn = document.createElement("button");
	clearBtn.className = "mc-copy-btn";
	clearBtn.textContent = "Clear";
	clearBtn.addEventListener("click", () => {
		selectedRecs.clear();
		renderResults(lastResults);
	});
	bar.appendChild(clearBtn);

	const container = getEl("results");
	container.insertBefore(bar, container.firstChild);
}

function renderResults(results: Recommendation[]): void {
	const container = getEl("results");
	container.innerHTML = "";

	// Selection bar at the top
	if (selectedRecs.size > 0) {
		renderSelectionBar();
	}

	for (const rec of results) {
		const card = document.createElement("div");
		card.className = "mc-result";

		// Header: checkbox + rank + confidence + cited badge
		const header = document.createElement("div");
		header.className = "mc-result-header";

		const checkbox = document.createElement("input");
		checkbox.type = "checkbox";
		checkbox.className = "mc-select-checkbox";
		checkbox.checked = selectedRecs.has(rec.paper_id);
		checkbox.title = "Select for multi-citation";
		checkbox.addEventListener("change", () => {
			if (checkbox.checked) {
				selectedRecs.set(rec.paper_id, rec);
			} else {
				selectedRecs.delete(rec.paper_id);
			}
			renderSelectionBar();
		});
		header.appendChild(checkbox);

		const rank = document.createElement("span");
		rank.className = "mc-rank";
		rank.textContent = `${rec.rank}.`;
		header.appendChild(rank);

		const conf = rec.confidence || 0;
		const confBadge = document.createElement("span");
		confBadge.className =
			"mc-confidence " +
			(conf >= 0.55
				? "mc-confidence-high"
				: conf >= 0.35
					? "mc-confidence-mid"
					: "mc-confidence-low");
		confBadge.textContent = conf.toFixed(2);
		confBadge.title = `Confidence: ${conf.toFixed(3)}`;
		header.appendChild(confBadge);

		if (tracker.isTracked(rec.paper_id)) {
			const citedBadge = document.createElement("span");
			citedBadge.className = "mc-cited-badge";
			citedBadge.textContent = "Cited";
			header.appendChild(citedBadge);
		}

		card.appendChild(header);

		// Title
		const title = document.createElement("div");
		title.className = "mc-title";
		title.textContent = rec.title || "Untitled";
		card.appendChild(title);

		// Authors + year
		if (rec.authors || rec.year) {
			const meta = document.createElement("div");
			meta.className = "mc-meta";
			const parts: string[] = [];
			if (rec.authors && rec.authors.length > 0) {
				let names = rec.authors.slice(0, 3).join(", ");
				if (rec.authors.length > 3) names += " et al.";
				parts.push(names);
			}
			if (rec.year) parts.push(`(${rec.year})`);
			meta.textContent = parts.join(" ");
			card.appendChild(meta);
		}

		// Matched paragraph evidence
		if (settings.showParagraphs && rec.matched_paragraph) {
			const para = document.createElement("div");
			para.className = "mc-paragraph";
			renderHighlightedText(rec.matched_paragraph, para, 350);
			card.appendChild(para);
		}

		// Actions row
		const actions = document.createElement("div");
		actions.className = "mc-actions";

		const insertBtn = document.createElement("button");
		insertBtn.className = "mc-insert-btn";
		insertBtn.textContent = "+ Insert";
		insertBtn.addEventListener("click", () => insertCitation(rec));
		actions.appendChild(insertBtn);

		const copyBtn = document.createElement("button");
		copyBtn.className = "mc-copy-btn";
		copyBtn.textContent = "Copy";
		copyBtn.addEventListener("click", () => {
			const citation = formatCitation(rec, settings.insertFormat);
			navigator.clipboard.writeText(citation).then(() => {
				copyBtn.textContent = "Copied!";
				setTimeout(() => {
					copyBtn.textContent = "Copy";
				}, 1500);
			});
		});
		actions.appendChild(copyBtn);

		card.appendChild(actions);
		container.appendChild(card);
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

/**
 * Render text with **bold** markers converted to <strong> elements.
 * Truncates around the highlighted portion. Matches Google Docs sidebar.
 */
function renderHighlightedText(
	text: string,
	container: HTMLElement,
	maxLength: number,
): void {
	const startBold = text.indexOf("**");
	const endBold = text.indexOf("**", startBold + 2);

	if (startBold === -1 || endBold === -1) {
		const truncated = text.length > maxLength;
		container.textContent = truncated
			? text.slice(0, maxLength) + "..."
			: text;
		return;
	}

	const highlightEnd = endBold + 2;
	const highlightLength = highlightEnd - startBold;
	const available = maxLength - highlightLength;
	const ctxBefore = Math.floor(available * 0.3);
	const ctxAfter = available - ctxBefore;

	const sliceStart = Math.max(0, startBold - ctxBefore);
	const sliceEnd = Math.min(text.length, highlightEnd + ctxAfter);
	let display = text.slice(sliceStart, sliceEnd);

	if (sliceStart > 0) {
		const spaceIdx = display.indexOf(" ");
		if (spaceIdx > 0 && spaceIdx < 20) {
			display = display.slice(spaceIdx + 1);
		}
		display = "..." + display;
	}
	if (sliceEnd < text.length) {
		display = display + "...";
	}

	const parts = display.split(/(\*\*.+?\*\*)/g);
	for (const part of parts) {
		if (
			part.indexOf("**") === 0 &&
			part.lastIndexOf("**") === part.length - 2
		) {
			const strong = document.createElement("strong");
			strong.textContent = part.slice(2, -2);
			container.appendChild(strong);
		} else if (part) {
			container.appendChild(document.createTextNode(part));
		}
	}
}

// --- Bibliography ---

function renderBibliography(): void {
	let section = document.getElementById("bibliographySection");
	if (!section) {
		section = document.createElement("div");
		section.id = "bibliographySection";
		// Append after the results area
		const results = getEl("results");
		results.parentElement!.appendChild(section);
	}
	section.innerHTML = "";

	const citations = tracker.getAll();
	if (citations.length === 0) {
		section.style.display = "none";
		return;
	}
	section.style.display = "";

	// Collapsible header
	const header = document.createElement("div");
	header.className = "mc-bib-header";
	header.textContent = `Bibliography (${citations.length} citation${citations.length !== 1 ? "s" : ""})`;
	let expanded = true;
	const body = document.createElement("div");

	header.addEventListener("click", () => {
		expanded = !expanded;
		body.style.display = expanded ? "" : "none";
		header.classList.toggle("collapsed", !expanded);
	});
	section.appendChild(header);

	// Export buttons
	const exportRow = document.createElement("div");
	exportRow.className = "mc-bib-export";

	const bibtexBtn = document.createElement("button");
	bibtexBtn.className = "mc-copy-btn";
	bibtexBtn.textContent = "BibTeX";
	bibtexBtn.addEventListener("click", () => {
		navigator.clipboard.writeText(exportBibTeX(citations)).then(() => {
			bibtexBtn.textContent = "Copied!";
			setTimeout(() => { bibtexBtn.textContent = "BibTeX"; }, 1500);
		});
	});
	exportRow.appendChild(bibtexBtn);

	const risBtn = document.createElement("button");
	risBtn.className = "mc-copy-btn";
	risBtn.textContent = "RIS";
	risBtn.addEventListener("click", () => {
		navigator.clipboard.writeText(exportRIS(citations)).then(() => {
			risBtn.textContent = "Copied!";
			setTimeout(() => { risBtn.textContent = "RIS"; }, 1500);
		});
	});
	exportRow.appendChild(risBtn);

	const apaBtn = document.createElement("button");
	apaBtn.className = "mc-copy-btn";
	apaBtn.textContent = "Copy APA";
	apaBtn.addEventListener("click", () => {
		navigator.clipboard.writeText(exportFormattedText(citations)).then(() => {
			apaBtn.textContent = "Copied!";
			setTimeout(() => { apaBtn.textContent = "Copy APA"; }, 1500);
		});
	});
	exportRow.appendChild(apaBtn);

	body.appendChild(exportRow);

	// Citation list
	for (const cite of citations) {
		const row = document.createElement("div");
		row.className = "mc-bib-entry";

		const info = document.createElement("span");
		info.className = "mc-bib-info";
		const authorStr =
			cite.authors.length > 0
				? cite.authors[0].split(",")[0].split(" ").pop() ?? ""
				: "";
		const yearStr = cite.year != null ? ` (${cite.year})` : "";
		info.textContent = `${authorStr}${yearStr} - ${cite.title}`;
		row.appendChild(info);

		const removeBtn = document.createElement("button");
		removeBtn.className = "mc-remove-btn";
		removeBtn.textContent = "x";
		removeBtn.title = "Remove from bibliography";
		removeBtn.addEventListener("click", async () => {
			await tracker.remove(cite.paper_id);
			renderBibliography();
			if (lastResults.length > 0) renderResults(lastResults);
		});
		row.appendChild(removeBtn);

		body.appendChild(row);
	}

	section.appendChild(body);
}
