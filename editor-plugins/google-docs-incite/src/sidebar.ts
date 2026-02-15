/**
 * inCite Google Docs Sidebar
 *
 * Runs as a standalone web page loaded in an iframe sidebar.
 * Communicates with the inCite API server (localhost:8230) via fetch.
 *
 * Usage:
 *   1. Run `incite serve` on your local machine
 *   2. Open this sidebar in Google Docs (via Apps Script add-on or manual iframe)
 *   3. Select text in Google Docs, click "Get Recommendations"
 *
 * The Google Docs API (Apps Script) provides document text to this sidebar
 * via google.script.run calls. This file handles the UI and API communication.
 */

import {
	InCiteClient,
	extractContext,
	formatCitation,
	stripCitations,
} from "@incite/shared";
import type { Recommendation, InCiteSettings } from "@incite/shared";

// --- Settings ---

const SETTINGS: InCiteSettings = {
	apiUrl: "http://127.0.0.1:8230",
	k: 10,
	authorBoost: 1.0,
	contextSentences: 6,
	citationPatterns: [
		"\\[@[^\\]]*\\]",
		"\\[cite\\]",
		"\\\\cite\\{[^}]*\\}",
	],
	insertFormat: "({first_author}, {year})",
	autoDetectEnabled: false,  // Manual-only for Google Docs
	debounceMs: 800,
	showParagraphs: true,
};

const client = new InCiteClient(SETTINGS.apiUrl);

// --- DOM Helpers ---

function getEl(id: string): HTMLElement {
	return document.getElementById(id)!;
}

// --- Server Health ---

async function checkHealth(): Promise<void> {
	const statusEl = getEl("status");
	try {
		const health = await client.health();
		if (health.ready) {
			statusEl.textContent = `Connected (${health.corpus_size} papers, ${health.mode} mode)`;
			statusEl.className = "status connected";
		} else {
			statusEl.textContent = "Server loading...";
			statusEl.className = "status loading";
		}
	} catch {
		statusEl.textContent = "Server offline — run 'incite serve'";
		statusEl.className = "status offline";
	}
}

// --- Recommendations ---

async function getRecommendations(): Promise<void> {
	const queryEl = getEl("query") as HTMLTextAreaElement;
	const resultsEl = getEl("results");
	const query = queryEl.value.trim();

	if (!query) {
		resultsEl.innerHTML = "<p class='hint'>Paste citation context above and click Get Recommendations.</p>";
		return;
	}

	resultsEl.innerHTML = "<p class='loading'>Searching...</p>";

	try {
		const cleaned = stripCitations(query);
		const response = await client.recommend(
			cleaned,
			SETTINGS.k,
			SETTINGS.authorBoost,
		);

		if (response.recommendations.length === 0) {
			resultsEl.innerHTML = "<p class='hint'>No recommendations found.</p>";
			return;
		}

		resultsEl.innerHTML = "";
		for (const rec of response.recommendations) {
			resultsEl.appendChild(renderRecommendation(rec));
		}
	} catch (err) {
		resultsEl.innerHTML = `<p class='error'>Error: ${err}</p>`;
	}
}

function renderRecommendation(rec: Recommendation): HTMLElement {
	const el = document.createElement("div");
	el.className = "recommendation";

	const title = document.createElement("div");
	title.className = "rec-title";
	title.textContent = `${rec.rank}. ${rec.title}`;
	el.appendChild(title);

	const meta = document.createElement("div");
	meta.className = "rec-meta";
	const authors = rec.authors?.join(", ") ?? "";
	const year = rec.year ?? "";
	meta.textContent = `${authors} (${year}) — score: ${rec.score.toFixed(3)}`;
	el.appendChild(meta);

	if (rec.matched_paragraph && SETTINGS.showParagraphs) {
		const para = document.createElement("div");
		para.className = "rec-paragraph";
		para.textContent = rec.matched_paragraph.substring(0, 300);
		el.appendChild(para);
	}

	// Copy citation button
	const copyBtn = document.createElement("button");
	copyBtn.className = "copy-btn";
	copyBtn.textContent = "Copy citation";
	copyBtn.onclick = () => {
		const citation = formatCitation(rec, SETTINGS.insertFormat);
		navigator.clipboard.writeText(citation);
		copyBtn.textContent = "Copied!";
		setTimeout(() => { copyBtn.textContent = "Copy citation"; }, 1500);
	};
	el.appendChild(copyBtn);

	return el;
}

// --- Init ---

function init(): void {
	// Wire up button
	getEl("recommend-btn").addEventListener("click", getRecommendations);

	// Settings URL input
	const urlInput = getEl("api-url") as HTMLInputElement;
	urlInput.value = SETTINGS.apiUrl;
	urlInput.addEventListener("change", () => {
		SETTINGS.apiUrl = urlInput.value;
		client.setBaseUrl(SETTINGS.apiUrl);
		checkHealth();
	});

	// Initial health check
	checkHealth();

	// Poll health every 30s
	setInterval(checkHealth, 30000);
}

// Run on DOM ready
if (document.readyState === "loading") {
	document.addEventListener("DOMContentLoaded", init);
} else {
	init();
}
