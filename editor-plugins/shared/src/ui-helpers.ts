import type { Recommendation, EvidenceSnippet } from "./types";
import type { TrackedCitation } from "./citation-tracker";

/**
 * CSS class name mapping — each plugin passes its own class names.
 * All functions fall back to DEFAULT_CLASS_MAP.
 */
export interface UIClassMap {
	resultCard: string;
	resultHeader: string;
	resultHeaderLeft: string;
	selectCheckbox: string;
	rankBadge: string;
	citedBadge: string;
	confidenceBadge: string;
	confidenceHigh: string;
	confidenceMid: string;
	confidenceLow: string;
	resultTitle: string;
	resultMeta: string;
	evidence: string;
	evidenceSecondary: string;
	evidenceScore: string;
	resultAbstract: string;
	resultActions: string;
	insertBtn: string;
	copyBtn: string;
	bibSection: string;
	bibToggle: string;
	bibContent: string;
	bibExportBar: string;
	bibList: string;
	bibItem: string;
	bibItemText: string;
	bibItemAuthors: string;
	bibItemTitle: string;
	bibRemove: string;
}

export const DEFAULT_CLASS_MAP: UIClassMap = {
	resultCard: "incite-result-card",
	resultHeader: "incite-result-header",
	resultHeaderLeft: "incite-result-header-left",
	selectCheckbox: "incite-select-checkbox",
	rankBadge: "incite-rank-badge",
	citedBadge: "incite-cited-badge",
	confidenceBadge: "incite-confidence-badge",
	confidenceHigh: "incite-confidence-high",
	confidenceMid: "incite-confidence-mid",
	confidenceLow: "incite-confidence-low",
	resultTitle: "incite-result-title",
	resultMeta: "incite-result-meta",
	evidence: "incite-evidence",
	evidenceSecondary: "incite-evidence-secondary",
	evidenceScore: "incite-evidence-score",
	resultAbstract: "incite-result-abstract",
	resultActions: "incite-result-actions",
	insertBtn: "incite-insert-btn",
	copyBtn: "incite-copy-btn",
	bibSection: "incite-bibliography-section",
	bibToggle: "incite-bib-toggle",
	bibContent: "incite-bib-content",
	bibExportBar: "incite-bib-export-bar",
	bibList: "incite-bib-list",
	bibItem: "incite-bib-item",
	bibItemText: "incite-bib-item-text",
	bibItemAuthors: "incite-bib-item-authors",
	bibItemTitle: "incite-bib-item-title",
	bibRemove: "incite-bib-remove",
};

/** Options for rendering result cards. */
export interface RenderResultOptions {
	showParagraphs: boolean;
	showAbstracts?: boolean;
	isCited?: boolean;
}

/** Escape HTML special characters (pure string, no DOM required). */
export function escapeHtml(text: string): string {
	return text
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;")
		.replace(/'/g, "&#39;");
}

/** Escape text for use in HTML attributes. */
export function escapeAttr(text: string): string {
	return text
		.replace(/&/g, "&amp;")
		.replace(/'/g, "&#39;")
		.replace(/"/g, "&quot;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;");
}

/** Determine confidence level from a score. */
export function confidenceLevel(score: number): "high" | "mid" | "low" {
	if (score >= 0.55) return "high";
	if (score >= 0.35) return "mid";
	return "low";
}

/**
 * Render text with **bold** markers converted to <strong>, with smart
 * truncation that preserves highlight context.
 */
export function renderHighlightedTextHTML(text: string, maxLength: number): string {
	const startBold = text.indexOf("**");
	const endBold = text.indexOf("**", startBold + 2);

	if (startBold === -1 || endBold === -1) {
		const truncated = text.length > maxLength;
		return escapeHtml(truncated ? text.slice(0, maxLength) + "..." : text);
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

	// Split on **text** patterns, render bold portions as <strong>
	const parts = display.split(/(\*\*.+?\*\*)/g);
	let html = "";
	for (const part of parts) {
		if (part.startsWith("**") && part.endsWith("**")) {
			html += `<strong>${escapeHtml(part.slice(2, -2))}</strong>`;
		} else if (part) {
			html += escapeHtml(part);
		}
	}
	return html;
}

/** Render evidence snippets as HTML. */
export function renderEvidenceHTML(
	rec: Recommendation,
	options: RenderResultOptions,
	cm: Partial<UIClassMap> = {}
): string {
	const c = { ...DEFAULT_CLASS_MAP, ...cm };
	if (!options.showParagraphs) return "";

	let html = "";
	if (rec.matched_paragraphs && rec.matched_paragraphs.length > 0) {
		for (let i = 0; i < rec.matched_paragraphs.length; i++) {
			const snippet = rec.matched_paragraphs[i];
			const cls = i === 0 ? c.evidence : `${c.evidence} ${c.evidenceSecondary}`;
			const badge = snippet.score != null
				? `<span class="${c.evidenceScore}">${Math.round(snippet.score * 100)}%</span> `
				: "";
			html += `<div class="${cls}">${badge}${renderHighlightedTextHTML(snippet.text, 300)}</div>`;
		}
	} else if (rec.matched_paragraph) {
		html += `<div class="${c.evidence}">${renderHighlightedTextHTML(rec.matched_paragraph, 300)}</div>`;
	}
	return html;
}

/** Render a full result card as HTML string. */
export function renderResultCardHTML(
	rec: Recommendation,
	options: RenderResultOptions,
	cm: Partial<UIClassMap> = {}
): string {
	const c = { ...DEFAULT_CLASS_MAP, ...cm };
	const confidence = rec.confidence ?? rec.score;
	const level = confidenceLevel(confidence);
	const confClass = level === "high" ? c.confidenceHigh
		: level === "mid" ? c.confidenceMid
		: c.confidenceLow;
	const confLabel = `${Math.round(confidence * 100)}%`;

	let html = `<div class="${c.resultCard}">`;

	// Header: checkbox + rank + badges
	html += `<div class="${c.resultHeader}">`;
	html += `<div class="${c.resultHeaderLeft}">`;
	html += `<input type="checkbox" class="${c.selectCheckbox}" data-action="select" data-rec='${escapeAttr(JSON.stringify(rec))}' />`;
	html += `<span class="${c.rankBadge}">#${rec.rank}</span>`;
	if (options.isCited) {
		html += `<span class="${c.citedBadge}">Cited</span>`;
	}
	html += `</div>`;
	html += `<span class="${c.confidenceBadge} ${confClass}">${confLabel}</span>`;
	html += `</div>`;

	// Title
	html += `<div class="${c.resultTitle}">${escapeHtml(rec.title)}</div>`;

	// Authors + year
	const meta: string[] = [];
	if (rec.authors && rec.authors.length > 0) {
		const names = rec.authors.slice(0, 3).join(", ");
		meta.push(rec.authors.length > 3 ? names + " et al." : names);
	}
	if (rec.year) meta.push(`(${rec.year})`);
	if (meta.length > 0) {
		html += `<div class="${c.resultMeta}">${escapeHtml(meta.join(" "))}</div>`;
	}

	// Evidence
	html += renderEvidenceHTML(rec, options, cm);

	// Abstract
	if (options.showAbstracts && rec.abstract) {
		html += `<div class="${c.resultAbstract}">${escapeHtml(rec.abstract)}</div>`;
	}

	// Actions
	const recJson = escapeAttr(JSON.stringify(rec));
	const bibtexKey = rec.bibtex_key ?? rec.paper_id;
	html += `<div class="${c.resultActions}">`;
	html += `<button class="${c.insertBtn}" data-action="insert" data-rec='${recJson}'>Insert</button>`;
	html += `<button class="${c.copyBtn}" data-action="copy" data-copy="${escapeAttr(bibtexKey)}">Copy Key</button>`;
	html += `</div>`;

	html += `</div>`;
	return html;
}

/** Render the bibliography section as HTML string. */
export function renderBibliographyHTML(
	citations: TrackedCitation[],
	cm: Partial<UIClassMap> = {}
): string {
	const c = { ...DEFAULT_CLASS_MAP, ...cm };

	let html = `<div class="${c.bibSection}">`;
	html += `<button class="${c.bibToggle}">`;
	html += `Bibliography (${citations.length} citation${citations.length !== 1 ? "s" : ""})`;
	html += `<span class="toggle-arrow">&#9662;</span>`;
	html += `</button>`;

	html += `<div class="${c.bibContent}" style="display:none;">`;

	// Export buttons
	html += `<div class="${c.bibExportBar}">`;
	html += `<button class="${c.copyBtn}" data-action="bib-export" data-format="bibtex">BibTeX</button>`;
	html += `<button class="${c.copyBtn}" data-action="bib-export" data-format="ris">RIS</button>`;
	html += `<button class="${c.copyBtn}" data-action="bib-export" data-format="apa">APA</button>`;
	html += `</div>`;

	// Citation list
	html += `<div class="${c.bibList}">`;
	for (const cite of citations) {
		const authorStr = cite.authors.length > 0
			? cite.authors.length > 2
				? cite.authors[0].split(" ").pop() + " et al."
				: cite.authors.map((a: string) => a.split(" ").pop()).join(" & ")
			: "";
		const yearStr = cite.year != null ? ` (${cite.year})` : "";

		html += `<div class="${c.bibItem}" data-paper-id="${escapeAttr(cite.paper_id)}">`;
		html += `<div class="${c.bibItemText}">`;
		html += `<span class="${c.bibItemAuthors}">${escapeHtml(authorStr + yearStr)}</span> `;
		html += `<span class="${c.bibItemTitle}">${escapeHtml(cite.title)}</span>`;
		html += `</div>`;
		html += `<button class="${c.bibRemove}" data-action="bib-remove" data-paper-id="${escapeAttr(cite.paper_id)}" title="Remove">&times;</button>`;
		html += `</div>`;
	}
	html += `</div>`;

	html += `</div>`; // bib-content
	html += `</div>`; // bibliography-section
	return html;
}
