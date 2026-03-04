/**
 * CSL citation formatting client helpers.
 *
 * All plugins call the same server-side endpoints — works identically
 * for local (`localhost:8230`) and cloud (`inciteref.com`) deployments.
 *
 * The actual API calls live on InCiteClient (api-client.ts). This module
 * provides typed wrappers and the style-picker UI helper.
 */
import type { InCiteClient } from "./api-client";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** A CSL style entry from the server manifest. */
export interface CSLStyle {
	id: string;
	title: string;
	category: string; // "author-date" | "numeric" | "note" | "label" | "author"
	file: string;
}

/** Response from POST /api/csl/format in bibliography mode. */
export interface CSLBibliographyResponse {
	bibliography: {
		html?: string;
		text?: string;
		style: string;
		count: number;
	};
	style: string;
	count: number;
}

/** A single rendered citation cluster. */
export interface CSLCitationCluster {
	cluster_index: number;
	html?: string;
	text?: string;
}

/** Response from POST /api/csl/format in citation mode. */
export interface CSLCitationResponse {
	citations: CSLCitationCluster[];
	bibliography: {
		html?: string;
		text?: string;
	};
	style: string;
}

/** Input cluster for citation mode. */
export interface CitationClusterInput {
	citation_ids: string[];
	position: number;
}

/** Response from GET /api/csl/styles. */
export interface CSLStylesResponse {
	styles: CSLStyle[];
}

/** Response from GET /api/csl/styles/{id}. */
export interface CSLStyleDetailResponse extends CSLStyle {
	preview: {
		html?: string;
		text?: string;
	};
}

// ---------------------------------------------------------------------------
// API functions (thin typed wrappers around InCiteClient methods)
// ---------------------------------------------------------------------------

/**
 * Format a bibliography for the given paper IDs.
 */
export async function formatBibliographyCSL(
	client: InCiteClient,
	paperIds: string[],
	style: string = "apa",
	outputFormat: string = "html",
): Promise<CSLBibliographyResponse> {
	const resp = await client.formatBibliography(paperIds, style, outputFormat);
	return resp as CSLBibliographyResponse;
}

/**
 * Format in-text citations with document-level state (Phase 2).
 */
export async function formatCitationCSL(
	client: InCiteClient,
	paperIds: string[],
	clusters: CitationClusterInput[],
	style: string = "apa",
	outputFormat: string = "html",
): Promise<CSLCitationResponse> {
	const resp = await client.formatCitations(paperIds, clusters, style, outputFormat);
	return resp as CSLCitationResponse;
}

/**
 * List available CSL styles from the server.
 */
export async function getAvailableStyles(
	client: InCiteClient,
): Promise<CSLStyle[]> {
	const resp = await client.getCSLStyles() as CSLStylesResponse;
	return resp.styles;
}

/**
 * Get metadata and preview for a specific style.
 */
export async function getStyleDetail(
	client: InCiteClient,
	styleId: string,
): Promise<CSLStyleDetailResponse> {
	const resp = await client.getCSLStyleDetail(styleId);
	return resp as CSLStyleDetailResponse;
}

/**
 * Install a style from the CSL GitHub repository.
 */
export async function installStyle(
	client: InCiteClient,
	styleId: string,
): Promise<{ status: string; style_id: string }> {
	const resp = await client.installCSLStyle(styleId);
	return resp as { status: string; style_id: string };
}

// ---------------------------------------------------------------------------
// UI helper
// ---------------------------------------------------------------------------

/**
 * Render a style picker dropdown as an HTML string.
 *
 * @param styles - Available styles from `getAvailableStyles()`.
 * @param currentStyle - Currently selected style ID.
 * @param selectId - HTML id for the <select> element.
 */
export function renderStylePickerHTML(
	styles: CSLStyle[],
	currentStyle: string,
	selectId: string = "incite-csl-style",
): string {
	// Group by category
	const groups: Record<string, CSLStyle[]> = {};
	for (const style of styles) {
		const cat = style.category || "other";
		if (!groups[cat]) groups[cat] = [];
		groups[cat].push(style);
	}

	// Category display names
	const categoryNames: Record<string, string> = {
		"author-date": "Author-Date",
		numeric: "Numeric",
		note: "Note",
		label: "Label",
		author: "Author",
		other: "Other",
	};

	const categoryOrder = ["author-date", "numeric", "note", "author", "label", "other"];

	let html = `<select id="${selectId}" name="csl-style">`;
	for (const cat of categoryOrder) {
		const items = groups[cat];
		if (!items || items.length === 0) continue;
		const label = categoryNames[cat] || cat;
		html += `<optgroup label="${label}">`;
		for (const style of items.sort((a, b) => a.title.localeCompare(b.title))) {
			const selected = style.id === currentStyle ? " selected" : "";
			html += `<option value="${style.id}"${selected}>${style.title}</option>`;
		}
		html += "</optgroup>";
	}
	html += "</select>";
	return html;
}
