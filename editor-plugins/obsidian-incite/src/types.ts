// Re-export shared types — single source of truth in @incite/shared
export type {
	InCiteSettings,
	Recommendation,
	TimingInfo,
	RecommendResponse,
	HealthResponse,
} from "@incite/shared";

export { formatCitation, formatMultiCitation } from "@incite/shared";
export { CitationTracker, recommendationToTracked } from "@incite/shared";
export type { TrackedCitation, CitationStorage } from "@incite/shared";
export { exportBibTeX, exportRIS, exportFormattedText } from "@incite/shared";

/** Default settings for the Obsidian plugin. */
export const DEFAULT_SETTINGS: import("@incite/shared").InCiteSettings = {
	apiUrl: "http://127.0.0.1:8230",
	k: 10,
	authorBoost: 1.0,
	contextSentences: 6,
	citationPatterns: [
		"\\[@[^\\]]*\\]",     // [@key] or [@?] — Pandoc/Zotero
		"\\[cite\\]",         // [cite] — placeholder
		"\\\\cite\\{[^}]*\\}" // \cite{key} — LaTeX
	],
	insertFormat: "[({first_author}, {year})]({zotero_uri})",
	autoDetectEnabled: true,
	debounceMs: 500,
	showParagraphs: true,
};
