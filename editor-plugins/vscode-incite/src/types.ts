// Re-export shared types — single source of truth in @incite/shared
export type {
	InCiteSettings,
	Recommendation,
	TimingInfo,
	RecommendResponse,
	HealthResponse,
} from "@incite/shared";

export { formatCitation } from "@incite/shared";

/** Default settings for the VS Code extension. */
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
	insertFormat: "\\cite{${bibtex_key}}",
	autoDetectEnabled: true,
	debounceMs: 800,
	showParagraphs: true,
};
