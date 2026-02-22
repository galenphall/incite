export type {
	ApiMode,
	InCiteSettings,
	ClientConfig,
	Recommendation,
	EvidenceSnippet,
	TimingInfo,
	RecommendResponse,
	HealthResponse,
	PaperMetadata,
	SavePapersRequest,
	SavedPaperResult,
	SavePapersResponse,
	LibraryCheckResult,
	Collection,
	Tag,
	LibraryStatusResponse,
	ServerConfigResponse,
} from "./types";
export { DEFAULT_SETTINGS, getActiveUrl } from "./types";

export type { HttpTransport } from "./api-client";
export { InCiteClient, FetchTransport } from "./api-client";
export { extractContext, splitSentences, stripCitations } from "./context-extractor";
export type { ExtractedContext } from "./context-extractor";
export { CitationWatcherCore } from "./citation-watcher-core";
export { formatCitation, formatMultiCitation, detectCitationStyle } from "./format";
export type { CitationStyle } from "./format";
export { CitationTracker, recommendationToTracked } from "./citation-tracker";
export type { TrackedCitation, CitationStorage } from "./citation-tracker";
export { exportBibTeX, exportRIS, exportFormattedText, escapeLaTeX } from "./bibliography";

export type { UIClassMap, RenderResultOptions } from "./ui-helpers";
export {
	DEFAULT_CLASS_MAP,
	escapeHtml,
	escapeAttr,
	confidenceLevel,
	confidenceLabel,
	renderHighlightedTextHTML,
	renderEvidenceHTML,
	renderResultCardHTML,
	renderBibliographyHTML,
} from "./ui-helpers";
