export type {
	InCiteSettings,
	Recommendation,
	EvidenceSnippet,
	TimingInfo,
	RecommendResponse,
	HealthResponse,
} from "./types";

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
