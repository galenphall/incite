// Re-export shared types — single source of truth in @incite/shared
export type {
	ApiMode,
	InCiteSettings,
	ClientConfig,
	Recommendation,
	TimingInfo,
	RecommendResponse,
	HealthResponse,
} from "@incite/shared";

export { DEFAULT_SETTINGS as SHARED_DEFAULTS, getActiveUrl } from "@incite/shared";
export { formatCitation, formatMultiCitation } from "@incite/shared";
export { CitationTracker, recommendationToTracked } from "@incite/shared";
export type { TrackedCitation, CitationStorage } from "@incite/shared";
export { exportBibTeX, exportRIS, exportFormattedText } from "@incite/shared";
export { confidenceLevel } from "@incite/shared";

import { DEFAULT_SETTINGS as SHARED_DEFAULTS } from "@incite/shared";
import type { InCiteSettings } from "@incite/shared";

/** Default settings for the Obsidian plugin. */
export const DEFAULT_SETTINGS: InCiteSettings = {
	...SHARED_DEFAULTS,
	apiMode: "local",  // Obsidian defaults to local
	insertFormat: "[({first_author}, {year})]({zotero_uri})",
	autoDetectEnabled: true,
	debounceMs: 500,
};
