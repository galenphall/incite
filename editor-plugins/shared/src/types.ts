/** Settings for inCite editor plugins. */
export interface InCiteSettings {
	apiUrl: string;
	k: number;
	authorBoost: number;
	contextSentences: number;
	citationPatterns: string[];
	insertFormat: string;
	autoDetectEnabled: boolean;
	debounceMs: number;
	showParagraphs: boolean;
}

/** A single evidence snippet from a paper. */
export interface EvidenceSnippet {
	text: string;
	score: number;
	section?: string;
	page?: number;
}

/** A single recommendation from the API. */
export interface Recommendation {
	paper_id: string;
	rank: number;
	score: number;
	title: string;
	authors?: string[];
	year?: number;
	abstract?: string;
	bibtex_key?: string;
	doi?: string;
	journal?: string;
	score_breakdown?: Record<string, number>;
	matched_paragraph?: string;
	matched_paragraphs?: EvidenceSnippet[];
	zotero_uri?: string;
	confidence?: number;
}

/** Timing info from the API. */
export interface TimingInfo {
	total_ms: number;
	embed_query_ms: number;
	vector_search_ms: number;
	bm25_search_ms?: number;
	fusion_ms?: number;
}

/** Full API response from /recommend. */
export interface RecommendResponse {
	query: string;
	recommendations: Recommendation[];
	timing: TimingInfo;
	corpus_size: number;
	method: string;
	embedder: string;
	mode: string;
	timestamp: string;
}

/** API health check response. */
export interface HealthResponse {
	status: string;
	ready: boolean;
	corpus_size?: number;
	mode?: string;  // "paper", "paragraph", "sentence", or "grobid"
}
