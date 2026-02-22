/** API mode: cloud server or local incite serve */
export type ApiMode = "cloud" | "local";

/** Settings for inCite editor plugins. */
export interface InCiteSettings {
	apiMode: ApiMode;
	cloudUrl: string;
	localUrl: string;
	apiToken: string;
	k: number;
	authorBoost: number;
	contextSentences: number;
	citationPatterns: string[];
	insertFormat: string;
	autoDetectEnabled: boolean;
	debounceMs: number;
	showParagraphs: boolean;
	collectionId: string | null;
}

/** Default settings shared across all plugins. */
export const DEFAULT_SETTINGS: InCiteSettings = {
	apiMode: "cloud",
	cloudUrl: "https://inciteref.com",
	localUrl: "http://127.0.0.1:8230",
	apiToken: "",
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
	debounceMs: 500,
	showParagraphs: true,
	collectionId: null,
};

/** Get the active API URL based on the current mode. */
export function getActiveUrl(settings: InCiteSettings): string {
	return settings.apiMode === "cloud" ? settings.cloudUrl : settings.localUrl;
}

/** Configuration for the InCiteClient. */
export interface ClientConfig {
	apiMode: ApiMode;
	cloudUrl: string;
	localUrl: string;
	apiToken: string;
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

/** Metadata for a paper extracted from a web page. */
export interface PaperMetadata {
	title: string;
	authors?: string[];
	year?: number;
	doi?: string;
	abstract?: string;
	journal?: string;
	url?: string;
	arxiv_id?: string;
	pdf_url?: string;
	full_text?: string;
	structured_text?: {
		sections: { heading?: string; paragraphs: string[] }[];
		extraction_method: string;
		source_hostname: string;
	};
}

/** Request to save papers to the user's library. */
export interface SavePapersRequest {
	papers: PaperMetadata[];
	collection_id?: string | null;
	tags?: string[];
	enrich?: boolean;
}

/** A single saved paper result. */
export interface SavedPaperResult {
	canonical_id: string;
	title: string;
	status: "created" | "exists" | "error";
	error?: string;
}

/** Response from the save papers endpoint. */
export interface SavePapersResponse {
	saved: SavedPaperResult[];
	already_existed: Array<{ canonical_id: string; title: string }>;
	errors: Array<{ title: string; error: string }>;
}

/** Result of checking if a paper is in the library. */
export interface LibraryCheckResult {
	doi?: string | null;
	title?: string;
	in_library: boolean;
	canonical_id?: string | null;
	collections?: string[];
	tags?: string[];
}

/** A library collection. */
export interface Collection {
	id: string;
	name: string;
	color?: string | null;
	item_count: number;
}

/** A library tag. */
export interface Tag {
	id: string;
	name: string;
	color?: string | null;
}

/** Cloud library processing status. */
export interface LibraryStatusResponse {
	library_status: string;
	num_papers: number;
	num_chunks: number;
	grobid_fulltext_papers: number;
	grobid_fulltext_chunks: number;
	abstract_only_papers: number;
	job_status?: string;
	stage?: string;
	current?: number;
	total?: number;
	error?: string;
	collections?: Collection[];
}

/** Local server configuration. */
export interface ServerConfigResponse {
	method: string;
	embedder: string;
	mode: string;
	two_stage: boolean;
	available_embedders: string[];
	available_methods: string[];
	available_modes: string[];
	alpha?: number;
}
