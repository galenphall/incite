import type {
	HealthResponse,
	RecommendResponse,
	SavePapersRequest,
	SavePapersResponse,
	LibraryCheckResult,
	Collection,
	Tag,
	ClientConfig,
	LibraryStatusResponse,
	ServerConfigResponse,
} from "./types";
import { getActiveUrl } from "./types";

/**
 * HTTP transport interface — each editor provides its own implementation.
 *
 * Obsidian uses its built-in requestUrl(); VS Code and Google Docs use fetch().
 * Optional `headers` param allows cloud auth without bypassing the transport.
 */
export interface HttpTransport {
	get(url: string, headers?: Record<string, string>): Promise<unknown>;
	post(url: string, body: unknown, headers?: Record<string, string>): Promise<unknown>;
}

/** fetch()-based transport for environments with native fetch (VS Code, Google Docs, Word). */
export class FetchTransport implements HttpTransport {
	async get(url: string, headers?: Record<string, string>): Promise<unknown> {
		const response = await fetch(url, {
			method: "GET",
			headers: {
				"Accept": "application/json",
				...headers,
			},
		});
		if (!response.ok) {
			throw new Error(`HTTP ${response.status}: ${response.statusText}`);
		}
		return response.json();
	}

	async post(url: string, body: unknown, headers?: Record<string, string>): Promise<unknown> {
		const response = await fetch(url, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				"Accept": "application/json",
				...headers,
			},
			body: JSON.stringify(body),
		});
		if (!response.ok) {
			throw new Error(`HTTP ${response.status}: ${response.statusText}`);
		}
		return response.json();
	}
}

/** HTTP client for the inCite API server. */
export class InCiteClient {
	private config: ClientConfig;
	private transport: HttpTransport;

	/**
	 * Create an InCiteClient.
	 *
	 * Accepts either a `ClientConfig` object (cloud-aware) or a plain `baseUrl`
	 * string for backward compatibility.
	 */
	constructor(configOrBaseUrl: ClientConfig | string, transport?: HttpTransport) {
		if (typeof configOrBaseUrl === "string") {
			// Backward-compatible: treat as local-only with the given URL
			this.config = {
				apiMode: "local",
				cloudUrl: "https://inciteref.com",
				localUrl: configOrBaseUrl,
				apiToken: "",
			};
		} else {
			this.config = { ...configOrBaseUrl };
		}
		this.transport = transport ?? new FetchTransport();
	}

	/** Update client configuration (partial merge). */
	updateConfig(partial: Partial<ClientConfig>): void {
		Object.assign(this.config, partial);
	}

	/** @deprecated Use updateConfig() instead. */
	setBaseUrl(url: string): void {
		this.config.localUrl = url;
	}

	/** Get the currently active base URL. */
	private getBaseUrl(): string {
		return this.config.apiMode === "cloud" ? this.config.cloudUrl : this.config.localUrl;
	}

	/** Get auth headers for the current mode. */
	private getAuthHeaders(): Record<string, string> {
		if (this.config.apiMode === "cloud" && this.config.apiToken) {
			return { "Authorization": `Bearer ${this.config.apiToken}` };
		}
		return {};
	}

	/** Get the endpoint path, adjusting for cloud vs local. */
	private getEndpoint(localPath: string): string {
		if (this.config.apiMode === "cloud") {
			// Map local endpoints to cloud API v1 endpoints
			switch (localPath) {
				case "/health": return "/api/v1/health";
				case "/recommend": return "/api/v1/recommend";
				default: return localPath; // Already a full path (e.g. /api/v1/library/*)
			}
		}
		return localPath;
	}

	/** Make an authenticated GET request. */
	private async authGet(path: string): Promise<unknown> {
		const url = `${this.getBaseUrl()}${this.getEndpoint(path)}`;
		return this.transport.get(url, this.getAuthHeaders());
	}

	/** Make an authenticated POST request. */
	private async authPost(path: string, body: unknown): Promise<unknown> {
		const url = `${this.getBaseUrl()}${this.getEndpoint(path)}`;
		return this.transport.post(url, body, this.getAuthHeaders());
	}

	/** Check if the server is healthy and ready. */
	async health(): Promise<HealthResponse> {
		const resp = await this.authGet("/health");
		const data = resp as HealthResponse;

		// Normalize cloud response: {status: "ready"} → {ready: true}
		if (this.config.apiMode === "cloud" && data.ready === undefined) {
			return {
				...data,
				ready: data.status === "ready",
			};
		}

		return data;
	}

	/** Get citation recommendations for a query. */
	async recommend(
		query: string,
		k: number,
		authorBoost: number,
		cursorSentenceIndex?: number,
		collectionId?: string | null
	): Promise<RecommendResponse> {
		const body: Record<string, unknown> = {
			query,
			k,
			author_boost: authorBoost,
		};
		if (cursorSentenceIndex !== undefined) {
			body.cursor_sentence_index = cursorSentenceIndex;
		}
		if (collectionId) {
			body.collection_id = collectionId;
		}
		const resp = await this.authPost("/recommend", body);
		return resp as RecommendResponse;
	}

	/** Save papers to the user's library. */
	async savePapers(request: SavePapersRequest): Promise<SavePapersResponse> {
		const resp = await this.authPost("/api/v1/library/papers", request);
		return resp as SavePapersResponse;
	}

	/** Check which papers are already in the user's library. */
	async checkLibrary(papers: Array<{ doi?: string | null; title: string }>): Promise<LibraryCheckResult[]> {
		const resp = await this.authPost("/api/v1/library/check", { papers }) as { results: LibraryCheckResult[] };
		return resp.results;
	}

	/** List the user's collections. */
	async getCollections(): Promise<Collection[]> {
		const resp = await this.authGet("/api/v1/library/collections") as { collections: Collection[] };
		return resp.collections;
	}

	/** Search tags by prefix. */
	async searchTags(query: string): Promise<Tag[]> {
		const q = encodeURIComponent(query);
		const resp = await this.authGet(`/api/v1/library/tags/search?q=${q}`) as { tags: Tag[] };
		return resp.tags;
	}

	/** Get local server config (embedder, method, mode, etc.). */
	async serverConfig(): Promise<ServerConfigResponse> {
		const url = `${this.config.localUrl}/config`;
		return this.transport.get(url) as Promise<ServerConfigResponse>;
	}

	/** Get cloud library processing status. */
	async libraryStatus(): Promise<LibraryStatusResponse> {
		return this.authGet("/api/v1/upload-library/status") as Promise<LibraryStatusResponse>;
	}

	/** Trigger cloud library refresh (Zotero Web API sync). */
	async libraryRefresh(): Promise<{ status: string }> {
		return this.authPost("/api/library/refresh", {}) as Promise<{ status: string }>;
	}
}
