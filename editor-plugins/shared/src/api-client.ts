import type { HealthResponse, RecommendResponse } from "./types";

/**
 * HTTP transport interface — each editor provides its own implementation.
 *
 * Obsidian uses its built-in requestUrl(); VS Code and Google Docs use fetch().
 */
export interface HttpTransport {
	get(url: string): Promise<unknown>;
	post(url: string, body: unknown): Promise<unknown>;
}

/** fetch()-based transport for environments with native fetch (VS Code, Google Docs). */
export class FetchTransport implements HttpTransport {
	async get(url: string): Promise<unknown> {
		const response = await fetch(url, {
			method: "GET",
			headers: { "Accept": "application/json" },
		});
		if (!response.ok) {
			throw new Error(`HTTP ${response.status}: ${response.statusText}`);
		}
		return response.json();
	}

	async post(url: string, body: unknown): Promise<unknown> {
		const response = await fetch(url, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				"Accept": "application/json",
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
	private baseUrl: string;
	private transport: HttpTransport;

	constructor(baseUrl: string, transport?: HttpTransport) {
		this.baseUrl = baseUrl;
		this.transport = transport ?? new FetchTransport();
	}

	setBaseUrl(url: string): void {
		this.baseUrl = url;
	}

	/** Check if the server is healthy and ready. */
	async health(): Promise<HealthResponse> {
		const resp = await this.transport.get(`${this.baseUrl}/health`);
		return resp as HealthResponse;
	}

	/** Get citation recommendations for a query. */
	async recommend(
		query: string,
		k: number,
		authorBoost: number,
		cursorSentenceIndex?: number
	): Promise<RecommendResponse> {
		const body: Record<string, unknown> = {
			query,
			k,
			author_boost: authorBoost,
		};
		if (cursorSentenceIndex !== undefined) {
			body.cursor_sentence_index = cursorSentenceIndex;
		}
		const resp = await this.transport.post(`${this.baseUrl}/recommend`, body);
		return resp as RecommendResponse;
	}
}
