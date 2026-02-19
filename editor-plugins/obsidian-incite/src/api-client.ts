import { requestUrl, RequestUrlParam } from "obsidian";
import { InCiteClient as SharedClient } from "@incite/shared";
import type { HttpTransport, ClientConfig } from "@incite/shared";

/** Obsidian-specific HTTP transport using requestUrl(). */
class ObsidianTransport implements HttpTransport {
	async get(url: string, headers?: Record<string, string>): Promise<unknown> {
		const params: RequestUrlParam = {
			url,
			method: "GET",
			headers: headers ? { ...headers } : undefined,
		};
		const response = await requestUrl(params);
		return response.json;
	}

	async post(url: string, body: unknown, headers?: Record<string, string>): Promise<unknown> {
		const params: RequestUrlParam = {
			url,
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				...headers,
			},
			body: JSON.stringify(body),
		};
		const response = await requestUrl(params);
		return response.json;
	}
}

/** inCite API client configured for Obsidian's HTTP transport. */
export class InCiteClient extends SharedClient {
	constructor(config: ClientConfig) {
		super(config, new ObsidianTransport());
	}
}
