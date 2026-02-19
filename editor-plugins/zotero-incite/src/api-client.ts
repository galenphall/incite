import { InCiteClient as SharedClient } from "@incite/shared";
import type { HttpTransport, ClientConfig } from "@incite/shared";

/** Zotero-specific HTTP transport using Zotero.HTTP.request(). */
class ZoteroTransport implements HttpTransport {
	async get(url: string, headers?: Record<string, string>): Promise<unknown> {
		const resp = await Zotero.HTTP.request("GET", url, {
			headers: { Accept: "application/json", ...headers },
			responseType: "text",
		});
		if (resp.status < 200 || resp.status >= 300) {
			throw new Error(`HTTP ${resp.status}: ${resp.responseText}`);
		}
		return JSON.parse(resp.responseText);
	}

	async post(url: string, body: unknown, headers?: Record<string, string>): Promise<unknown> {
		const resp = await Zotero.HTTP.request("POST", url, {
			headers: {
				"Content-Type": "application/json",
				"Accept": "application/json",
				...headers,
			},
			body: JSON.stringify(body),
			responseType: "text",
		});
		if (resp.status < 200 || resp.status >= 300) {
			throw new Error(`HTTP ${resp.status}: ${resp.responseText}`);
		}
		return JSON.parse(resp.responseText);
	}
}

/** inCite API client configured for Zotero's HTTP transport. */
export class InCiteClient extends SharedClient {
	constructor(config: ClientConfig) {
		super(config, new ZoteroTransport());
	}
}
