/**
 * API client factory for the Word add-in.
 *
 * Creates an InCiteClient from the current settings, using FetchTransport.
 */

import { InCiteClient, FetchTransport } from "@incite/shared";
import type { ClientConfig } from "@incite/shared";
import { settings } from "./settings";

/** Create an InCiteClient configured from the current settings. */
export function createClient(): InCiteClient {
	const config: ClientConfig = {
		apiMode: settings.apiMode,
		cloudUrl: settings.cloudUrl,
		localUrl: settings.localUrl,
		apiToken: settings.apiToken,
	};
	return new InCiteClient(config, new FetchTransport());
}
