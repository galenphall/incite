export type {
	ApiMode,
	InCiteSettings,
	ClientConfig,
	HealthResponse,
} from "@incite/shared";

export { DEFAULT_SETTINGS, getActiveUrl } from "@incite/shared";

/** Zotero-specific default settings (local mode, no insert/autodetect). */
export const ZOTERO_DEFAULTS = {
	apiMode: "local" as const,
	cloudUrl: "https://inciteref.com",
	localUrl: "http://127.0.0.1:8230",
	apiToken: "",
};

/** Preference key prefix. */
export const PREF_PREFIX = "extensions.incite";

/** Plugin ID matching manifest.json. */
export const PLUGIN_ID = "incite@galenphall.com";
