import type { ApiMode, ClientConfig } from "@incite/shared";
import { PREF_PREFIX, PLUGIN_ID, ZOTERO_DEFAULTS } from "./types";

/** Read a preference with fallback to default. */
function getPref<T extends string | number | boolean>(key: string, fallback: T): T {
	const val = Zotero.Prefs.get(`${PREF_PREFIX}.${key}`, true);
	return (val !== undefined ? val : fallback) as T;
}

/** Write a preference. */
export function setPref(key: string, value: string | number | boolean): void {
	Zotero.Prefs.set(`${PREF_PREFIX}.${key}`, value, true);
}

/** Load all settings from Zotero preferences into a ClientConfig & Zotero extras. */
export function loadClientConfig(): ClientConfig & { includeGroupLibraries: boolean; connectedEmail: string } {
	return {
		apiMode: getPref("apiMode", ZOTERO_DEFAULTS.apiMode) as ApiMode,
		cloudUrl: getPref("cloudUrl", ZOTERO_DEFAULTS.cloudUrl),
		localUrl: getPref("localUrl", ZOTERO_DEFAULTS.localUrl),
		apiToken: getPref("apiToken", ZOTERO_DEFAULTS.apiToken),
		includeGroupLibraries: getPref("includeGroupLibraries", true),
		connectedEmail: getPref("connectedEmail", ""),
	};
}

/** Register the preferences pane in Zotero's settings UI. */
export function registerPreferencesPane(): void {
	Zotero.PreferencePanes.register({
		pluginID: PLUGIN_ID,
		src: rootURI + "content/preferences.xhtml",
		label: "inCite",
		image: rootURI + "content/icons/icon16.svg",
	});
}
