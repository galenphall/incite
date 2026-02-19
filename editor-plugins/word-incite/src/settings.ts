/**
 * Settings management for the Word add-in.
 *
 * Uses shared InCiteSettings as the base type. Handles localStorage
 * persistence and migration from legacy field names.
 */

import type { InCiteSettings, ApiMode } from "@incite/shared";
import { DEFAULT_SETTINGS as SHARED_DEFAULTS } from "@incite/shared";

export type WordSettings = InCiteSettings;

export const DEFAULT_SETTINGS: WordSettings = {
	...SHARED_DEFAULTS,
	// Word default differs: shared uses "({first_author}, {year})" already,
	// but we keep the explicit override in case shared changes.
	insertFormat: "({first_author}, {year})",
};

export let settings: WordSettings = { ...DEFAULT_SETTINGS };

/**
 * Load settings from localStorage, migrating legacy fields.
 * Falls back to defaults on parse error.
 */
export function loadSettings(): void {
	try {
		const saved = localStorage.getItem("incite-settings");
		if (saved) {
			const parsed = JSON.parse(saved);
			// Migrate old apiUrl to localUrl
			if (parsed.apiUrl && !parsed.localUrl) {
				parsed.localUrl = parsed.apiUrl;
				delete parsed.apiUrl;
			}
			settings = { ...DEFAULT_SETTINGS, ...parsed };
		}
	} catch (err) {
		console.error("loadSettings failed:", err);
	}
}

/** Persist current settings to localStorage. */
export function saveSettings(): void {
	try {
		localStorage.setItem("incite-settings", JSON.stringify(settings));
	} catch (err) {
		console.error("saveSettings failed:", err);
	}
}

/** Update a single setting, persist, and return the new value. */
export function updateSetting<K extends keyof WordSettings>(
	key: K,
	value: WordSettings[K],
): void {
	settings[key] = value;
	saveSettings();
}
