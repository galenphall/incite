/**
 * Office.js document-settings-backed citation storage for the Word add-in.
 *
 * Persists citations inside the .docx file itself via
 * Office.context.document.settings, so they travel with the document
 * across devices.
 */

import type { CitationStorage, TrackedCitation } from "@incite/shared";

const SETTINGS_KEY = "incite-citations";

/** CitationStorage implementation using Office document settings. */
export class DocumentCitationStorage implements CitationStorage {
	async load(_docKey: string): Promise<TrackedCitation[]> {
		try {
			const raw = Office.context.document.settings.get(SETTINGS_KEY);
			if (!raw) return [];
			return JSON.parse(raw) as TrackedCitation[];
		} catch (err) {
			console.error("DocumentCitationStorage.load failed:", err);
			return [];
		}
	}

	async save(_docKey: string, citations: TrackedCitation[]): Promise<void> {
		try {
			Office.context.document.settings.set(SETTINGS_KEY, JSON.stringify(citations));
			await new Promise<void>((resolve, reject) => {
				Office.context.document.settings.saveAsync((result) => {
					if (result.status === Office.AsyncResultStatus.Failed) {
						reject(new Error(result.error?.message ?? "saveAsync failed"));
					} else {
						resolve();
					}
				});
			});
		} catch (err) {
			console.error("DocumentCitationStorage.save failed:", err);
		}
	}
}
