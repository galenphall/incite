/**
 * LocalStorage-backed citation storage for the Word add-in.
 */

import type { CitationStorage, TrackedCitation } from "@incite/shared";

/** CitationStorage implementation using browser localStorage. */
export class LocalStorageCitationStorage implements CitationStorage {
	async load(docKey: string): Promise<TrackedCitation[]> {
		try {
			const raw = localStorage.getItem(`incite-citations-${docKey}`);
			return raw ? JSON.parse(raw) : [];
		} catch (err) {
			console.error("loadSavedCitations failed:", err);
			return [];
		}
	}

	async save(docKey: string, citations: TrackedCitation[]): Promise<void> {
		try {
			localStorage.setItem(
				`incite-citations-${docKey}`,
				JSON.stringify(citations),
			);
		} catch (err) {
			console.error("saveCitations failed:", err);
		}
	}
}
