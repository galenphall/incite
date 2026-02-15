import type { Plugin } from "obsidian";
import type { CitationStorage, TrackedCitation } from "@incite/shared";

const STORAGE_KEY = "incite-tracked-citations";

interface StorageData {
	[docKey: string]: TrackedCitation[];
}

/**
 * Obsidian-native citation storage using plugin.loadData()/saveData().
 * Citations are keyed by the active file path.
 */
export class ObsidianCitationStorage implements CitationStorage {
	private plugin: Plugin;

	constructor(plugin: Plugin) {
		this.plugin = plugin;
	}

	async load(docKey: string): Promise<TrackedCitation[]> {
		const data = await this.plugin.loadData();
		const stored: StorageData = data?.[STORAGE_KEY] ?? {};
		return stored[docKey] ?? [];
	}

	async save(docKey: string, citations: TrackedCitation[]): Promise<void> {
		const data = (await this.plugin.loadData()) ?? {};
		const stored: StorageData = data[STORAGE_KEY] ?? {};
		stored[docKey] = citations;
		data[STORAGE_KEY] = stored;
		await this.plugin.saveData(data);
	}
}
