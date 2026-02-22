import type { App, TFile } from "obsidian";
import type { CitationStorage, TrackedCitation } from "@incite/shared";

const FRONTMATTER_KEY = "incite-citations";

/** Strip the abstract field from a citation before saving to frontmatter. */
function stripAbstract(citation: TrackedCitation): Omit<TrackedCitation, "abstract"> {
	const { abstract, ...rest } = citation;
	return rest;
}

/**
 * Citation storage backed by Obsidian YAML frontmatter.
 *
 * Citations are stored in the `incite-citations` frontmatter field of each
 * markdown file, making them portable and visible to other tools.
 */
export class FrontmatterCitationStorage implements CitationStorage {
	private app: App;

	constructor(app: App) {
		this.app = app;
	}

	async load(docKey: string): Promise<TrackedCitation[]> {
		const file = this.resolveFile(docKey);
		if (!file) return [];

		const cache = this.app.metadataCache.getFileCache(file);
		const stored = cache?.frontmatter?.[FRONTMATTER_KEY];
		if (!Array.isArray(stored)) return [];

		return stored as TrackedCitation[];
	}

	async save(docKey: string, citations: TrackedCitation[]): Promise<void> {
		const file = this.resolveFile(docKey);
		if (!file) return;

		const stripped = citations.map(stripAbstract);
		await this.app.fileManager.processFrontMatter(file, (fm) => {
			fm[FRONTMATTER_KEY] = stripped;
		});
	}

	private resolveFile(docKey: string): TFile | null {
		const abstract = this.app.vault.getAbstractFileByPath(docKey);
		if (!abstract || !("stat" in abstract)) return null;
		return abstract as TFile;
	}
}
