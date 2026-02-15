import type { Recommendation } from "./types";

/** A citation that has been inserted into a document. */
export interface TrackedCitation {
	paper_id: string;
	bibtex_key: string;
	title: string;
	authors: string[];
	year?: number;
	doi?: string;
	journal?: string;
	abstract?: string;
	insertedAt: number;
}

/** Storage backend for persisting tracked citations. */
export interface CitationStorage {
	load(docKey: string): Promise<TrackedCitation[]>;
	save(docKey: string, citations: TrackedCitation[]): Promise<void>;
}

/** Convert a Recommendation to a TrackedCitation. */
export function recommendationToTracked(rec: Recommendation): TrackedCitation {
	return {
		paper_id: rec.paper_id,
		bibtex_key: rec.bibtex_key ?? rec.paper_id,
		title: rec.title,
		authors: rec.authors ?? [],
		year: rec.year,
		doi: rec.doi,
		journal: rec.journal,
		abstract: rec.abstract,
		insertedAt: Date.now(),
	};
}

/**
 * Tracks which citations have been inserted into a document.
 *
 * Backed by a pluggable CitationStorage implementation per editor plugin.
 */
export class CitationTracker {
	private citations: TrackedCitation[] = [];
	private storage: CitationStorage;
	private docKey: string;

	constructor(storage: CitationStorage, docKey: string) {
		this.storage = storage;
		this.docKey = docKey;
	}

	/** Load tracked citations from storage. */
	async load(): Promise<void> {
		this.citations = await this.storage.load(this.docKey);
	}

	/** Track one or more citations after insertion. Persists immediately. */
	async track(recs: Recommendation[]): Promise<void> {
		for (const rec of recs) {
			if (!this.isTracked(rec.paper_id)) {
				this.citations.push(recommendationToTracked(rec));
			}
		}
		await this.storage.save(this.docKey, this.citations);
	}

	/** Remove a tracked citation by paper ID. */
	async remove(paperId: string): Promise<void> {
		this.citations = this.citations.filter((c) => c.paper_id !== paperId);
		await this.storage.save(this.docKey, this.citations);
	}

	/** Get all tracked citations sorted by insertion time. */
	getAll(): TrackedCitation[] {
		return [...this.citations].sort((a, b) => a.insertedAt - b.insertedAt);
	}

	/** Check whether a paper has already been cited. */
	isTracked(paperId: string): boolean {
		return this.citations.some((c) => c.paper_id === paperId);
	}

	/** Number of tracked citations. */
	get count(): number {
		return this.citations.length;
	}
}
