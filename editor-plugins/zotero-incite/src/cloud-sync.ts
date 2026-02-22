/**
 * One-way cloud-to-Zotero sync: pulls new papers, collections, and tags
 * from the inCite cloud into the user's local Zotero library.
 */

export interface SyncState {
	status: "idle" | "fetching" | "syncing" | "done" | "error";
	message: string;
	created?: number;
	skipped?: number;
	tagsAdded?: number;
}

let syncState: SyncState = { status: "idle", message: "" };
let syncRunning = false;

export function getSyncState(): SyncState {
	return { ...syncState };
}

export function resetSyncState(): void {
	syncState = { status: "idle", message: "" };
}

/** Preference key prefix for storing canonical_id -> Zotero item key mapping. */
const MAPPING_PREF = "extensions.incite.syncMapping";

interface CloudPaper {
	canonical_id: string;
	title: string;
	abstract: string;
	authors: string[];
	year: number | null;
	doi: string;
	journal: string;
	tags: string[];
	collections: number[];
}

interface CloudCollection {
	id: number;
	name: string;
	color: string | null;
}

interface CloudSyncResponse {
	papers: CloudPaper[];
	collections: CloudCollection[];
}

/** Load the stored canonical_id -> Zotero item key mapping from prefs. */
function loadMapping(): Record<string, string> {
	try {
		const raw = Zotero.Prefs.get(MAPPING_PREF, true) as string | undefined;
		if (raw) return JSON.parse(raw);
	} catch {
		// Corrupt pref — start fresh
	}
	return {};
}

/** Save the mapping to prefs. */
function saveMapping(mapping: Record<string, string>): void {
	Zotero.Prefs.set(MAPPING_PREF, JSON.stringify(mapping), true);
}

/**
 * Sync papers from the inCite cloud into the local Zotero library.
 *
 * - Fetches all papers + collections from cloud
 * - Matches cloud papers to existing Zotero items by DOI, then title
 * - Creates new Zotero items for unmatched papers
 * - Syncs collections: find-or-create Zotero collections by name, add items
 * - Syncs tags: add missing cloud tags to existing Zotero items
 */
export async function syncFromCloud(serverUrl: string, apiToken: string): Promise<void> {
	if (syncRunning) {
		syncState = { status: "error", message: "Sync already in progress" };
		return;
	}
	syncRunning = true;
	const baseUrl = serverUrl.replace(/\/+$/, "");

	try {
		// Step 1: Fetch all papers from cloud
		syncState = { status: "fetching", message: "Fetching papers from cloud..." };
		const resp = await Zotero.HTTP.request("GET", `${baseUrl}/api/v1/library/papers`, {
			headers: {
				Authorization: `Bearer ${apiToken}`,
				Accept: "application/json",
			},
			responseType: "text",
			timeout: 60000,
		});
		if (resp.status < 200 || resp.status >= 300) {
			throw new Error(`Cloud API error: HTTP ${resp.status}`);
		}
		const data: CloudSyncResponse = JSON.parse(resp.responseText);

		if (!data.papers || data.papers.length === 0) {
			syncState = { status: "done", message: "No papers in cloud library.", created: 0, skipped: 0 };
			return;
		}

		// Step 2: Build local lookup indexes
		syncState = { status: "syncing", message: "Scanning local Zotero library..." };
		const libraryID = Zotero.Libraries.userLibraryID;
		const allItems = await Zotero.Items.getAll(libraryID, true, false);
		const mapping = loadMapping();

		// Index existing items by DOI and title for matching
		const doiIndex = new Map<string, Zotero.Item>();
		const titleIndex = new Map<string, Zotero.Item>();
		for (const item of allItems) {
			if (!item.isRegularItem()) continue;
			const doi = item.getField("DOI")?.toLowerCase().trim();
			if (doi) doiIndex.set(doi, item);
			const title = item.getField("title")?.toLowerCase().trim();
			if (title) titleIndex.set(title, item);
		}

		let created = 0;
		let skipped = 0;
		let tagsAdded = 0;

		// Step 3: Process each cloud paper
		for (let i = 0; i < data.papers.length; i++) {
			const cp = data.papers[i];
			syncState = {
				status: "syncing",
				message: `Processing paper ${i + 1}/${data.papers.length}...`,
				created,
				skipped,
				tagsAdded,
			};

			// Check if already mapped from a previous sync
			if (mapping[cp.canonical_id]) {
				// Item was previously synced — just ensure tags are up to date
				try {
					const existingItem = await findItemByKey(mapping[cp.canonical_id], libraryID);
					if (existingItem) {
						tagsAdded += await syncTags(existingItem, cp.tags);
						skipped++;
						continue;
					}
				} catch {
					// Key no longer valid — fall through to match/create
				}
			}

			// Try to match by DOI
			let matchedItem: Zotero.Item | undefined;
			if (cp.doi) {
				matchedItem = doiIndex.get(cp.doi.toLowerCase().trim());
			}

			// Try to match by title
			if (!matchedItem && cp.title) {
				matchedItem = titleIndex.get(cp.title.toLowerCase().trim());
			}

			if (matchedItem) {
				// Existing item — record mapping, sync tags
				mapping[cp.canonical_id] = matchedItem.key;
				tagsAdded += await syncTags(matchedItem, cp.tags);
				skipped++;
			} else {
				// Create new Zotero item
				const newItem = new Zotero.Item("journalArticle") as Zotero.Item;
				newItem.libraryID = libraryID;
				newItem.setField("title", cp.title || "");
				if (cp.abstract) newItem.setField("abstractNote", cp.abstract);
				if (cp.doi) newItem.setField("DOI", cp.doi);
				if (cp.journal) newItem.setField("publicationTitle", cp.journal);
				if (cp.year) newItem.setField("date", String(cp.year));

				// Set authors
				if (cp.authors && cp.authors.length > 0) {
					const creators = cp.authors.map((name) => {
						const parts = name.trim().split(/\s+/);
						const lastName = parts.pop() || name;
						const firstName = parts.join(" ");
						return { firstName, lastName, creatorType: "author" };
					});
					newItem.setCreators(creators);
				}

				// Add tags
				for (const tag of cp.tags) {
					newItem.addTag(tag);
				}

				await newItem.saveTx();
				mapping[cp.canonical_id] = newItem.key;

				// Add to indexes for dedup within this sync
				if (cp.doi) doiIndex.set(cp.doi.toLowerCase().trim(), newItem);
				if (cp.title) titleIndex.set(cp.title.toLowerCase().trim(), newItem);

				created++;
			}
		}

		// Step 4: Sync collections
		syncState = { status: "syncing", message: "Syncing collections...", created, skipped, tagsAdded };
		if (data.collections && data.collections.length > 0) {
			await syncCollections(data.collections, data.papers, mapping, libraryID);
		}

		// Save mapping
		saveMapping(mapping);

		syncState = {
			status: "done",
			message: `Sync complete. Created ${created} papers, skipped ${skipped} existing.${tagsAdded > 0 ? ` Added ${tagsAdded} tags.` : ""}`,
			created,
			skipped,
			tagsAdded,
		};
	} catch (e) {
		syncState = { status: "error", message: String(e) };
	} finally {
		syncRunning = false;
	}
}

/** Find a Zotero item by its key in the given library. */
async function findItemByKey(key: string, libraryID: number): Promise<Zotero.Item | null> {
	try {
		const item = await Zotero.Items.getByLibraryAndKey(libraryID, key);
		return item || null;
	} catch {
		return null;
	}
}

/** Add missing tags to an existing Zotero item. Returns count of tags added. */
async function syncTags(item: Zotero.Item, cloudTags: string[]): Promise<number> {
	if (!cloudTags || cloudTags.length === 0) return 0;

	const existingTags = new Set(item.getTags().map((t) => t.tag));
	let added = 0;
	for (const tag of cloudTags) {
		if (!existingTags.has(tag)) {
			item.addTag(tag);
			added++;
		}
	}
	if (added > 0) {
		await item.saveTx();
	}
	return added;
}

/** Sync cloud collections into Zotero. */
async function syncCollections(
	cloudCollections: CloudCollection[],
	cloudPapers: CloudPaper[],
	mapping: Record<string, string>,
	libraryID: number,
): Promise<void> {
	// Get existing Zotero collections
	const existingCollections = Zotero.Collections.getByLibrary(libraryID);
	const nameToCollection = new Map<string, Zotero.Collection>();
	for (const col of existingCollections) {
		nameToCollection.set(col.name, col);
	}

	// Build cloud collection ID -> Zotero collection
	const cloudIdToZoteroCollection = new Map<number, Zotero.Collection>();

	for (const cc of cloudCollections) {
		let zotCol = nameToCollection.get(cc.name);
		if (!zotCol) {
			// Create new collection
			zotCol = new Zotero.Collection();
			zotCol.libraryID = libraryID;
			zotCol.name = cc.name;
			await zotCol.saveTx();
			nameToCollection.set(cc.name, zotCol);
		}
		cloudIdToZoteroCollection.set(cc.id, zotCol);
	}

	// Add papers to their collections
	for (const cp of cloudPapers) {
		if (!cp.collections || cp.collections.length === 0) continue;
		const itemKey = mapping[cp.canonical_id];
		if (!itemKey) continue;

		const item = await findItemByKey(itemKey, libraryID);
		if (!item) continue;

		for (const colId of cp.collections) {
			const zotCol = cloudIdToZoteroCollection.get(colId);
			if (zotCol && !zotCol.hasItem(item.id)) {
				zotCol.addItem(item.id);
				await zotCol.saveTx();
			}
		}
	}
}
