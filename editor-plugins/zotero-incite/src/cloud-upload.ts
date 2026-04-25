/**
 * Uploads the user's Zotero library to the inCite cloud server.
 * Supports incremental (delta) sync when the server provides version tracking,
 * falling back to full upload for first sync or when the server requires it.
 */

import { readZoteroLibrary, type ZoteroPaper } from "./zotero-library";
import { PREF_PREFIX } from "./types";

export interface UploadState {
	status:
		| "idle"
		| "reading"
		| "checking_version"
		| "delta_syncing"
		| "uploading_metadata"
		| "uploading_pdfs"
		| "processing"
		| "done"
		| "error";
	message: string;
	current?: number;
	total?: number;
}

export interface SyncMetadataState {
	status: "idle" | "reading" | "syncing" | "done" | "error";
	message: string;
	updated?: number;
	skipped?: number;
}

let uploadState: UploadState = { status: "idle", message: "" };
let syncMetadataState: SyncMetadataState = { status: "idle", message: "" };

export function getUploadState(): UploadState {
	return { ...uploadState };
}

export function resetUploadState(): void {
	uploadState = { status: "idle", message: "" };
}

const PDF_BATCH_SIZE = 5;

// --- Sync state persistence via Zotero preferences ---

const SYNC_VERSION_PREF = `${PREF_PREFIX}.lastSyncVersion`;
const SYNC_KEYS_PREF = `${PREF_PREFIX}.lastSyncItemKeys`;
const SYNC_TIME_PREF = `${PREF_PREFIX}.lastSyncTime`;

interface SyncState {
	version: number;
	itemKeys: string[];
	syncTime: string; // ISO 8601 timestamp
}

/** Load the last successful sync state from Zotero preferences. */
function loadSyncState(): SyncState | null {
	try {
		const version = Zotero.Prefs.get(SYNC_VERSION_PREF, true) as number | undefined;
		const keysJson = Zotero.Prefs.get(SYNC_KEYS_PREF, true) as string | undefined;
		const syncTime = Zotero.Prefs.get(SYNC_TIME_PREF, true) as string | undefined;

		if (version === undefined || !keysJson || !syncTime) return null;

		const itemKeys = JSON.parse(keysJson) as string[];
		return { version, itemKeys, syncTime };
	} catch {
		return null;
	}
}

/** Save sync state after a successful sync. */
function saveSyncState(version: number, itemKeys: string[]): void {
	Zotero.Prefs.set(SYNC_VERSION_PREF, version, true);
	Zotero.Prefs.set(SYNC_KEYS_PREF, JSON.stringify(itemKeys), true);
	Zotero.Prefs.set(SYNC_TIME_PREF, new Date().toISOString(), true);
}

// --- Server version check ---

interface ServerVersionResponse {
	version: number;
	total_papers: number;
	status: string;
}

/**
 * Check the server's current library version.
 * Returns null if the endpoint doesn't exist (older server) or on error.
 */
async function getServerVersion(
	baseUrl: string,
	authHeaders: Record<string, string>,
): Promise<ServerVersionResponse | null> {
	try {
		const resp = await Zotero.HTTP.request("GET", `${baseUrl}/api/v1/upload-library/version`, {
			headers: authHeaders,
			responseType: "text",
			timeout: 15000,
		});
		if (resp.status >= 200 && resp.status < 300) {
			return JSON.parse(resp.responseText) as ServerVersionResponse;
		}
		return null;
	} catch {
		// Endpoint doesn't exist or network error — fall back to full upload
		return null;
	}
}

// --- Delta computation ---

interface DeltaResult {
	added: ZoteroPaper[];
	updated: ZoteroPaper[];
	deleted: string[];
}

/**
 * Compute the delta between current Zotero items and the last sync state.
 * This is a pure set-diff — no Zotero API calls needed.
 *
 * - Added: items whose keys are not in lastSyncedKeys
 * - Existing: items whose keys ARE in lastSyncedKeys (candidates for update check)
 * - Deleted: keys in lastSyncedKeys that are not in current items
 */
function computeDelta(
	currentPapers: ZoteroPaper[],
	lastSyncedKeys: string[],
): { added: ZoteroPaper[]; existing: ZoteroPaper[]; deleted: string[] } {
	const lastKeysSet = new Set(lastSyncedKeys);
	const currentKeysSet = new Set(currentPapers.map((p) => p.id));

	const added: ZoteroPaper[] = [];
	const existing: ZoteroPaper[] = [];

	for (const paper of currentPapers) {
		if (!lastKeysSet.has(paper.id)) {
			added.push(paper);
		} else {
			existing.push(paper);
		}
	}

	const deleted: string[] = [];
	for (const key of lastSyncedKeys) {
		if (!currentKeysSet.has(key)) {
			deleted.push(key);
		}
	}

	return { added, existing, deleted };
}

/**
 * Filter existing papers to only those actually modified since last sync.
 * Uses Zotero's getField("dateModified") to check modification timestamps.
 */
async function filterModifiedSinceSync(
	candidates: ZoteroPaper[],
	lastSyncTime: string,
): Promise<ZoteroPaper[]> {
	const syncDate = new Date(lastSyncTime);
	if (candidates.length === 0) return [];

	const candidateKeys = new Set(candidates.map((p) => p.id));

	// Build a lookup of item key -> dateModified across all relevant libraries
	const keyToModified = new Map<string, Date>();
	const includeGroups = Zotero.Prefs.get("extensions.incite.includeGroupLibraries", true);
	// eslint-disable-next-line @typescript-eslint/no-explicit-any -- Zotero 7 API not fully typed
	const libraries: any[] = includeGroups
		? Zotero.Libraries.getAll()
		: [Zotero.Libraries.get(Zotero.Libraries.userLibraryID)];

	for (const lib of libraries) {
		const items = await Zotero.Items.getAll(lib.libraryID, true, false);
		for (const item of items) {
			if (!item.isRegularItem()) continue;
			if (!candidateKeys.has(item.key)) continue;
			try {
				const modified = (item.getField("dateModified") as string) || "";
				if (modified) {
					keyToModified.set(item.key, new Date(modified));
				}
			} catch {
				// Skip items we can't read — they won't be included
			}
		}
	}

	const actuallyUpdated: ZoteroPaper[] = [];
	for (const paper of candidates) {
		const modified = keyToModified.get(paper.id);
		if (modified && modified > syncDate) {
			actuallyUpdated.push(paper);
		}
	}

	return actuallyUpdated;
}

// --- Delta upload ---

interface DeltaUploadResponse {
	version: number;
	added: number;
	updated: number;
	deleted: number;
	processing_triggered: boolean;
	requires_full_sync: boolean;
	needs_processing: boolean;
}

/** Extract the metadata payload for a paper (DRY helper). */
function paperToMetadata(p: ZoteroPaper): Record<string, unknown> {
	return {
		id: p.id,
		title: p.title,
		abstract: p.abstract,
		authors: p.authors,
		structured_authors: p.structured_authors,
		year: p.year,
		doi: p.doi,
		journal: p.journal,
		volume: p.volume,
		issue: p.issue,
		pages: p.pages,
		item_type: p.item_type,
	};
}

/**
 * Upload the Zotero library to the inCite cloud server.
 *
 * Attempts incremental (delta) sync when the server supports it:
 * 1. Check server version via GET /api/v1/upload-library/version
 * 2. If server supports delta and we have prior sync state, compute and send delta
 * 3. Fall back to full upload if: first sync, server requires it, or delta endpoint missing
 *
 * Full upload steps:
 * 1. Read Zotero library
 * 2. POST metadata to /api/v1/upload-library
 * 3. Upload PDFs in batches to /api/v1/upload-library/pdfs
 * 4. Trigger processing via /api/v1/upload-library/process
 */
export async function uploadToCloud(serverUrl: string, apiToken: string): Promise<void> {
	const baseUrl = serverUrl.replace(/\/+$/, "");
	const authHeaders: Record<string, string> = {
		Authorization: `Bearer ${apiToken}`,
		Accept: "application/json",
	};

	try {
		// Step 1: Read Zotero library
		uploadState = { status: "reading", message: "Reading Zotero library..." };
		const papers = await readZoteroLibrary();

		if (papers.length === 0) {
			uploadState = { status: "error", message: "No papers found in Zotero library" };
			return;
		}

		// Step 2: Check if server supports incremental sync
		uploadState = { status: "checking_version", message: "Checking server version..." };
		const serverVersion = await getServerVersion(baseUrl, authHeaders);
		const syncState = loadSyncState();

		// Try incremental sync if:
		// - Server supports the version endpoint
		// - We have a previous sync state
		if (serverVersion && syncState) {
			const deltaResult = await attemptDeltaSync(
				baseUrl,
				authHeaders,
				papers,
				syncState,
			);
			if (deltaResult === "done") {
				return; // Delta sync succeeded
			}
			// deltaResult === "full_required" — fall through to full upload
			uploadState = { status: "uploading_metadata", message: "Full sync required. Uploading all metadata..." };
		}

		// Full upload flow (first sync, server doesn't support delta, or delta requested full sync)
		await performFullUpload(baseUrl, authHeaders, papers);

		// After successful full upload, save sync state
		if (serverVersion) {
			// Re-check server version after upload to get the new version number
			const newVersion = await getServerVersion(baseUrl, authHeaders);
			if (newVersion) {
				saveSyncState(
					newVersion.version,
					papers.map((p) => p.id),
				);
			}
		} else {
			// Server doesn't support versioning — save with version 0 so we
			// at least track which keys were synced for future delta attempts
			saveSyncState(0, papers.map((p) => p.id));
		}

		uploadState = { status: "done", message: "Upload complete! Server is processing your library." };
	} catch (e) {
		uploadState = { status: "error", message: String(e) };
	}
}

/**
 * Attempt an incremental (delta) sync.
 * Returns "done" if successful, "full_required" if a full upload is needed.
 */
async function attemptDeltaSync(
	baseUrl: string,
	authHeaders: Record<string, string>,
	papers: ZoteroPaper[],
	syncState: SyncState,
): Promise<"done" | "full_required"> {
	// Compute what changed since last sync
	uploadState = { status: "delta_syncing", message: "Computing changes since last sync..." };
	const rawDelta = computeDelta(papers, syncState.itemKeys);

	// Filter existing papers to only those actually modified since last sync
	const actuallyUpdated = await filterModifiedSinceSync(rawDelta.existing, syncState.syncTime);

	const delta: DeltaResult = {
		added: rawDelta.added,
		updated: actuallyUpdated,
		deleted: rawDelta.deleted,
	};

	// If no changes, we're done
	if (delta.added.length === 0 && delta.updated.length === 0 && delta.deleted.length === 0) {
		uploadState = { status: "done", message: "Library up to date. No changes to sync." };
		return "done";
	}

	// Report what we're syncing
	const parts: string[] = [];
	if (delta.added.length > 0) parts.push(`${delta.added.length} new`);
	if (delta.updated.length > 0) parts.push(`${delta.updated.length} updated`);
	if (delta.deleted.length > 0) parts.push(`${delta.deleted.length} deleted`);
	uploadState = {
		status: "delta_syncing",
		message: `Syncing ${parts.join(", ")} papers...`,
	};

	// Send delta to server
	const deltaBody = {
		since_version: syncState.version,
		added: delta.added.map(paperToMetadata),
		updated: delta.updated.map(paperToMetadata),
		deleted: delta.deleted,
	};

	try {
		const resp = await Zotero.HTTP.request("POST", `${baseUrl}/api/v1/upload-library/delta`, {
			headers: { ...authHeaders, "Content-Type": "application/json" },
			body: JSON.stringify(deltaBody),
			responseType: "text",
			timeout: 60000,
		});

		if (resp.status < 200 || resp.status >= 300) {
			// Delta endpoint returned an error — fall back to full upload
			Zotero.debug(`inCite: delta sync failed with HTTP ${resp.status}, falling back to full upload`);
			return "full_required";
		}

		const result = JSON.parse(resp.responseText) as DeltaUploadResponse;

		// Server may request a full sync (e.g., version mismatch, data integrity issue)
		if (result.requires_full_sync) {
			Zotero.debug("inCite: server requested full sync");
			return "full_required";
		}

		// Upload PDFs for newly added and updated papers
		const papersWithPdf = [...delta.added, ...delta.updated].filter((p) => p.pdfPath);
		if (papersWithPdf.length > 0) {
			await uploadPdfs(baseUrl, authHeaders, papersWithPdf);
		}

		// Trigger processing after PDFs are uploaded (server can't trigger
		// during the delta response because PDFs aren't on disk yet)
		if (result.needs_processing && papersWithPdf.length > 0) {
			uploadState = { status: "processing", message: "Starting server-side processing..." };
			const procResp = await Zotero.HTTP.request("POST", `${baseUrl}/api/v1/upload-library/process`, {
				headers: authHeaders,
				responseType: "text",
				timeout: 30000,
			});
			if (procResp.status < 200 || procResp.status >= 300) {
				Zotero.debug(`inCite: process trigger after delta sync failed: HTTP ${procResp.status}`);
			}
		}

		// Save updated sync state
		saveSyncState(
			result.version,
			papers.map((p) => p.id),
		);

		const msg = `Sync complete: ${result.added} added, ${result.updated} updated, ${result.deleted} deleted.`;
		uploadState = { status: "done", message: msg };
		return "done";
	} catch (e) {
		// Network error or unexpected failure — fall back to full upload
		Zotero.debug(`inCite: delta sync error: ${e}, falling back to full upload`);
		return "full_required";
	}
}

/**
 * Perform a full library upload (the original flow).
 * Does NOT set the final "done" state — the caller handles that after saving sync state.
 */
async function performFullUpload(
	baseUrl: string,
	authHeaders: Record<string, string>,
	papers: ZoteroPaper[],
): Promise<void> {
	// Upload metadata
	uploadState = { status: "uploading_metadata", message: `Uploading metadata for ${papers.length} papers...` };
	const metadataBody = {
		source: "zotero-plugin",
		papers: papers.map(paperToMetadata),
	};

	const metaResp = await Zotero.HTTP.request("POST", `${baseUrl}/api/v1/upload-library`, {
		headers: { ...authHeaders, "Content-Type": "application/json" },
		body: JSON.stringify(metadataBody),
		responseType: "text",
		timeout: 60000,
	});
	if (metaResp.status < 200 || metaResp.status >= 300) {
		throw new Error(`Metadata upload failed: HTTP ${metaResp.status}`);
	}

	// Upload PDFs
	const papersWithPdf = papers.filter((p) => p.pdfPath);
	if (papersWithPdf.length > 0) {
		await uploadPdfs(baseUrl, authHeaders, papersWithPdf);
	}

	// Trigger processing
	uploadState = { status: "processing", message: "Starting server-side processing..." };
	const procResp = await Zotero.HTTP.request("POST", `${baseUrl}/api/v1/upload-library/process`, {
		headers: authHeaders,
		responseType: "text",
		timeout: 30000,
	});
	if (procResp.status < 200 || procResp.status >= 300) {
		throw new Error(`Process trigger failed: HTTP ${procResp.status}`);
	}

	// Sync metadata (best-effort)
	try {
		const syncBody = {
			papers: papers.map(paperToMetadata),
		};
		await Zotero.HTTP.request("POST", `${baseUrl}/api/v1/upload-library/sync-metadata`, {
			headers: { ...authHeaders, "Content-Type": "application/json" },
			body: JSON.stringify(syncBody),
			responseType: "text",
			timeout: 60000,
		});
	} catch {
		// Best-effort — upload still succeeded even if metadata sync fails
	}
}

/** Upload PDFs in batches. Updates uploadState with progress. */
async function uploadPdfs(
	baseUrl: string,
	authHeaders: Record<string, string>,
	papersWithPdf: ZoteroPaper[],
): Promise<void> {
	uploadState = {
		status: "uploading_pdfs",
		message: `Uploading ${papersWithPdf.length} PDFs...`,
		current: 0,
		total: papersWithPdf.length,
	};

	for (let i = 0; i < papersWithPdf.length; i += PDF_BATCH_SIZE) {
		const batch = papersWithPdf.slice(i, i + PDF_BATCH_SIZE);

		// Build multipart body manually — FormData/Blob are unavailable in Zotero's Gecko environment
		const boundary = "----IncitePdfUpload" + Date.now();
		const parts: Uint8Array[] = [];
		const encoder = new TextEncoder();

		for (const paper of batch) {
			const bytes = await IOUtils.read(paper.pdfPath!);
			const header =
				`--${boundary}\r\n` +
				`Content-Disposition: form-data; name="files"; filename="${paper.id}.pdf"\r\n` +
				`Content-Type: application/pdf\r\n\r\n`;
			parts.push(encoder.encode(header));
			parts.push(bytes);
			parts.push(encoder.encode("\r\n"));
		}
		parts.push(encoder.encode(`--${boundary}--\r\n`));

		// Concatenate all parts into a single Uint8Array
		let totalLen = 0;
		for (const p of parts) totalLen += p.length;
		const body = new Uint8Array(totalLen);
		let offset = 0;
		for (const p of parts) {
			body.set(p, offset);
			offset += p.length;
		}

		const pdfResp = await Zotero.HTTP.request("POST", `${baseUrl}/api/v1/upload-library/pdfs`, {
			headers: {
				...authHeaders,
				"Content-Type": `multipart/form-data; boundary=${boundary}`,
			},
			body,
			responseType: "text",
			timeout: 120000,
		});
		if (pdfResp.status < 200 || pdfResp.status >= 300) {
			throw new Error(`PDF upload failed: HTTP ${pdfResp.status}`);
		}

		const uploaded = Math.min(i + PDF_BATCH_SIZE, papersWithPdf.length);
		uploadState = {
			status: "uploading_pdfs",
			message: `Uploaded ${uploaded}/${papersWithPdf.length} PDFs...`,
			current: uploaded,
			total: papersWithPdf.length,
		};
	}
}

export function getSyncMetadataState(): SyncMetadataState {
	return { ...syncMetadataState };
}

/**
 * Sync metadata only (no PDFs, no reprocessing).
 * Pushes current Zotero metadata to the cloud to update author names, etc.
 */
export async function syncMetadataToCloud(serverUrl: string, apiToken: string): Promise<void> {
	const baseUrl = serverUrl.replace(/\/+$/, "");
	const authHeaders: Record<string, string> = {
		Authorization: `Bearer ${apiToken}`,
		Accept: "application/json",
	};

	try {
		syncMetadataState = { status: "reading", message: "Reading Zotero library..." };
		const papers = await readZoteroLibrary();

		if (papers.length === 0) {
			syncMetadataState = { status: "error", message: "No papers found in Zotero library" };
			return;
		}

		syncMetadataState = { status: "syncing", message: `Syncing metadata for ${papers.length} papers...` };
		const body = {
			papers: papers.map(paperToMetadata),
		};

		const resp = await Zotero.HTTP.request("POST", `${baseUrl}/api/v1/upload-library/sync-metadata`, {
			headers: { ...authHeaders, "Content-Type": "application/json" },
			body: JSON.stringify(body),
			responseType: "text",
			timeout: 60000,
		});

		if (resp.status < 200 || resp.status >= 300) {
			throw new Error(`Metadata sync failed: HTTP ${resp.status}`);
		}

		const result = JSON.parse(resp.responseText);
		syncMetadataState = {
			status: "done",
			message: `Synced metadata: ${result.updated} updated, ${result.skipped} skipped`,
			updated: result.updated,
			skipped: result.skipped,
		};
	} catch (e) {
		syncMetadataState = { status: "error", message: String(e) };
	}
}
