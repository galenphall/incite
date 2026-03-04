/**
 * Uploads the user's Zotero library to the inCite cloud server.
 * Reads papers via zotero-library.ts, uploads metadata + PDFs, then triggers processing.
 */

import { readZoteroLibrary, type ZoteroPaper } from "./zotero-library";

export interface UploadState {
	status: "idle" | "reading" | "uploading_metadata" | "uploading_pdfs" | "processing" | "done" | "error";
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

/**
 * Upload the Zotero library to the inCite cloud server.
 *
 * Steps:
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

		// Step 2: Upload metadata
		uploadState = { status: "uploading_metadata", message: `Uploading metadata for ${papers.length} papers...` };
		const metadataBody = {
			source: "zotero-plugin",
			papers: papers.map((p) => ({
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
			})),
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

		// Step 3: Upload PDFs in batches
		const papersWithPdf = papers.filter((p) => p.pdfPath);
		if (papersWithPdf.length > 0) {
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

		// Step 4: Trigger processing
		uploadState = { status: "processing", message: "Starting server-side processing..." };
		const procResp = await Zotero.HTTP.request("POST", `${baseUrl}/api/v1/upload-library/process`, {
			headers: authHeaders,
			responseType: "text",
			timeout: 30000,
		});
		if (procResp.status < 200 || procResp.status >= 300) {
			throw new Error(`Process trigger failed: HTTP ${procResp.status}`);
		}

		// Step 5: Sync metadata (pushes full author names from Zotero)
		try {
			const syncBody = {
				papers: papers.map((p) => ({
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
				})),
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

		uploadState = { status: "done", message: "Upload complete! Server is processing your library." };
	} catch (e) {
		uploadState = { status: "error", message: String(e) };
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
			papers: papers.map((p) => ({
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
			})),
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
