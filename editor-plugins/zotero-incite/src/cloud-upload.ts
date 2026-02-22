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

let uploadState: UploadState = { status: "idle", message: "" };

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
			papers: papers.map((p) => ({
				id: p.id,
				title: p.title,
				abstract: p.abstract,
				authors: p.authors,
				year: p.year,
				doi: p.doi,
				journal: p.journal,
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
				const formData = new FormData();

				for (const paper of batch) {
					const bytes = await IOUtils.read(paper.pdfPath!);
					const blob = new Blob([bytes.buffer as ArrayBuffer], { type: "application/pdf" });
					formData.append("files", blob, `${paper.id}.pdf`);
				}

				const pdfResp = await fetch(`${baseUrl}/api/v1/upload-library/pdfs`, {
					method: "POST",
					headers: { Authorization: `Bearer ${apiToken}` },
					body: formData,
				});
				if (!pdfResp.ok) {
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

		uploadState = { status: "done", message: "Upload complete! Server is processing your library." };
	} catch (e) {
		uploadState = { status: "error", message: String(e) };
	}
}
