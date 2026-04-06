/**
 * Authenticated PDF download + upload to the inCite backend.
 *
 * The Chrome extension downloads PDFs using the user's browser cookies
 * (which include institutional proxy authentication), then uploads the
 * bytes to our backend for storage and GROBID processing.
 */

import { getActiveUrl } from "@incite/shared";
import type { ChromeExtensionSettings } from "./types";

const MAX_PDF_SIZE = 50 * 1024 * 1024; // 50 MB
const RETRY_STORAGE_KEY = "incite_pdf_retry_queue";
const MAX_RETRIES = 3;
const RETRY_BACKOFF_MS = 30_000;

interface RetryEntry {
  canonical_id: string;
  pdf_url: string;
  attempts: number;
  last_attempt: number;
}

/**
 * Download a PDF using the browser's cookies and upload it to the backend.
 * Returns true if upload succeeded, false otherwise.
 */
export async function downloadAndUploadPdf(
  canonical_id: string,
  pdf_url: string,
  settings: ChromeExtensionSettings,
): Promise<boolean> {
  if (!settings.apiToken) return false;

  try {
    // Download PDF using browser cookies (credentials: "include" sends
    // cookies for the target domain — works for publisher sites and
    // EZproxy patterns like publisher.com.proxy.library.edu).
    const response = await fetch(pdf_url, {
      credentials: "include",
      headers: { Accept: "application/pdf,*/*" },
    });

    if (!response.ok) {
      console.warn(`[inCite PDF] Download failed (${response.status}) for ${pdf_url}`);
      await addToRetryQueue(canonical_id, pdf_url);
      return false;
    }

    const arrayBuffer = await response.arrayBuffer();
    const pdfData = new Uint8Array(arrayBuffer);

    // Validate PDF
    if (pdfData.length > MAX_PDF_SIZE) {
      console.warn(`[inCite PDF] Too large (${pdfData.length} bytes): ${pdf_url}`);
      return false; // Don't retry — it's permanently too large
    }

    const header = new TextDecoder().decode(pdfData.slice(0, 5));
    if (header !== "%PDF-") {
      console.warn(`[inCite PDF] Invalid PDF header from ${pdf_url}`);
      await addToRetryQueue(canonical_id, pdf_url);
      return false;
    }

    // Upload to backend
    return await uploadPdfToBackend(canonical_id, pdfData, settings);
  } catch (err) {
    console.warn(`[inCite PDF] Download error for ${pdf_url}:`, err);
    await addToRetryQueue(canonical_id, pdf_url);
    return false;
  }
}

/**
 * Upload PDF bytes to the backend via multipart form.
 */
async function uploadPdfToBackend(
  canonical_id: string,
  pdfData: Uint8Array,
  settings: ChromeExtensionSettings,
): Promise<boolean> {
  const baseUrl = getActiveUrl(settings);
  const encodedId = encodeURIComponent(canonical_id);
  const url = `${baseUrl}/api/v1/library/papers/${encodedId}/pdf`;

  const formData = new FormData();
  const blob = new Blob([pdfData as unknown as BlobPart], { type: "application/pdf" });
  formData.append("pdf", blob, `${canonical_id}.pdf`);

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${settings.apiToken}`,
      },
      body: formData,
    });

    if (response.status === 201 || response.status === 409) {
      // 201 = stored, 409 = already exists — both are success
      return true;
    }

    console.warn(`[inCite PDF] Upload failed (${response.status}) for ${canonical_id}`);
    return false;
  } catch (err) {
    console.warn(`[inCite PDF] Upload error for ${canonical_id}:`, err);
    return false;
  }
}

// --- Retry queue ---

async function getRetryQueue(): Promise<RetryEntry[]> {
  const result = await chrome.storage.local.get(RETRY_STORAGE_KEY);
  return result[RETRY_STORAGE_KEY] ?? [];
}

async function saveRetryQueue(queue: RetryEntry[]): Promise<void> {
  await chrome.storage.local.set({ [RETRY_STORAGE_KEY]: queue });
}

async function addToRetryQueue(canonical_id: string, pdf_url: string): Promise<void> {
  const queue = await getRetryQueue();

  // Check if already in queue
  const existing = queue.find((e) => e.canonical_id === canonical_id);
  if (existing) {
    existing.attempts += 1;
    existing.last_attempt = Date.now();
  } else {
    queue.push({
      canonical_id,
      pdf_url,
      attempts: 1,
      last_attempt: Date.now(),
    });
  }

  await saveRetryQueue(queue);
}

/**
 * Process the retry queue. Called on service worker startup.
 * Retries failed PDF uploads with exponential backoff.
 */
export async function processRetryQueue(settings: ChromeExtensionSettings): Promise<void> {
  if (!settings.apiToken) return;

  const queue = await getRetryQueue();
  if (queue.length === 0) return;

  const now = Date.now();
  const remaining: RetryEntry[] = [];

  for (const entry of queue) {
    // Skip if max retries exceeded
    if (entry.attempts >= MAX_RETRIES) continue;

    // Skip if not enough time has passed (backoff)
    const backoff = RETRY_BACKOFF_MS * entry.attempts;
    if (now - entry.last_attempt < backoff) {
      remaining.push(entry);
      continue;
    }

    const success = await downloadAndUploadPdf(entry.canonical_id, entry.pdf_url, settings);
    if (!success) {
      // downloadAndUploadPdf already re-added to queue on failure,
      // but we need to avoid double-adding. The re-add will update
      // the existing entry's attempt count.
    }
    // If success, don't add back to remaining
  }

  await saveRetryQueue(remaining);
}
