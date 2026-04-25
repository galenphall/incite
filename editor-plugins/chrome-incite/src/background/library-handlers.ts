/**
 * Library management handlers.
 *
 * Handles saving papers, checking library status, managing collections/tags,
 * badge updates, PDF uploads, and URL-based metadata resolution.
 */
import { getActiveUrl } from "@incite/shared";
import type { PaperMetadata } from "../translators/types";
import type { ChromeExtensionSettings } from "../shared/types";
import { getClient, getActiveTab, configFromSettings } from "./client";
import { detectedPapers } from "./state";
import { loadSettings } from "../shared/settings";
import { downloadAndUploadPdf } from "../shared/pdf-uploader";
import { handleGetDetectedPapers } from "./detection-handlers";

// --- Badge updates ---

export async function updateBadge(tabId: number, type: "single" | "multiple", papers: PaperMetadata[]) {
  // Check if paper is already in library
  const settings = await loadSettings();
  if (settings.apiToken && type === "single" && papers.length === 1) {
    try {
      const apiClient = await getClient();
      const checkPapers = papers.map((p) => ({ doi: p.doi ?? null, title: p.title }));
      const results = await apiClient.checkLibrary(checkPapers);
      if (results?.[0]?.in_library) {
        await chrome.action.setBadgeText({ tabId, text: "\u2713" });
        await chrome.action.setBadgeBackgroundColor({ tabId, color: "#2ECC71" });
        return;
      }
    } catch {
      // Ignore check errors, show detection badge instead
    }
  }

  if (type === "single") {
    await chrome.action.setBadgeText({ tabId, text: "1" });
    await chrome.action.setBadgeBackgroundColor({ tabId, color: "#4A90D9" });
  } else {
    await chrome.action.setBadgeText({ tabId, text: "+" });
    await chrome.action.setBadgeBackgroundColor({ tabId, color: "#4A90D9" });
  }
}

// --- PDF upload after save (fire-and-forget) ---

export function firePdfUploads(
  papers: PaperMetadata[],
  saveResult: { saved?: Array<{ canonical_id: string; title: string }>; already_existed?: Array<{ canonical_id: string; title: string }> },
  settings: ChromeExtensionSettings,
): void {
  const savedItems = saveResult.saved ?? [];
  for (const item of savedItems) {
    const paper = papers.find((p) => p.title?.trim() === item.title);
    if (paper?.pdf_url) {
      downloadAndUploadPdf(item.canonical_id, paper.pdf_url, settings).catch(() => {});
    }
  }
}

// --- Quick-save helpers (shared by keyboard shortcut and context menu) ---

export async function quickSavePapers(papers: PaperMetadata[]): Promise<boolean> {
  const settings = await loadSettings();
  if (!settings.apiToken) return false;

  const tab = await getActiveTab();
  const tabId = tab?.id;

  try {
    const apiClient = await getClient();
    const stored = await chrome.storage.local.get("lastCollectionId");
    const collectionId = stored.lastCollectionId ?? null;
    const result = await apiClient.savePapers({
      papers,
      collection_id: collectionId,
      tags: [],
      enrich: true,
    });

    // Fire-and-forget PDF uploads
    firePdfUploads(papers, result, settings);

    if (tabId) {
      await chrome.action.setBadgeText({ tabId, text: "\u2713" });
      await chrome.action.setBadgeBackgroundColor({ tabId, color: "#2ECC71" });
    }
    return true;
  } catch {
    if (tabId) {
      await chrome.action.setBadgeText({ tabId, text: "!" });
      await chrome.action.setBadgeBackgroundColor({ tabId, color: "#E74C3C" });
      setTimeout(() => chrome.action.setBadgeText({ tabId, text: "" }), 2000);
    }
    return false;
  }
}

export async function resolveMetadataFromUrl(url: string): Promise<PaperMetadata | null> {
  let s2Url: string | null = null;

  // Check for DOI
  const doiMatch = url.match(/doi\.org\/(.+)/);
  if (doiMatch) {
    const doi = decodeURIComponent(doiMatch[1]).replace(/\/$/, "");
    s2Url = `https://api.semanticscholar.org/graph/v1/paper/DOI:${encodeURIComponent(doi)}?fields=title,abstract,authors,year,venue,externalIds`;
  }

  // Check for arXiv
  if (!s2Url) {
    const arxivMatch = url.match(/arxiv\.org\/(?:abs|pdf)\/([0-9]+\.[0-9]+(?:v\d+)?)/);
    if (arxivMatch) {
      const arxivId = arxivMatch[1];
      s2Url = `https://api.semanticscholar.org/graph/v1/paper/ARXIV:${arxivId}?fields=title,abstract,authors,year,venue,externalIds`;
    }
  }

  if (!s2Url) return null;

  try {
    const response = await fetch(s2Url);
    if (!response.ok) return null;
    const data = await response.json();

    return {
      title: data.title,
      abstract: data.abstract ?? undefined,
      authors: data.authors?.map((a: { name: string }) => a.name),
      year: data.year ?? undefined,
      doi: data.externalIds?.DOI ?? undefined,
      arxiv_id: data.externalIds?.ArXiv ?? undefined,
      journal: data.venue ?? undefined,
      url,
    };
  } catch {
    return null;
  }
}

// --- Handler implementations ---

/** Handle the custom hotkey for saving papers (same flow as save-to-library command). */
export async function handleSavePapersHotkey(): Promise<void> {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) return;

  const cached = detectedPapers.get(tab.id);
  let papers: PaperMetadata[] = [];

  if (cached && cached.type === "single") {
    papers = cached.papers;
  } else if (!cached) {
    const result = await handleGetDetectedPapers();
    if (result.type === "single") {
      papers = result.papers ?? [];
    }
  }

  if (papers.length > 0) {
    await quickSavePapers(papers);
  } else {
    await chrome.action.setBadgeText({ tabId: tab.id, text: "!" });
    await chrome.action.setBadgeBackgroundColor({ tabId: tab.id, color: "#E74C3C" });
    setTimeout(() => chrome.action.setBadgeText({ tabId: tab.id, text: "" }), 2000);
  }
}

export async function handleSavePapers(message: { papers: PaperMetadata[]; collectionId?: string | null; tags?: string[]; enrich?: boolean }) {
  const settings = await loadSettings();
  if (!settings.apiToken) return { error: "Not signed in" };

  const apiClient = await getClient();
  const result = await apiClient.savePapers({
    papers: message.papers,
    collection_id: message.collectionId ?? null,
    tags: message.tags ?? [],
    enrich: message.enrich ?? true,
  });

  // Update badge to checkmark on the active tab
  const tab = await getActiveTab();
  if (tab?.id) {
    await chrome.action.setBadgeText({ tabId: tab.id, text: "\u2713" });
    await chrome.action.setBadgeBackgroundColor({ tabId: tab.id, color: "#2ECC71" });
  }

  // Fire-and-forget PDF uploads for saved papers with pdf_url
  firePdfUploads(message.papers, result, settings);

  return result;
}

export async function handleCheckLibrary(message: { papers: PaperMetadata[] }) {
  const settings = await loadSettings();
  if (!settings.apiToken) return { results: [] };

  const apiClient = await getClient();
  const checkPapers = message.papers.map((p) => ({ doi: p.doi ?? null, title: p.title }));
  const results = await apiClient.checkLibrary(checkPapers);
  return { results: results ?? [] };
}

export async function handleGetCollections() {
  const settings = await loadSettings();
  if (!settings.apiToken) return { collections: [] };

  const apiClient = await getClient();
  const collections = await apiClient.getCollections();
  return { collections };
}

export async function handleSearchTags(message: { query: string }) {
  const settings = await loadSettings();
  if (!settings.apiToken) return { tags: [] };

  const apiClient = await getClient();
  const tags = await apiClient.searchTags(message.query);
  return { tags };
}

export async function handleUpdateLibraryItem(message: { canonicalId: string; collectionId?: string | null; tags?: string[] }) {
  const settings = await loadSettings();
  if (!settings.apiToken) return { error: "Not signed in" };

  return await apiUpdateLibraryItem(message.canonicalId, message.collectionId, message.tags, settings);
}

/** Update a library item — not yet in shared client, so use direct fetch. */
export async function apiUpdateLibraryItem(
  canonicalId: string,
  collectionId: string | null | undefined,
  tags: string[] | undefined,
  settings: ChromeExtensionSettings,
) {
  const baseUrl = getActiveUrl(settings);
  const encodedId = encodeURIComponent(canonicalId);

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "application/json",
  };
  if (settings.apiMode === "cloud" && settings.apiToken) {
    headers["Authorization"] = `Bearer ${settings.apiToken}`;
  }

  const response = await fetch(`${baseUrl}/api/v1/library/papers/${encodedId}/update`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      collection_id: collectionId ?? null,
      tags: tags ?? [],
    }),
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`Update failed (${response.status}): ${text || response.statusText}`);
  }

  return response.json();
}
