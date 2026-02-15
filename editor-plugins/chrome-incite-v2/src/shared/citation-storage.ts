import type { CitationStorage, TrackedCitation } from "@incite/shared";

/** Extract a document ID from a URL for keying citation storage. */
function extractDocId(url: string): string | null {
  // Google Docs: docs.google.com/document/d/{ID}/...
  const gdocsMatch = url.match(/docs\.google\.com\/document\/d\/([^/]+)/);
  if (gdocsMatch) return `gdocs:${gdocsMatch[1]}`;

  // Overleaf: overleaf.com/project/{ID}
  const overleafMatch = url.match(/overleaf\.com\/project\/([^/?#]+)/);
  if (overleafMatch) return `overleaf:${overleafMatch[1]}`;

  return null;
}

/** CitationStorage backed by chrome.storage.local, keyed by document ID. */
export class ChromeCitationStorage implements CitationStorage {
  private readonly prefix = "incite_citations_";

  async load(docKey: string): Promise<TrackedCitation[]> {
    const key = this.prefix + docKey;
    const result = await chrome.storage.local.get(key);
    return (result[key] as TrackedCitation[]) ?? [];
  }

  async save(docKey: string, citations: TrackedCitation[]): Promise<void> {
    const key = this.prefix + docKey;
    await chrome.storage.local.set({ [key]: citations });
  }
}

/** Get the document key for the current active tab. Returns null if unsupported. */
export async function getDocKeyFromActiveTab(): Promise<string | null> {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.url) return null;
  return extractDocId(tab.url);
}
