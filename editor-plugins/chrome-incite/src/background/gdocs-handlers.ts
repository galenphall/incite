/**
 * Google Docs REST API handlers.
 *
 * Handles citation insertion, bibliography management, citation scanning,
 * and document cleanup via the Google Docs REST API. Uses the Zotero-style
 * placeholder insertion approach for cursor-position resolution.
 */
import { InCiteClient } from "@incite/shared";
import { getActiveTab } from "./client";
import { gdocsApi, gdocsCursorCache } from "./state";
import { loadSettings } from "../shared/settings";
import { extractDocId } from "../shared/gdocs-api";
import type { DocsDocument, DocsRequest } from "../shared/gdocs-api";
import { resolveDocumentIndex, extractFullText } from "../shared/gdocs-index-resolver";
import type { CursorContext } from "../shared/gdocs-index-resolver";
import { GDocsCitationInserter } from "../shared/gdocs-citation-inserter";
import type { CursorContextResponseMessage } from "../shared/messages";

/** Get the current Google Doc ID from the active tab URL. */
export function getActiveDocId(tabUrl: string): string | null {
  return extractDocId(tabUrl);
}

/** Get a GDocsCitationInserter for the active document. */
export function getInserter(docId: string): GDocsCitationInserter {
  return new GDocsCitationInserter(gdocsApi, docId);
}

/**
 * Get selected text from Google Docs via the texteventtarget copy trick.
 * Returns the selected text directly, or null if nothing is selected.
 */
export async function getGDocsCursorText(tab: chrome.tabs.Tab): Promise<string | null> {
  if (!tab.id) return null;

  try {
    const response = await new Promise<CursorContextResponseMessage>((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error("Cursor context timeout")), 8000);
      chrome.tabs.sendMessage(
        tab.id!,
        { type: "GET_CURSOR_CONTEXT", requestId: crypto.randomUUID() },
        (resp: CursorContextResponseMessage) => {
          clearTimeout(timeout);
          if (chrome.runtime.lastError) {
            reject(new Error(chrome.runtime.lastError.message));
            return;
          }
          resolve(resp);
        }
      );
    });

    if (response?.paragraphText && response.paragraphText.trim().length > 0) {
      return response.paragraphText.trim();
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Get cursor context from the Google Docs content script via texteventtarget probing,
 * then resolve it to a document index using the REST API.
 */
export async function getGDocsCursorIndex(tab: chrome.tabs.Tab): Promise<{ index: number; fullText: string } | null> {
  const docId = getActiveDocId(tab.url ?? "");
  if (!docId || !tab.id) return null;

  // Ask content script for cursor context
  const cursorContext = await new Promise<CursorContext | null>((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("Cursor context timeout")), 8000);
    const requestId = crypto.randomUUID();

    chrome.tabs.sendMessage(
      tab.id!,
      { type: "GET_CURSOR_CONTEXT", requestId },
      (response: CursorContextResponseMessage) => {
        clearTimeout(timeout);
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
          return;
        }
        if (response?.error || !response?.paragraphText) {
          resolve(null);
          return;
        }
        resolve({
          paragraphText: response.paragraphText,
          cursorOffset: response.cursorOffset ?? 0,
          textBefore: response.textBefore,
          textAfter: response.textAfter,
        });
      }
    );
  });

  // Fetch document via REST API
  const doc = await gdocsApi.getDocument(docId);
  const fullText = extractFullText(doc);

  if (cursorContext) {
    // Resolve cursor context to document index
    const resolved = resolveDocumentIndex(doc, cursorContext);
    if (resolved) {
      gdocsCursorCache.set(docId, { index: resolved.index, timestamp: Date.now() });
      return { index: resolved.index, fullText };
    }
  }

  // Cursor probing failed — return null so caller falls back to content script
  return null;
}

/**
 * Insert a placeholder at the cursor via the content script, then find it
 * in the document via REST API and return its character index.
 *
 * This is the Zotero approach: the browser-side content script inserts a
 * placeholder at the live cursor (which the REST API can't access), then
 * we find it in the document structure and replace it with the real content.
 *
 * The full pipeline:
 *   0. Capture selected text (the paste in step 1 will replace it)
 *   1. Content script pastes placeholder at cursor via synthetic paste event
 *   2. Fetch document via REST API and find the placeholder string
 *   3. Delete placeholder via REST API and restore the original selected text
 *   4. Return the resolved document index for the caller to insert content at
 *
 * Why not just use the REST API cursor position? The REST API has no concept of
 * a "cursor" — it only knows document indices. The live cursor exists only in
 * the browser's canvas rendering, accessible only through the texteventtarget.
 */
export async function insertPlaceholderAndLocate(
  tab: chrome.tabs.Tab,
  docId: string
): Promise<{ index: number; placeholder: string; replacedText: string }> {
  // Use Unicode brackets ⟦⟧ to avoid collisions with document content
  const placeholder = `\u27E6INCITE-${crypto.randomUUID().slice(0, 8)}\u27E7`;

  // Step 0: Capture the currently selected text before the paste replaces it.
  // If the user has text selected, the synthetic paste will overwrite it.
  // We need to save it so we can restore it after locating the placeholder.
  const selectedText = await getGDocsCursorText(tab) ?? "";

  // Step 1: Ask content script to paste the placeholder at the cursor.
  // The content script dispatches a synthetic ClipboardEvent("paste") on the
  // texteventtarget iframe — this is the same mechanism Google Docs uses for
  // real paste operations.
  const result = await new Promise<{ success: boolean }>((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("Placeholder insertion timeout")), 5000);
    chrome.tabs.sendMessage(
      tab.id!,
      { type: "INSERT_PLACEHOLDER", placeholder },
      (response) => {
        clearTimeout(timeout);
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
          return;
        }
        resolve(response ?? { success: false });
      }
    );
  });

  if (!result.success) {
    throw new Error("Content script could not insert placeholder at cursor");
  }

  // Step 2: Fetch the document via REST API and find the placeholder.
  // The placeholder is now physically in the document (visible to the user
  // for a brief moment). We need to find its position in the document's
  // structural element tree to get a valid batchUpdate index.
  const doc = await gdocsApi.getDocument(docId);
  const fullText = extractFullText(doc);
  const placeholderPos = fullText.indexOf(placeholder);

  if (placeholderPos === -1) {
    throw new Error("Placeholder not found in document — paste may not have worked");
  }

  // Convert text position to document index by scanning structural elements.
  // The REST API uses element-based indices (which count structural elements),
  // not plain-text character offsets.
  const docIndex = textOffsetToDocIndex(doc, placeholderPos);

  // Step 3: Delete the placeholder and restore the original selected text.
  // This leaves the document in its original state, but now we know exactly
  // where the cursor was in REST API index terms.
  const requests: DocsRequest[] = [
    { deleteContentRange: { range: { startIndex: docIndex, endIndex: docIndex + placeholder.length } } },
  ];
  if (selectedText) {
    // Re-insert the text that the paste replaced
    requests.push({ insertText: { location: { index: docIndex }, text: selectedText } });
  }
  await gdocsApi.batchUpdate(docId, requests);

  // The insertion point is after the restored text
  const insertAt = docIndex + selectedText.length;
  return { index: insertAt, placeholder, replacedText: selectedText };
}

/**
 * Convert a character offset in the concatenated plain text to a Google Docs
 * document index (the index used in batchUpdate requests).
 */
export function textOffsetToDocIndex(doc: DocsDocument, textOffset: number): number {
  let charsSeen = 0;
  for (const element of doc.body.content) {
    if (!element.paragraph) continue;
    for (const pe of element.paragraph.elements) {
      if (!pe.textRun?.content) continue;
      const len = pe.textRun.content.length;
      if (charsSeen + len > textOffset) {
        return pe.startIndex + (textOffset - charsSeen);
      }
      charsSeen += len;
    }
  }
  // Fallback: end of document body
  const lastElement = doc.body.content[doc.body.content.length - 1];
  return lastElement?.endIndex ?? 1;
}

export async function handleGDocsInsertCitation(message: { paperId: string; text: string; paperUrl: string }) {
  try {
    const tab = await getActiveTab();
    if (!tab?.url) return { type: "GDOCS_RESULT", success: false, error: "No active tab" };

    const docId = getActiveDocId(tab.url);
    if (!docId) return { type: "GDOCS_RESULT", success: false, error: "Not a Google Doc" };

    // Zotero-style: insert placeholder at cursor, find it via REST API, replace with citation
    const { index: insertIndex } = await insertPlaceholderAndLocate(tab, docId);

    const inserter = getInserter(docId);
    const { endIndex } = await inserter.insertCitation(insertIndex, message.text, message.paperUrl, message.paperId);

    // Cache position for subsequent insertions
    gdocsCursorCache.set(docId, { index: endIndex, timestamp: Date.now() });

    return { type: "GDOCS_RESULT", success: true };
  } catch (err: unknown) {
    const errMsg = err instanceof Error ? err.message : String(err);
    return { type: "GDOCS_RESULT", success: false, error: errMsg };
  }
}

export async function handleGDocsInsertMultiCitation(message: { fullText: string; segments: { text: string; paperUrl: string; paperId: string; offsetInFullText: number }[] }) {
  try {
    const tab = await getActiveTab();
    if (!tab?.url) return { type: "GDOCS_RESULT", success: false, error: "No active tab" };

    const docId = getActiveDocId(tab.url);
    if (!docId) return { type: "GDOCS_RESULT", success: false, error: "Not a Google Doc" };

    // Zotero-style: insert placeholder at cursor, find it via REST API, replace with citation
    const { index: insertIndex } = await insertPlaceholderAndLocate(tab, docId);

    const inserter = getInserter(docId);
    const { endIndex } = await inserter.insertGroupedCitation(insertIndex, message.fullText, message.segments);

    gdocsCursorCache.set(docId, { index: endIndex, timestamp: Date.now() });

    return { type: "GDOCS_RESULT", success: true };
  } catch (err: unknown) {
    const errMsg = err instanceof Error ? err.message : String(err);
    return { type: "GDOCS_RESULT", success: false, error: errMsg };
  }
}

export async function handleGDocsInsertBibliography(message: { entries: { paperId: string; formatted: string; url?: string }[] }) {
  try {
    const tab = await getActiveTab();
    if (!tab?.url) return { type: "GDOCS_RESULT", success: false, error: "No active tab" };

    const docId = getActiveDocId(tab.url);
    if (!docId) return { type: "GDOCS_RESULT", success: false, error: "Not a Google Doc" };

    const inserter = getInserter(docId);
    await inserter.insertBibliography(message.entries);
    return { type: "GDOCS_RESULT", success: true };
  } catch (err: unknown) {
    const errMsg = err instanceof Error ? err.message : String(err);
    return { type: "GDOCS_RESULT", success: false, error: errMsg };
  }
}

export async function handleGDocsScanCitations(message: { trackedPaperIds: string[] }) {
  try {
    const tab = await getActiveTab();
    if (!tab?.url) return { type: "GDOCS_RESULT", success: false, error: "No active tab" };

    const docId = getActiveDocId(tab.url);
    if (!docId) return { type: "GDOCS_RESULT", success: false, error: "Not a Google Doc" };

    const inserter = getInserter(docId);
    const data = await inserter.scanCitations(message.trackedPaperIds);
    return { type: "GDOCS_RESULT", success: true, data };
  } catch (err: unknown) {
    const errMsg = err instanceof Error ? err.message : String(err);
    return { type: "GDOCS_RESULT", success: false, error: errMsg };
  }
}

export async function handleGDocsClean() {
  try {
    const tab = await getActiveTab();
    if (!tab?.url) return { type: "GDOCS_RESULT", success: false, error: "No active tab" };

    const docId = getActiveDocId(tab.url);
    if (!docId) return { type: "GDOCS_RESULT", success: false, error: "Not a Google Doc" };

    const inserter = getInserter(docId);
    const data = await inserter.cleanInciteData();
    return { type: "GDOCS_RESULT", success: true, data };
  } catch (err: unknown) {
    const errMsg = err instanceof Error ? err.message : String(err);
    return { type: "GDOCS_RESULT", success: false, error: errMsg };
  }
}

export async function handleGDocsRefreshCitations(message: { trackedPaperIds: string[]; refreshText: boolean }) {
  try {
    const tab = await getActiveTab();
    if (!tab?.url) return { type: "GDOCS_RESULT", success: false, error: "No active tab" };

    const docId = getActiveDocId(tab.url);
    if (!docId) return { type: "GDOCS_RESULT", success: false, error: "Not a Google Doc" };

    const inserter = getInserter(docId);
    const trackedIds = message.trackedPaperIds ?? [];

    // Step 1: Enhanced scan with tracker IDs for reconciliation
    const scan = await inserter.scanCitationsEnhanced(trackedIds);
    const foundPaperIds = Array.from(scan.instances.keys());

    // Step 2: Fix copy-paste duplicates
    let duplicatesFixed = 0;
    if (scan.duplicateRanges.length > 0) {
      duplicatesFixed = await inserter.fixDuplicateRanges(scan.duplicateRanges);
    }

    // Step 3: Fetch metadata for all found papers (needed for both reconciliation and reformat)
    let paperMetadata: Array<{
      canonical_id: string;
      title: string;
      abstract: string;
      authors: string[];
      year: number | null;
      doi: string;
      journal: string;
    }> = [];
    let citationsRefreshed = 0;

    if (foundPaperIds.length > 0) {
      const settings = await loadSettings();
      const client = new InCiteClient({
        apiMode: settings.apiMode,
        cloudUrl: settings.cloudUrl,
        localUrl: settings.localUrl,
        apiToken: settings.apiToken ?? "",
      });
      const papers = await client.getPapers(foundPaperIds);
      paperMetadata = Array.from(papers.values());

      // Step 4: Refresh citation text if requested
      if (message.refreshText) {
        const result = await inserter.reformatCitations((paperId: string) => {
          const paper = papers.get(paperId);
          if (!paper) return null;
          const authors = paper.authors ?? [];
          let authorStr: string;
          if (authors.length === 0) authorStr = "Unknown";
          else if (authors.length === 1) authorStr = authors[0];
          else if (authors.length === 2) authorStr = `${authors[0]} & ${authors[1]}`;
          else authorStr = `${authors[0]} et al.`;
          const yearStr = paper.year ? String(paper.year) : "n.d.";
          return `(${authorStr}, ${yearStr})`;
        });
        citationsRefreshed = result.updated;
      }
    }

    return {
      type: "GDOCS_RESULT",
      success: true,
      data: {
        foundPaperIds,
        orphanedPaperIds: scan.orphaned,
        untrackedPaperIds: scan.untracked,
        paperMetadata,
        duplicatesFixed,
        citationsRefreshed,
      },
    };
  } catch (err: unknown) {
    const errMsg = err instanceof Error ? err.message : String(err);
    return { type: "GDOCS_RESULT", success: false, error: errMsg };
  }
}
