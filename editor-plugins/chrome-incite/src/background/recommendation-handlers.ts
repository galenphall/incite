/**
 * Recommendation and citation insertion handlers.
 *
 * Handles getting recommendations from the API (both context-based and
 * direct text), health checks, and inserting citations into the active editor.
 */
import { extractContext, stripCitations, formatCitation, formatMultiCitation } from "@incite/shared";
import type { Recommendation } from "@incite/shared";
import { getClient, getActiveTab, detectEditorType } from "./client";
import { loadSettings } from "../shared/settings";
import type { GetContextMessage, ContextResponseMessage } from "../shared/messages";
import { getGDocsCursorText, getGDocsCursorContextForRecs } from "./gdocs-handlers";

// --- Get context from content script ---

async function getContextFromTab(tab: chrome.tabs.Tab): Promise<string> {
  const requestId = crypto.randomUUID();

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("Content script did not respond")), 8000);

    chrome.tabs.sendMessage(
      tab.id!,
      { type: "GET_CONTEXT", requestId } as GetContextMessage,
      (response: ContextResponseMessage) => {
        clearTimeout(timeout);
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
          return;
        }
        if (response?.error) {
          reject(new Error(response.error));
          return;
        }
        if (response?.text) {
          resolve(response.text);
        } else if (response?.fullText && response.cursorOffset !== undefined) {
          // Overleaf: extract context from full text + cursor offset
          loadSettings().then((settings) => {
            const ctx = extractContext(response.fullText!, response.cursorOffset!, settings.contextSentences);
            resolve(ctx.text);
          });
        } else {
          reject(new Error("No text selected. Select text and try again."));
        }
      }
    );
  });
}

// --- Handler implementations ---

export async function handleGetRecommendations(collectionId?: string | null) {
  const tab = await getActiveTab();
  if (!tab) return { type: "RECOMMENDATIONS_RESULT", error: "No active tab" };

  const settings = await loadSettings();
  const editorType = detectEditorType(tab.url ?? "");

  let query: string;
  let cursorSentenceIndex: number | undefined;

  if (editorType === "googledocs") {
    // Google Docs (canvas mode): use texteventtarget copy trick to get selected text.
    const cursorText = await getGDocsCursorText(tab);
    if (cursorText) {
      // User has text selected — use it directly as the query
      query = cursorText;
    } else {
      // Cursor only (no selection) — extract context around cursor via REST API
      const ctx = await getGDocsCursorContextForRecs(tab, settings.contextSentences);
      if (!ctx) {
        return {
          type: "RECOMMENDATIONS_RESULT",
          error: "Could not extract context. Place your cursor in a paragraph, or select text and try again.",
        };
      }
      query = ctx.text;
      cursorSentenceIndex = ctx.cursorSentenceIndex;
    }
  } else {
    // Overleaf and others: use existing content script approach
    query = await getContextFromTab(tab);
  }

  const stripped = stripCitations(query);

  if (!stripped || stripped.length < 10) {
    return { type: "RECOMMENDATIONS_RESULT", error: "Selected text is too short for recommendations." };
  }

  const apiClient = await getClient();
  const response = await apiClient.recommend(stripped, settings.k, settings.authorBoost, cursorSentenceIndex, collectionId);
  return { type: "RECOMMENDATIONS_RESULT", response, query: stripped };
}

export async function handleGetRecommendationsForText(text: string, collectionId?: string | null) {
  const settings = await loadSettings();
  const stripped = stripCitations(text);

  if (!stripped || stripped.length < 10) {
    return { type: "RECOMMENDATIONS_RESULT", error: "Text is too short for recommendations." };
  }

  const apiClient = await getClient();
  const response = await apiClient.recommend(stripped, settings.k, settings.authorBoost, undefined, collectionId);
  return { type: "RECOMMENDATIONS_RESULT", response };
}

export async function handleCheckHealth() {
  try {
    const apiClient = await getClient();
    const response = await apiClient.health();
    return { type: "HEALTH_RESULT", response };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return { type: "HEALTH_RESULT", error: message };
  }
}

export async function handleInsertCitation(rec: Recommendation) {
  const tab = await getActiveTab();
  if (!tab?.id) return { type: "INSERT_RESULT", success: false };

  const settings = await loadSettings();
  const editorType = detectEditorType(tab.url ?? "");

  const template =
    editorType === "overleaf" ? settings.overleafCitationFormat : settings.googleDocsCitationFormat;
  const citation = formatCitation(rec, template);

  return new Promise((resolve) => {
    chrome.tabs.sendMessage(
      tab.id!,
      { type: "INSERT_CITATION", citation, editorType },
      (response) => {
        if (chrome.runtime.lastError) {
          resolve({ type: "INSERT_RESULT", success: false });
          return;
        }
        resolve(response ?? { type: "INSERT_RESULT", success: true });
      }
    );
  });
}

export async function handleInsertMultiCitation(recs: Recommendation[]) {
  const tab = await getActiveTab();
  if (!tab?.id) return { type: "INSERT_RESULT", success: false };

  const settings = await loadSettings();
  const editorType = detectEditorType(tab.url ?? "");

  const template =
    editorType === "overleaf" ? settings.overleafCitationFormat : settings.googleDocsCitationFormat;
  const citation = formatMultiCitation(recs, template);

  return new Promise((resolve) => {
    chrome.tabs.sendMessage(
      tab.id!,
      { type: "INSERT_CITATION", citation, editorType },
      (response) => {
        if (chrome.runtime.lastError) {
          resolve({ type: "INSERT_RESULT", success: false });
          return;
        }
        resolve(response ?? { type: "INSERT_RESULT", success: true });
      }
    );
  });
}
