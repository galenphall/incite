/**
 * Message router for the service worker.
 *
 * Installs the chrome.runtime.onMessage listener and routes incoming
 * messages to domain-specific handler modules. Synchronous messages
 * (PANEL_READY, SAVE_PAPERS_HOTKEY, PAGE_PAPERS_DETECTED) are handled
 * inline; all others are dispatched through routeMessage().
 */
import type { Recommendation } from "@incite/shared";
import { setPanelReady, detectedPapers } from "./state";
import type { ExtendedPanelMessage, LibraryMessage } from "../shared/messages";
import { loadSettings, saveSettings } from "../shared/settings";

// Domain-specific handlers
import {
  handleGetRecommendations,
  handleGetRecommendationsForText,
  handleCheckHealth,
  handleInsertCitation,
  handleInsertMultiCitation,
} from "./recommendation-handlers";
import {
  handleSavePapers,
  handleCheckLibrary,
  handleGetCollections,
  handleSearchTags,
  handleUpdateLibraryItem,
  handleSavePapersHotkey,
  updateBadge,
} from "./library-handlers";
import { handleGetDetectedPapers } from "./detection-handlers";
import {
  handleGDocsInsertCitation,
  handleGDocsInsertMultiCitation,
  handleGDocsInsertBibliography,
  handleGDocsScanCitations,
  handleGDocsClean,
  handleGDocsRefreshCitations,
} from "./gdocs-handlers";

/**
 * Route an async message to the appropriate handler.
 * Called for all messages not handled synchronously in the listener.
 */
async function routeMessage(message: ExtendedPanelMessage, _sender: chrome.runtime.MessageSender) {
  switch (message.type) {
    case "GET_RECOMMENDATIONS":
      return await handleGetRecommendations((message as { type: string; collectionId?: string | null }).collectionId);
    case "GET_RECOMMENDATIONS_FOR_TEXT":
      return await handleGetRecommendationsForText((message as { type: string; text: string; collectionId?: string | null }).text, (message as { type: string; text: string; collectionId?: string | null }).collectionId);
    case "CHECK_HEALTH":
      return await handleCheckHealth();
    case "GET_SETTINGS":
      return { type: "SETTINGS_RESULT", settings: await loadSettings() };
    case "SAVE_SETTINGS":
      return { type: "SETTINGS_RESULT", settings: await saveSettings(message.settings) };
    case "INSERT_CITATION_REQUEST":
      return await handleInsertCitation(message.recommendation);
    case "INSERT_MULTI_CITATION_REQUEST":
      return await handleInsertMultiCitation((message as { type: string; recommendations: Recommendation[] }).recommendations);

    // --- Save to Library messages ---
    case "GET_DETECTED_PAPERS":
      return await handleGetDetectedPapers();
    case "SAVE_PAPERS":
      return await handleSavePapers(message as LibraryMessage & { type: "SAVE_PAPERS" });
    case "CHECK_LIBRARY":
      return await handleCheckLibrary(message as LibraryMessage & { type: "CHECK_LIBRARY" });
    case "GET_COLLECTIONS":
      return await handleGetCollections();
    case "SEARCH_TAGS":
      return await handleSearchTags(message as LibraryMessage & { type: "SEARCH_TAGS" });
    case "UPDATE_LIBRARY_ITEM":
      return await handleUpdateLibraryItem(message as LibraryMessage & { type: "UPDATE_LIBRARY_ITEM" });

    // --- Google Docs REST API messages ---
    case "GDOCS_INSERT_CITATION":
      return await handleGDocsInsertCitation(message as { type: string; paperId: string; text: string; paperUrl: string });
    case "GDOCS_INSERT_MULTI_CITATION":
      return await handleGDocsInsertMultiCitation(message as { type: string; fullText: string; segments: { text: string; paperUrl: string; paperId: string; offsetInFullText: number }[] });
    case "GDOCS_INSERT_BIBLIOGRAPHY":
      return await handleGDocsInsertBibliography(message as { type: string; entries: { paperId: string; formatted: string; url?: string }[] });
    case "GDOCS_SCAN_CITATIONS":
      return await handleGDocsScanCitations(message as { type: string; trackedPaperIds: string[] });
    case "GDOCS_CLEAN":
      return await handleGDocsClean();
    case "GDOCS_REFRESH_CITATIONS":
      return await handleGDocsRefreshCitations(message as { type: string; trackedPaperIds: string[]; refreshText: boolean });

    default:
      return { error: "Unknown message type" };
  }
}

/**
 * Install the chrome.runtime.onMessage listener.
 * Must be called exactly once at service worker startup.
 */
export function installMessageRouter(): void {
  chrome.runtime.onMessage.addListener((message: ExtendedPanelMessage, sender, sendResponse) => {
    // --- Synchronous / fire-and-forget messages ---
    //
    // These are handled inline rather than routed through routeMessage() because:
    // - PANEL_READY: Must respond synchronously (return false) so the panel knows
    //   the service worker acknowledged it before proceeding with initialization.
    // - SAVE_PAPERS_HOTKEY: Needs to respond quickly to the injected keydown listener
    //   to avoid Chrome's "message port closed" warning on slow async chains.
    // - PAGE_PAPERS_DETECTED: Fire-and-forget from content scripts — we just cache
    //   the result and update the badge, no async work needed.

    if (message.type === "PANEL_READY") {
      setPanelReady(true);
      sendResponse({ ack: true });
      return false;
    }

    // Open the Chrome side panel (Mode D) — requested by the overlay rail's expand button
    if (message.type === "OPEN_SIDE_PANEL") {
      chrome.tabs.query({ active: true, currentWindow: true }).then(([tab]) => {
        if (tab?.id) chrome.sidePanel.open({ tabId: tab.id });
      });
      sendResponse({ ack: true });
      return false;
    }

    // Handle SAVE_PAPERS_HOTKEY from injected keydown listener
    if (message.type === "SAVE_PAPERS_HOTKEY") {
      handleSavePapersHotkey().then(() => sendResponse({ ack: true })).catch(() => sendResponse({ ack: false }));
      return true;
    }

    // Handle PAGE_PAPERS_DETECTED from content scripts
    if (message.type === "PAGE_PAPERS_DETECTED") {
      const tabId = sender.tab?.id;
      if (tabId !== undefined) {
        const msg = message as LibraryMessage & { type: "PAGE_PAPERS_DETECTED" };
        detectedPapers.set(tabId, {
          type: msg.detection.type,
          papers: msg.papers ?? [],
          translatorName: msg.translatorName,
        });
        updateBadge(tabId, msg.detection.type, msg.papers ?? []);
      }
      sendResponse({ ack: true });
      return false;
    }

    // --- Async messages (return true = keep channel open) ---

    routeMessage(message, sender).then(sendResponse).catch((err) => {
      sendResponse({ error: err instanceof Error ? err.message : String(err) });
    });
    return true;
  });
}
