/**
 * Service worker entry point.
 *
 * This is the slim orchestrator that wires together:
 * - Message routing (delegated to message-router.ts)
 * - Side panel lifecycle (setPanelBehavior, onCommand)
 * - Tab mode switching (writing sites → side panel, academic sites → popup)
 * - Context menu setup and click handling
 * - Cloud settings sync and retry queue on startup
 *
 * All message handling logic lives in the handler modules:
 * - recommendation-handlers.ts — recommendations, health, citation insertion
 * - library-handlers.ts — save papers, collections, tags, badges, PDF uploads
 * - detection-handlers.ts — paper detection on academic pages
 * - gdocs-handlers.ts — Google Docs REST API operations
 */
import type { PaperMetadata } from "../translators/types";
import { isAcademicSite } from "../translators/registry";
import { loadSettings, syncFromCloud } from "../shared/settings";
import { processRetryQueue } from "../shared/pdf-uploader";
import { installMessageRouter } from "./message-router";
import { injectHotkeyListener } from "./hotkey";
import { detectedPapers } from "./state";
import { handleGetDetectedPapers } from "./detection-handlers";
import { quickSavePapers, resolveMetadataFromUrl } from "./library-handlers";

// --- Install message router ---

installMessageRouter();

// --- Side Panel lifecycle ---

// On writing sites the popup is cleared and panel is enabled, so clicking opens the panel.
// On other sites the popup is set, which takes precedence.
chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });

chrome.commands.onCommand.addListener(async (command) => {
  if (command === "trigger-recommendations") {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab?.id) {
      try {
        // On writing sites, toggle the overlay command palette
        await chrome.tabs.sendMessage(tab.id, { type: "TOGGLE_COMMAND_PALETTE" });
      } catch {
        // Overlay not injected (non-writing page) — fall back to side panel
        await chrome.sidePanel.open({ tabId: tab.id });
        await sendHotkeyTriggerWithRetry();
      }
    }
  }

  if (command === "save-to-library") {
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
});

/**
 * Send TRIGGER_FROM_HOTKEY to the panel with retries.
 * The panel may not be loaded yet when the side panel first opens.
 */
async function sendHotkeyTriggerWithRetry(maxAttempts = 3, intervalMs = 200): Promise<void> {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      const response = await chrome.runtime.sendMessage({ type: "TRIGGER_FROM_HOTKEY" });
      if (response?.ack) return; // Panel acknowledged
    } catch {
      // Panel not ready yet
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  // All attempts failed — panel might not be loaded. It will trigger on its own if needed.
  console.error("sendHotkeyTriggerWithRetry: all attempts failed — panel not responding");
}

// --- Context-aware toolbar: writing sites → side panel, academic sites → popup ---

/**
 * Configure side panel / popup for a tab based on its URL.
 *
 * Writing sites (Google Docs, Overleaf) open the side panel for recommendations.
 * Academic sites (arXiv, PubMed, etc.) and all other pages show the popup for
 * save-to-library. The key mechanism: setting popup to "" (empty string) makes
 * the toolbar click open the side panel instead; setting a popup path makes
 * the popup take precedence over the side panel.
 */
async function configureTabMode(tabId: number, url: string): Promise<void> {
  const isWriting =
    url.includes("docs.google.com/document") ||
    url.includes("overleaf.com/project");
  const isAcademic = isAcademicSite(url);

  if (isWriting) {
    await chrome.sidePanel.setOptions({ tabId, path: "panel/panel.html", enabled: true });
    await chrome.action.setPopup({ tabId, popup: "" });
  } else if (isAcademic) {
    await chrome.sidePanel.setOptions({ tabId, enabled: false });
    await chrome.action.setPopup({ tabId, popup: "popup/popup.html" });
  } else {
    await chrome.sidePanel.setOptions({ tabId, enabled: false });
    await chrome.action.setPopup({ tabId, popup: "popup/popup.html" });
  }
}

chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (!tab.url) return;

  try {
    // Inject the configurable hotkey listener when a page finishes loading
    if (changeInfo.status === "complete") {
      await injectHotkeyListener(tabId);
    }

    await configureTabMode(tabId, tab.url);

    // Clear cached detection when navigating away
    if (changeInfo.url) {
      detectedPapers.delete(tabId);
      await chrome.action.setBadgeText({ tabId, text: "" });
    }
  } catch {
    // Tab may have been closed between event and handler execution
  }
});

// Also configure when switching to an already-loaded tab (e.g. after extension refresh)
chrome.tabs.onActivated.addListener(async ({ tabId }) => {
  try {
    const tab = await chrome.tabs.get(tabId);
    if (tab.url) await configureTabMode(tabId, tab.url);
  } catch {
    // Tab may have been closed
  }
});

// Configure all existing tabs on startup (covers extension install/refresh)
chrome.tabs.query({}).then((tabs) => {
  for (const tab of tabs) {
    if (tab.id && tab.url) configureTabMode(tab.id, tab.url);
  }
});

// Clean up when tab is closed
chrome.tabs.onRemoved.addListener((tabId) => {
  detectedPapers.delete(tabId);
});

// --- Context menu setup ---

// Context menus must be (re)created on onInstalled because Chrome clears them
// on extension install, update, and Chrome update. This is the only event where
// contextMenus.create is guaranteed to succeed without "duplicate ID" errors.
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "save-page-to-incite",
    title: "Save this page to inCite",
    contexts: ["page"],
  });
  chrome.contextMenus.create({
    id: "save-link-to-incite",
    title: "Save link to inCite",
    contexts: ["link"],
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "save-page-to-incite") {
    const tabId = tab?.id;
    if (!tabId) return;

    const cached = detectedPapers.get(tabId);
    let papers: PaperMetadata[] = [];

    if (cached) {
      papers = cached.papers;
    } else {
      const result = await handleGetDetectedPapers();
      papers = result.papers ?? [];
    }

    if (papers.length > 0) {
      await quickSavePapers(papers);
    } else {
      await chrome.action.setBadgeText({ tabId, text: "!" });
      await chrome.action.setBadgeBackgroundColor({ tabId, color: "#E74C3C" });
      setTimeout(() => chrome.action.setBadgeText({ tabId, text: "" }), 2000);
    }
  }

  if (info.menuItemId === "save-link-to-incite") {
    const linkUrl = info.linkUrl;
    if (!linkUrl) return;

    const paper = await resolveMetadataFromUrl(linkUrl);
    if (paper) {
      await quickSavePapers([paper]);
    } else {
      const tabId = tab?.id;
      if (tabId) {
        await chrome.action.setBadgeText({ tabId, text: "!" });
        await chrome.action.setBadgeBackgroundColor({ tabId, color: "#E74C3C" });
        setTimeout(() => chrome.action.setBadgeText({ tabId, text: "" }), 2000);
      }
    }
  }
});

// --- Cloud settings sync on startup ---

syncFromCloud().catch(() => {});

// Re-sync when apiToken changes
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "sync" && changes.incite_settings?.newValue?.apiToken) {
    syncFromCloud().catch(() => {});
  }
});

// --- Retry failed PDF uploads on startup ---

loadSettings().then((s) => processRetryQueue(s)).catch(() => {});
