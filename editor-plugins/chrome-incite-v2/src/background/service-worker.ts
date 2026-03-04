import { extractContext, stripCitations, formatCitation, formatMultiCitation, InCiteClient, FetchTransport, getActiveUrl } from "@incite/shared";
import type { Recommendation, RecommendResponse, HealthResponse, ClientConfig } from "@incite/shared";
import type {
  ChromeExtensionSettings,
  EditorType,
  GetContextMessage,
  ContextResponseMessage,
  CursorContextResponseMessage,
  PanelMessage,
} from "../shared/types";
import { loadSettings, saveSettings, syncFromCloud } from "../shared/settings";
import { isAcademicSite } from "../translators/registry";
import { downloadAndUploadPdf, processRetryQueue } from "../shared/pdf-uploader";
import { GoogleDocsAPI, extractDocId } from "../shared/gdocs-api";
import type { DocsDocument } from "../shared/gdocs-api";
import { resolveDocumentIndex, extractFullText } from "../shared/gdocs-index-resolver";
import type { CursorContext } from "../shared/gdocs-index-resolver";
import { GDocsCitationInserter } from "../shared/gdocs-citation-inserter";

// --- Types for Save to Library ---

interface PaperMetadata {
  title: string;
  authors?: string[];
  year?: number;
  doi?: string;
  abstract?: string;
  journal?: string;
  url?: string;
  arxiv_id?: string;
  pdf_url?: string;
  full_text?: string;
  volume?: string;
  issue?: string;
  pages?: string;
  pmid?: string;
  pmcid?: string;
  issn?: string;
  publisher?: string;
  keywords?: string[];
  language?: string;
}

interface DetectedPapersState {
  type: "single" | "multiple";
  papers: PaperMetadata[];
  translatorName: string;
}

// Cache of detected papers per tab
const detectedPapers = new Map<number, DetectedPapersState>();

// --- Google Docs API singleton ---
const gdocsApi = new GoogleDocsAPI();

/** Cache of the last resolved cursor index per document, for citation insertion. */
const gdocsCursorCache = new Map<string, { index: number; timestamp: number }>();

// --- Hotkey injection into tabs ---

/**
 * Content function injected into pages to listen for the save-paper hotkey.
 * Reads the hotkey from chrome.storage.sync and sends a message on match.
 * Guarded against double-injection via a window flag.
 */
function injectedHotkeyListener() {
  const FLAG = "__incite_hotkey_injected__";
  if ((window as unknown as Record<string, boolean>)[FLAG]) return;
  (window as unknown as Record<string, boolean>)[FLAG] = true;

  const STORAGE_KEY = "incite_settings";
  let hotkeyStr = "Alt+Shift+S";

  function parseHk(s: string) {
    const parts = s.split("+").map((p) => p.trim());
    const r = { ctrl: false, alt: false, shift: false, meta: false, key: "" };
    for (const part of parts) {
      const l = part.toLowerCase();
      if (l === "ctrl" || l === "control") r.ctrl = true;
      else if (l === "alt") r.alt = true;
      else if (l === "shift") r.shift = true;
      else if (l === "meta" || l === "cmd" || l === "command") r.meta = true;
      else r.key = l;
    }
    return r;
  }

  let parsed = parseHk(hotkeyStr);

  // Load initial hotkey from storage
  chrome.storage.sync.get(STORAGE_KEY, (result) => {
    const stored = result[STORAGE_KEY];
    if (stored?.savePaperHotkey) {
      hotkeyStr = stored.savePaperHotkey;
      parsed = parseHk(hotkeyStr);
    }
  });

  // Listen for hotkey changes without re-injection
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === "sync" && changes[STORAGE_KEY]?.newValue?.savePaperHotkey) {
      hotkeyStr = changes[STORAGE_KEY].newValue.savePaperHotkey;
      parsed = parseHk(hotkeyStr);
    }
  });

  document.addEventListener("keydown", (event: KeyboardEvent) => {
    if (
      event.ctrlKey === parsed.ctrl &&
      event.altKey === parsed.alt &&
      event.shiftKey === parsed.shift &&
      event.metaKey === parsed.meta &&
      event.key.toLowerCase() === parsed.key
    ) {
      event.preventDefault();
      event.stopPropagation();
      chrome.runtime.sendMessage({ type: "SAVE_PAPERS_HOTKEY" });
    }
  }, true);
}

/** Inject the hotkey listener into a tab. */
async function injectHotkeyListener(tabId: number): Promise<void> {
  try {
    await chrome.scripting.executeScript({
      target: { tabId },
      func: injectedHotkeyListener,
    });
  } catch {
    // Injection not allowed on this page (chrome://, edge://, etc.)
  }
}

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

// --- Shared API client (created lazily from settings) ---

let client: InCiteClient | null = null;

function configFromSettings(settings: ChromeExtensionSettings): ClientConfig {
  return {
    apiMode: settings.apiMode,
    cloudUrl: settings.cloudUrl,
    localUrl: settings.localUrl,
    apiToken: settings.apiToken,
  };
}

async function getClient(): Promise<InCiteClient> {
  const settings = await loadSettings();
  if (!client) {
    client = new InCiteClient(configFromSettings(settings), new FetchTransport());
  } else {
    client.updateConfig(configFromSettings(settings));
  }
  return client;
}

// --- Side Panel lifecycle ---

// On writing sites the popup is cleared and panel is enabled, so clicking opens the panel.
// On other sites the popup is set, which takes precedence.
chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });

// Track whether the panel is ready to receive messages
let panelReady = false;

chrome.commands.onCommand.addListener(async (command) => {
  if (command === "trigger-recommendations") {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab?.id) {
      await chrome.sidePanel.open({ tabId: tab.id });
      // Try to trigger recommendations with retry until panel acknowledges
      await sendHotkeyTriggerWithRetry();
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

/** Configure side panel / popup for a tab based on its URL. */
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

// --- Message handling ---

type LibraryMessage =
  | { type: "PAGE_PAPERS_DETECTED"; detection: { type: "single" | "multiple" }; papers?: PaperMetadata[]; translatorName: string }
  | { type: "GET_DETECTED_PAPERS" }
  | { type: "SAVE_PAPERS"; papers: PaperMetadata[]; collectionId?: string | null; tags?: string[]; enrich?: boolean }
  | { type: "CHECK_LIBRARY"; papers: PaperMetadata[] }
  | { type: "GET_COLLECTIONS" }
  | { type: "SEARCH_TAGS"; query: string }
  | { type: "EXTRACT_PAPERS" }
  | { type: "UPDATE_LIBRARY_ITEM"; canonicalId: string; collectionId?: string | null; tags?: string[] };

type ExtendedPanelMessage =
  | PanelMessage
  | { type: "PANEL_READY" }
  | { type: "SAVE_PAPERS_HOTKEY" }
  | { type: "GET_RECOMMENDATIONS_FOR_TEXT"; text: string }
  | LibraryMessage;

chrome.runtime.onMessage.addListener((message: ExtendedPanelMessage, sender, sendResponse) => {
  if (message.type === "PANEL_READY") {
    panelReady = true;
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

  handleMessage(message, sender).then(sendResponse).catch((err) => {
    sendResponse({ error: err instanceof Error ? err.message : String(err) });
  });
  return true; // Keep channel open for async response
});

// --- Badge updates ---

async function updateBadge(tabId: number, type: "single" | "multiple", papers: PaperMetadata[]) {
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

function firePdfUploads(
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

async function quickSavePapers(papers: PaperMetadata[]): Promise<boolean> {
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

async function resolveMetadataFromUrl(url: string): Promise<PaperMetadata | null> {
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

// --- Context menu setup ---

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

/** Handle the custom hotkey for saving papers (same flow as save-to-library command). */
async function handleSavePapersHotkey(): Promise<void> {
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

async function handleMessage(message: ExtendedPanelMessage, _sender: chrome.runtime.MessageSender) {
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

// --- Save to Library handlers ---

async function handleGetDetectedPapers() {
  const tab = await getActiveTab();
  if (!tab?.id) return { papers: [], type: null };

  const cached = detectedPapers.get(tab.id);
  if (cached) {
    return { papers: cached.papers, type: cached.type };
  }

  // Try to run generic detection via activeTab on the current page
  try {
    const results = await chrome.tabs.sendMessage(tab.id, { type: "EXTRACT_PAPERS" });
    if (results?.papers?.length > 0) {
      detectedPapers.set(tab.id, {
        type: results.type ?? "single",
        papers: results.papers,
        translatorName: "generic",
      });
      return { papers: results.papers, type: results.type ?? "single" };
    }
  } catch {
    // Content script not injected on this page
  }

  // Try injecting translator-runner via scripting API on the active tab
  try {
    const injectionResults = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => {
        // Inline generic detection for pages without content script
        function getMeta(names: string[]): string | null {
          for (const name of names) {
            const el = document.querySelector(`meta[name="${name}" i], meta[property="${name}" i]`);
            if (el) {
              const content = el.getAttribute("content");
              if (content?.trim()) return content.trim();
            }
          }
          return null;
        }

        function getAllMeta(name: string): string[] {
          return Array.from(document.querySelectorAll(`meta[name="${name}" i], meta[property="${name}" i]`))
            .map((el) => el.getAttribute("content")?.trim())
            .filter((c): c is string => !!c);
        }

        const NOISE_SELECTORS = [
          "nav", "footer", ".cookie-consent", ".cookie-banner", ".cookie-notice",
          ".share-tools", ".share-widget", ".social-share",
          ".author-info", ".author-notes",
          ".metrics", ".altmetric",
          ".supplementary-data", ".supplementary-materials",
          '[role="navigation"]', '[role="banner"]',
          ".sidebar", "#sidebar",
          ".advertisement", ".ad-container",
          ".related-articles", ".recommended-articles",
          ".references", "#references",
          ".footnotes", ".endnotes",
          ".bibliography", ".Footnotes", ".Tail",
          ".RelatedContent", ".ReferencedArticles",
          ".ListArticles", ".Copyright",
          "figure", ".figure", ".table-wrap",
        ].join(", ");

        const BLOCK_TAGS = new Set([
          "div", "section", "article", "aside", "blockquote",
          "table", "figure", "ul", "ol", "pre", "form",
          "header", "footer", "nav", "main",
        ]);

        const CITATION_RE = /\[(\d+(?:[,\s]*\d+)*(?:\s*[-–]\s*\d+)?)\]/g;
        const SUPERSCRIPT_RE = /[⁰¹²³⁴⁵⁶⁷⁸⁹]+/g;

        function cleanPara(text: string): string {
          return text
            .replace(CITATION_RE, "")
            .replace(SUPERSCRIPT_RE, "")
            .replace(/[\u00a0\u200b\u200c\u200d\ufeff]/g, " ")
            .replace(/ ([.,;:!?])/g, "$1")
            .replace(/  +/g, " ")
            .trim();
        }

        function isLeafTextBlock(el: Element): boolean {
          for (const child of el.children) {
            if (BLOCK_TAGS.has(child.tagName.toLowerCase())) return false;
          }
          return true;
        }

        function inlineExtractStructured(): { full_text: string | undefined; structured_text: any } {
          const containerSelectors = [
            ".jig-ncbiinpagenav .tsec",
            "#body .section",
            "article .c-article-body",
            "article .article-body",
            "#article-body",
            ".article-section__content",
            ".article-content",
            ".Body",
            '[role="main"]',
            "article",
            "main",
          ];

          const hostname = location.hostname ?? "";
          let extractionMethod = "generic";
          if (hostname.includes("ncbi.nlm.nih.gov")) extractionMethod = "pmc";
          else if (hostname.includes("sciencedirect.com")) extractionMethod = "elsevier";
          else if (hostname.includes("nature.com") || hostname.includes("springer.com")) extractionMethod = "springer";
          else if (hostname.includes("wiley.com")) extractionMethod = "wiley";

          for (const sel of containerSelectors) {
            const containers = document.querySelectorAll(sel);
            if (containers.length === 0) continue;

            const wrapper = document.createElement("div");
            for (const c of containers) {
              wrapper.appendChild(c.cloneNode(true));
            }
            const noiseEls = wrapper.querySelectorAll(NOISE_SELECTORS);
            for (const el of noiseEls) el.remove();

            const sections: { heading?: string; paragraphs: string[] }[] = [];
            let cur: { heading?: string; paragraphs: string[] } = { paragraphs: [] };

            const elements = wrapper.querySelectorAll("h2, h3, h4, p, div");
            for (const el of elements) {
              const tag = el.tagName.toLowerCase();
              if (tag === "h2" || tag === "h3" || tag === "h4") {
                if (cur.paragraphs.length > 0) sections.push(cur);
                const h = el.textContent?.trim() ?? "";
                cur = { heading: h || undefined, paragraphs: [] };
              } else {
                if (tag === "div" && !isLeafTextBlock(el)) continue;
                const raw = el.textContent?.trim();
                if (raw && raw.length > 30) {
                  const cleaned = cleanPara(raw);
                  if (cleaned.length > 30) cur.paragraphs.push(cleaned);
                }
              }
            }
            if (cur.paragraphs.length > 0) sections.push(cur);

            const allParas: string[] = [];
            for (const s of sections) for (const p of s.paragraphs) allParas.push(p);
            const fullText = allParas.join("\n\n");

            if (fullText.length >= 200) {
              return {
                full_text: fullText,
                structured_text: { sections, extraction_method: extractionMethod, source_hostname: hostname },
              };
            }
          }
          return { full_text: undefined, structured_text: undefined };
        }

        function extractAbstractFromDom(): string | null {
          const selectors = [
            ".Abstracts .abstract.author",
            ".abstract-content",
            '[class*="abstract"] p',
            "#abstract p",
            ".hlFld-Abstract p",
            ".abstractSection",
          ];
          for (const sel of selectors) {
            const els = document.querySelectorAll(sel);
            if (els.length === 0) continue;
            const texts: string[] = [];
            for (const el of els) {
              const text = el.textContent?.trim();
              if (text && text.length > 30) texts.push(cleanPara(text));
            }
            const combined = texts.join(" ");
            if (combined.length > 100) return combined.replace(/^Abstract\s*/i, "");
          }
          return null;
        }

        const title = getMeta(["citation_title", "DC.Title", "DC.title", "og:title"]);
        if (!title) return { papers: [], type: null };

        const authors = getAllMeta("citation_author");
        const doi = getMeta(["citation_doi", "DC.Identifier"]) ?? undefined;
        let abstract = getMeta(["citation_abstract", "DC.Description", "og:description"]) ?? undefined;
        if (!abstract || abstract.length < 200) {
          const domAbstract = extractAbstractFromDom();
          if (domAbstract && domAbstract.length > (abstract?.length ?? 0)) {
            abstract = domAbstract;
          }
        }
        const journal = getMeta(["citation_journal_title", "DC.Source"]) ?? undefined;
        const dateStr = getMeta(["citation_date", "citation_publication_date", "DC.Date"]);
        const year = dateStr ? parseInt(dateStr.match(/(\d{4})/)?.[1] ?? "", 10) || undefined : undefined;
        const pdf_url = getMeta(["citation_pdf_url"]) ?? undefined;

        // Additional metadata fields
        const volume = getMeta(["citation_volume", "PRISM.volume"]) ?? undefined;
        const issue = getMeta(["citation_issue", "PRISM.number"]) ?? undefined;
        const firstPage = getMeta(["citation_firstpage"]);
        const lastPage = getMeta(["citation_lastpage"]);
        const pages = firstPage ? (lastPage ? `${firstPage}-${lastPage}` : firstPage) : undefined;
        const pmid = getMeta(["citation_pmid"]) ?? undefined;
        const issn = getMeta(["citation_issn", "PRISM.issn", "PRISM.eIssn"]) ?? undefined;
        const publisher = getMeta(["citation_publisher", "DC.Publisher"]) ?? undefined;
        const language = getMeta(["citation_language", "DC.Language"]) ?? undefined;
        let keywords: string[] | undefined;
        const keywordStr = getMeta(["citation_keywords"]);
        if (keywordStr) {
          keywords = keywordStr.split(",").map((k: string) => k.trim()).filter(Boolean);
        }
        if (!keywords?.length) {
          const dcSubjects = getAllMeta("DC.Subject");
          if (dcSubjects.length) keywords = dcSubjects;
        }

        const { full_text, structured_text } = inlineExtractStructured();

        return {
          papers: [{
            title,
            authors: authors.length ? authors : undefined,
            year,
            doi,
            abstract,
            journal,
            url: location.href,
            pdf_url,
            full_text,
            structured_text,
            volume,
            issue,
            pages,
            pmid,
            issn,
            publisher,
            keywords: keywords?.length ? keywords : undefined,
            language,
          }],
          type: "single",
        };
      },
    });

    const result = injectionResults?.[0]?.result as
      | { papers: PaperMetadata[]; type: "single" | "multiple" | null }
      | undefined;
    if (result && result.papers && result.papers.length > 0) {
      const detectedType: "single" | "multiple" = result.type === "multiple" ? "multiple" : "single";
      detectedPapers.set(tab.id, {
        type: detectedType,
        papers: result.papers,
        translatorName: "generic-injected",
      });

      // Set popup mode and badge for this tab
      await chrome.action.setPopup({ tabId: tab.id, popup: "popup/popup.html" });
      await updateBadge(tab.id, detectedType, result.papers);

      return { papers: result.papers, type: detectedType };
    }
  } catch {
    // Injection not allowed on this page
  }

  return { papers: [], type: null };
}

async function handleSavePapers(message: { papers: PaperMetadata[]; collectionId?: string | null; tags?: string[]; enrich?: boolean }) {
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

async function handleCheckLibrary(message: { papers: PaperMetadata[] }) {
  const settings = await loadSettings();
  if (!settings.apiToken) return { results: [] };

  const apiClient = await getClient();
  const checkPapers = message.papers.map((p) => ({ doi: p.doi ?? null, title: p.title }));
  const results = await apiClient.checkLibrary(checkPapers);
  return { results: results ?? [] };
}

async function handleGetCollections() {
  const settings = await loadSettings();
  if (!settings.apiToken) return { collections: [] };

  const apiClient = await getClient();
  const collections = await apiClient.getCollections();
  return { collections };
}

async function handleSearchTags(message: { query: string }) {
  const settings = await loadSettings();
  if (!settings.apiToken) return { tags: [] };

  const apiClient = await getClient();
  const tags = await apiClient.searchTags(message.query);
  return { tags };
}

async function handleUpdateLibraryItem(message: { canonicalId: string; collectionId?: string | null; tags?: string[] }) {
  const settings = await loadSettings();
  if (!settings.apiToken) return { error: "Not signed in" };

  return await apiUpdateLibraryItem(message.canonicalId, message.collectionId, message.tags, settings);
}

/** Update a library item — not yet in shared client, so use direct fetch. */
async function apiUpdateLibraryItem(
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

// --- Detect editor type from active tab ---

async function getActiveTab(): Promise<chrome.tabs.Tab | null> {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab ?? null;
}

function detectEditorType(url: string): EditorType {
  if (url.includes("docs.google.com/document")) return "googledocs";
  if (url.includes("overleaf.com/project")) return "overleaf";
  return "unknown";
}

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

async function handleGetRecommendations(collectionId?: string | null) {
  const tab = await getActiveTab();
  if (!tab) return { type: "RECOMMENDATIONS_RESULT", error: "No active tab" };

  const settings = await loadSettings();
  const editorType = detectEditorType(tab.url ?? "");

  let query: string;
  let cursorSentenceIndex: number | undefined;

  if (editorType === "googledocs") {
    // Google Docs (canvas mode): use texteventtarget copy trick to get selected text.
    // This captures what the user selected; we use it directly as the query.
    const cursorText = await getGDocsCursorText(tab);
    if (!cursorText) {
      return { type: "RECOMMENDATIONS_RESULT", error: "Select text in your document and try again." };
    }
    query = cursorText;
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

async function handleGetRecommendationsForText(text: string, collectionId?: string | null) {
  const settings = await loadSettings();
  const stripped = stripCitations(text);

  if (!stripped || stripped.length < 10) {
    return { type: "RECOMMENDATIONS_RESULT", error: "Text is too short for recommendations." };
  }

  const apiClient = await getClient();
  const response = await apiClient.recommend(stripped, settings.k, settings.authorBoost, undefined, collectionId);
  return { type: "RECOMMENDATIONS_RESULT", response };
}

async function handleCheckHealth() {
  try {
    const apiClient = await getClient();
    const response = await apiClient.health();
    return { type: "HEALTH_RESULT", response };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return { type: "HEALTH_RESULT", error: message };
  }
}

async function handleInsertCitation(rec: Recommendation) {
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

async function handleInsertMultiCitation(recs: Recommendation[]) {
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

// --- Google Docs REST API handlers ---

/** Get the current Google Doc ID from the active tab URL. */
function getActiveDocId(tabUrl: string): string | null {
  return extractDocId(tabUrl);
}

/** Get a GDocsCitationInserter for the active document. */
function getInserter(docId: string): GDocsCitationInserter {
  return new GDocsCitationInserter(gdocsApi, docId);
}

/**
 * Get selected text from Google Docs via the texteventtarget copy trick.
 * Returns the selected text directly, or null if nothing is selected.
 */
async function getGDocsCursorText(tab: chrome.tabs.Tab): Promise<string | null> {
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
async function getGDocsCursorIndex(tab: chrome.tabs.Tab): Promise<{ index: number; fullText: string } | null> {
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
 */
async function insertPlaceholderAndLocate(
  tab: chrome.tabs.Tab,
  docId: string
): Promise<{ index: number; placeholder: string; replacedText: string }> {
  const placeholder = `⟦INCITE-${crypto.randomUUID().slice(0, 8)}⟧`;

  // Step 0: Capture the currently selected text before the paste replaces it
  const selectedText = await getGDocsCursorText(tab) ?? "";

  // Step 1: Ask content script to paste the placeholder at the cursor
  // (This will replace the selected text with the placeholder)
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

  // Step 2: Fetch the document and find the placeholder
  const doc = await gdocsApi.getDocument(docId);
  const fullText = extractFullText(doc);
  const placeholderPos = fullText.indexOf(placeholder);

  if (placeholderPos === -1) {
    throw new Error("Placeholder not found in document — paste may not have worked");
  }

  // Convert text position to document index by scanning structural elements
  const docIndex = textOffsetToDocIndex(doc, placeholderPos);

  // Step 3: Delete the placeholder and restore the original selected text
  const requests: import("../shared/gdocs-api").DocsRequest[] = [
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
function textOffsetToDocIndex(doc: DocsDocument, textOffset: number): number {
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

async function handleGDocsInsertCitation(message: { paperId: string; text: string; paperUrl: string }) {
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

async function handleGDocsInsertMultiCitation(message: { fullText: string; segments: { text: string; paperUrl: string; paperId: string; offsetInFullText: number }[] }) {
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

async function handleGDocsInsertBibliography(message: { entries: { paperId: string; formatted: string; url?: string }[] }) {
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

async function handleGDocsScanCitations(message: { trackedPaperIds: string[] }) {
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

async function handleGDocsClean() {
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

async function handleGDocsRefreshCitations(message: { trackedPaperIds: string[]; refreshText: boolean }) {
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
