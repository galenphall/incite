import { extractContext, stripCitations, formatCitation, formatMultiCitation, InCiteClient, FetchTransport, getActiveUrl } from "@incite/shared";
import type { Recommendation, RecommendResponse, HealthResponse, ClientConfig } from "@incite/shared";
import type {
  ChromeExtensionSettings,
  EditorType,
  GetContextMessage,
  ContextResponseMessage,
  PanelMessage,
} from "../shared/types";
import { loadSettings, saveSettings } from "../shared/settings";
import { isAcademicSite } from "../translators/registry";

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
}

interface DetectedPapersState {
  type: "single" | "multiple";
  papers: PaperMetadata[];
  translatorName: string;
}

// Cache of detected papers per tab
const detectedPapers = new Map<number, DetectedPapersState>();

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

// Don't auto-open side panel on click — we manage this per-tab via popup vs onClicked
chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: false });

// Track whether the panel is ready to receive messages
let panelReady = false;

// For writing sites (no popup set), clicking the icon opens the side panel
chrome.action.onClicked.addListener(async (tab) => {
  if (tab.id) {
    await chrome.sidePanel.open({ tabId: tab.id });
  }
});

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

chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (!tab.url) return;

  const isWriting =
    tab.url.includes("docs.google.com/document") ||
    tab.url.includes("overleaf.com/project");
  const isAcademic = isAcademicSite(tab.url);

  if (isWriting) {
    // Side panel mode (current behavior for writing contexts)
    await chrome.sidePanel.setOptions({ tabId, enabled: true });
    await chrome.action.setPopup({ tabId, popup: "" });
  } else if (isAcademic) {
    // Popup mode for save-to-library on academic sites
    await chrome.sidePanel.setOptions({ tabId, enabled: false });
    await chrome.action.setPopup({ tabId, popup: "popup/popup.html" });
  } else {
    // All other sites: popup mode for save-to-library (generic meta tag detection)
    await chrome.sidePanel.setOptions({ tabId, enabled: false });
    await chrome.action.setPopup({ tabId, popup: "popup/popup.html" });
  }

  // Clear cached detection when navigating away
  if (changeInfo.url) {
    detectedPapers.delete(tabId);
    await chrome.action.setBadgeText({ tabId, text: "" });
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
  | { type: "GET_RECOMMENDATIONS_FOR_TEXT"; text: string }
  | LibraryMessage;

chrome.runtime.onMessage.addListener((message: ExtendedPanelMessage, sender, sendResponse) => {
  if (message.type === "PANEL_READY") {
    panelReady = true;
    sendResponse({ ack: true });
    return false;
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
    await apiClient.savePapers({
      papers,
      collection_id: collectionId,
      tags: [],
      enrich: true,
    });

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
  const query = await getContextFromTab(tab);
  const stripped = stripCitations(query);

  if (!stripped || stripped.length < 10) {
    return { type: "RECOMMENDATIONS_RESULT", error: "Selected text is too short for recommendations." };
  }

  const apiClient = await getClient();
  const response = await apiClient.recommend(stripped, settings.k, settings.authorBoost, undefined, collectionId);
  return { type: "RECOMMENDATIONS_RESULT", response };
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
