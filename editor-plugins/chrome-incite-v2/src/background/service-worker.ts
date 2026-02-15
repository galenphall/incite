import { extractContext, stripCitations, formatCitation, formatMultiCitation } from "@incite/shared";
import type { Recommendation, RecommendResponse, HealthResponse } from "@incite/shared";
import type {
  ChromeExtensionSettings,
  EditorType,
  GetContextMessage,
  ContextResponseMessage,
  PanelMessage,
} from "../shared/types";
import { loadSettings, saveSettings } from "../shared/settings";

// --- Side Panel lifecycle ---

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
}

// Enable side panel on supported sites
chrome.tabs.onUpdated.addListener(async (tabId, _info, tab) => {
  if (!tab.url) return;
  const isSupported =
    tab.url.includes("docs.google.com/document") ||
    tab.url.includes("overleaf.com/project");
  await chrome.sidePanel.setOptions({
    tabId,
    enabled: isSupported,
  });
});

// --- Message handling ---

type ExtendedPanelMessage = PanelMessage | { type: "PANEL_READY" } | { type: "GET_RECOMMENDATIONS_FOR_TEXT"; text: string };

chrome.runtime.onMessage.addListener((message: ExtendedPanelMessage, sender, sendResponse) => {
  if (message.type === "PANEL_READY") {
    panelReady = true;
    sendResponse({ ack: true });
    return false;
  }

  handleMessage(message, sender).then(sendResponse).catch((err) => {
    sendResponse({ error: err instanceof Error ? err.message : String(err) });
  });
  return true; // Keep channel open for async response
});

async function handleMessage(message: ExtendedPanelMessage, _sender: chrome.runtime.MessageSender) {
  switch (message.type) {
    case "GET_RECOMMENDATIONS":
      return await handleGetRecommendations();
    case "GET_RECOMMENDATIONS_FOR_TEXT":
      return await handleGetRecommendationsForText((message as { type: string; text: string }).text);
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
    default:
      return { error: "Unknown message type" };
  }
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

// --- API calls ---

function getApiUrl(settings: ChromeExtensionSettings): string {
  return settings.apiMode === "cloud" ? settings.cloudUrl : settings.localUrl;
}

function getApiHeaders(settings: ChromeExtensionSettings): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "application/json",
  };
  if (settings.apiMode === "cloud" && settings.apiToken) {
    headers["Authorization"] = `Bearer ${settings.apiToken}`;
  }
  return headers;
}

async function apiRecommend(query: string, settings: ChromeExtensionSettings): Promise<RecommendResponse> {
  const baseUrl = getApiUrl(settings);
  const endpoint = settings.apiMode === "cloud" ? "/api/v1/recommend" : "/recommend";

  const response = await fetch(`${baseUrl}${endpoint}`, {
    method: "POST",
    headers: getApiHeaders(settings),
    body: JSON.stringify({
      query,
      k: settings.k,
      author_boost: settings.authorBoost,
    }),
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`API error ${response.status}: ${text || response.statusText}`);
  }

  return response.json();
}

async function apiHealth(settings: ChromeExtensionSettings): Promise<HealthResponse> {
  const baseUrl = getApiUrl(settings);
  const endpoint = settings.apiMode === "cloud" ? "/api/v1/health" : "/health";

  const response = await fetch(`${baseUrl}${endpoint}`, {
    method: "GET",
    headers: getApiHeaders(settings),
  });

  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`);
  }

  return response.json();
}

// --- Handler implementations ---

async function handleGetRecommendations() {
  const tab = await getActiveTab();
  if (!tab) return { type: "RECOMMENDATIONS_RESULT", error: "No active tab" };

  const settings = await loadSettings();
  const query = await getContextFromTab(tab);
  const stripped = stripCitations(query);

  if (!stripped || stripped.length < 10) {
    return { type: "RECOMMENDATIONS_RESULT", error: "Selected text is too short for recommendations." };
  }

  const response = await apiRecommend(stripped, settings);
  return { type: "RECOMMENDATIONS_RESULT", response };
}

async function handleGetRecommendationsForText(text: string) {
  const settings = await loadSettings();
  const stripped = stripCitations(text);

  if (!stripped || stripped.length < 10) {
    return { type: "RECOMMENDATIONS_RESULT", error: "Text is too short for recommendations." };
  }

  const response = await apiRecommend(stripped, settings);
  return { type: "RECOMMENDATIONS_RESULT", response };
}

async function handleCheckHealth() {
  const settings = await loadSettings();
  try {
    const response = await apiHealth(settings);
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
