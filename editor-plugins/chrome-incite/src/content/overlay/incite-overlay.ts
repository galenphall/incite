/**
 * InCite overlay content script — renders the collapsed rail (Mode A) and
 * command palette popup (Mode B) on writing sites via shadow DOM.
 *
 * Injected into Google Docs and Overleaf pages. Communicates with the
 * service worker for recommendations and citation insertion using the
 * same message protocol as the side panel.
 */
import type { Recommendation } from "@incite/shared";
import { formatCitation } from "@incite/shared";
import { OVERLAY_CSS } from "./overlay-styles";
import { renderRail, renderPopup } from "./overlay-render";
import type { OverlayState } from "./overlay-render";

// ─── Guard against double injection ───

const INJECTED_FLAG = "__incite_overlay_injected__";
if ((window as unknown as Record<string, boolean>)[INJECTED_FLAG]) {
  // Already injected — bail out
} else {
  (window as unknown as Record<string, boolean>)[INJECTED_FLAG] = true;
  initOverlay();
}

// ─── State ───

const state: OverlayState = {
  mode: "rail",
  results: [],
  loading: false,
  selectedIndex: -1,
  expandedEvidence: new Set(),
  timing: null,
  corpusSize: 0,
  error: null,
  collectionId: null,
  citedPaperIds: new Set(),
};

// Monotonic ID for in-flight fetches; responses with stale IDs are discarded.
let activeFetchId = 0;

// Element to restore focus to when the popup closes.
let previouslyFocused: Element | null = null;

// Shadow DOM references (set in initOverlay)
let shadow: ShadowRoot;
let railEl: HTMLElement;
let popupEl: HTMLElement | null = null;

// ─── Editor type detection ───

function detectEditorType(): "googledocs" | "overleaf" {
  try {
    return new URL(location.href).host === "docs.google.com" ? "googledocs" : "overleaf";
  } catch {
    return "overleaf";
  }
}

// ─── Initialization ───

function initOverlay(): void {
  // Create host element with style reset
  const host = document.createElement("div");
  host.id = "incite-overlay-host";
  host.style.cssText =
    "all: initial; position: fixed; top: 0; right: 0; bottom: 0; z-index: 2147483647; pointer-events: none; font-size: 12px;";
  document.body.appendChild(host);

  shadow = host.attachShadow({ mode: "closed" });

  // Inject fonts — Google Fonts work via <link> in shadow DOM in Chrome
  const fontLink = document.createElement("link");
  fontLink.rel = "stylesheet";
  fontLink.href =
    "https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap";
  shadow.appendChild(fontLink);

  // Inject styles
  const style = document.createElement("style");
  style.textContent = OVERLAY_CSS;
  shadow.appendChild(style);

  // Render rail
  const railContainer = document.createElement("div");
  railContainer.innerHTML = renderRail();
  railEl = railContainer.firstElementChild as HTMLElement;
  shadow.appendChild(railEl);

  // Wire rail events
  railEl.addEventListener("click", handleRailClick);

  // Listen for messages from service worker. Only TOGGLE_COMMAND_PALETTE is
  // ours; other content scripts on this tab handle their own message types.
  chrome.runtime.onMessage.addListener(handleMessage);

  // Load persisted collection selection
  chrome.storage.sync.get("incite_collection_id", (result) => {
    state.collectionId = result.incite_collection_id ?? null;
  });

  // Visibility preference: rail is opt-in to avoid covering host editor UI
  // (Google Docs's right sidebar buttons, Overleaf's PDF preview). The
  // keyboard hotkey still opens the popup directly when the rail is hidden.
  chrome.storage.sync.get("incite_overlay_visible", (result) => {
    applyRailVisibility(result.incite_overlay_visible === true);
  });

  // React to changes from the options page (or other tabs) without reload.
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "sync") return;
    if (changes.incite_overlay_visible) {
      applyRailVisibility(changes.incite_overlay_visible.newValue === true);
    }
    if (changes.incite_collection_id) {
      state.collectionId = changes.incite_collection_id.newValue ?? null;
    }
  });
}

function applyRailVisibility(visible: boolean): void {
  railEl.style.display = visible ? "" : "none";
}

// ─── Message Handling ───

function handleMessage(
  message: { type: string },
  _sender: chrome.runtime.MessageSender,
  sendResponse: (response: unknown) => void,
): boolean {
  if (message.type === "TOGGLE_COMMAND_PALETTE") {
    togglePopup();
    sendResponse({ ack: true });
    return false;
  }
  return false;
}

// ─── Rail Events ───

function handleRailClick(e: Event): void {
  const target = (e.target as HTMLElement).closest("[data-action]") as HTMLElement | null;
  if (!target) return;

  const action = target.dataset.action;
  switch (action) {
    case "recommend":
      togglePopup();
      break;
    case "expand":
      chrome.runtime.sendMessage({ type: "OPEN_SIDE_PANEL" });
      break;
    case "settings":
      window.open(chrome.runtime.getURL("options/options.html"), "_blank");
      break;
    case "hide":
      chrome.storage.sync.set({ incite_overlay_visible: false });
      showToast("Overlay hidden — re-enable in InCite settings");
      break;
  }
}

// ─── Popup Toggle ───

function togglePopup(): void {
  if (state.mode === "popup") {
    closePopup();
  } else {
    openPopup();
  }
}

function openPopup(): void {
  // Remember where focus was so we can restore it on close.
  previouslyFocused = document.activeElement;

  state.mode = "popup";
  state.loading = true;
  state.results = [];
  state.error = null;
  state.selectedIndex = -1;
  state.expandedEvidence.clear();
  state.timing = null;
  state.corpusSize = 0;

  renderPopupEl();

  // Focus the popup for keyboard navigation
  if (popupEl) {
    popupEl.focus();
  }

  // Single click-outside listener for the lifetime of this popup session.
  // Re-registering on every render would leak listeners on every keystroke.
  document.addEventListener("click", handleClickOutside, true);

  fetchRecommendations();
}

function closePopup(): void {
  state.mode = "rail";
  if (popupEl) {
    popupEl.remove();
    popupEl = null;
  }
  document.removeEventListener("click", handleClickOutside, true);

  // Invalidate any in-flight fetch so a late response can't reopen state.
  activeFetchId++;

  // Restore focus to the editor (or wherever it was) so typing resumes.
  if (previouslyFocused && previouslyFocused instanceof HTMLElement) {
    try {
      previouslyFocused.focus();
    } catch {
      // Element may have been removed from DOM — ignore.
    }
  }
  previouslyFocused = null;
}

// ─── Popup Rendering ───

function renderPopupEl(): void {
  // Remove old popup if present
  if (popupEl) {
    popupEl.remove();
  }

  const container = document.createElement("div");
  container.innerHTML = renderPopup(state);
  popupEl = container.firstElementChild as HTMLElement;
  shadow.appendChild(popupEl);

  // Wire popup events
  wirePopupEvents();
}

function updatePopup(): void {
  if (!popupEl || state.mode !== "popup") return;
  renderPopupEl();
  if (popupEl) {
    popupEl.focus();
  }
}

function wirePopupEvents(): void {
  if (!popupEl) return;

  // Close button
  const closeBtn = popupEl.querySelector("[data-action='close-popup']");
  if (closeBtn) {
    closeBtn.addEventListener("click", () => closePopup());
  }

  // Result row clicks
  popupEl.querySelectorAll(".result-row").forEach((row) => {
    row.addEventListener("click", () => {
      const index = parseInt((row as HTMLElement).dataset.index ?? "-1", 10);
      if (index >= 0 && index < state.results.length) {
        state.selectedIndex = index;
        updatePopup();
      }
    });

    // Double-click to insert
    row.addEventListener("dblclick", () => {
      const index = parseInt((row as HTMLElement).dataset.index ?? "-1", 10);
      if (index >= 0 && index < state.results.length) {
        state.selectedIndex = index;
        insertSelectedCitation();
      }
    });
  });

  // Keyboard navigation
  popupEl.addEventListener("keydown", handlePopupKeydown);
}

function handleClickOutside(e: Event): void {
  if (state.mode !== "popup") return;
  const path = e.composedPath();
  const host = document.getElementById("incite-overlay-host");
  if (host && !path.includes(host)) {
    closePopup();
  }
}

// ─── Keyboard Navigation ───

function handlePopupKeydown(e: KeyboardEvent): void {
  const resultCount = state.results.length;
  if (resultCount === 0 && !state.loading && e.key !== "Escape" && e.key !== "Tab") return;

  switch (e.key) {
    case "ArrowDown":
    case "j": {
      e.preventDefault();
      e.stopPropagation();
      if (resultCount > 0) {
        state.selectedIndex = state.selectedIndex < resultCount - 1
          ? state.selectedIndex + 1
          : 0;
        updatePopup();
        scrollSelectedIntoView();
      }
      break;
    }
    case "ArrowUp":
    case "k": {
      e.preventDefault();
      e.stopPropagation();
      if (resultCount > 0) {
        state.selectedIndex = state.selectedIndex > 0
          ? state.selectedIndex - 1
          : resultCount - 1;
        updatePopup();
        scrollSelectedIntoView();
      }
      break;
    }
    case "Enter": {
      e.preventDefault();
      e.stopPropagation();
      if (state.selectedIndex >= 0 && state.selectedIndex < resultCount) {
        insertSelectedCitation();
      }
      break;
    }
    case "Tab": {
      e.preventDefault();
      e.stopPropagation();
      if (e.shiftKey) {
        // Shift+Tab: open side panel (Mode D)
        chrome.runtime.sendMessage({ type: "OPEN_SIDE_PANEL" });
        closePopup();
      } else if (state.selectedIndex >= 0 && state.selectedIndex < resultCount) {
        // Toggle evidence
        if (state.expandedEvidence.has(state.selectedIndex)) {
          state.expandedEvidence.delete(state.selectedIndex);
        } else {
          state.expandedEvidence.add(state.selectedIndex);
        }
        updatePopup();
      }
      break;
    }
    case "Escape": {
      e.preventDefault();
      e.stopPropagation();
      closePopup();
      break;
    }
  }
}

function scrollSelectedIntoView(): void {
  if (!popupEl || state.selectedIndex < 0) return;
  const row = popupEl.querySelector(`[data-index="${state.selectedIndex}"]`) as HTMLElement | null;
  if (row) {
    row.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}

// ─── Recommendations ───

async function fetchRecommendations(): Promise<void> {
  const fetchId = ++activeFetchId;

  let nextResults: Recommendation[] = [];
  let nextTiming: { total_ms: number } | null = null;
  let nextCorpusSize = 0;
  let nextError: string | null = null;

  try {
    const response = await chrome.runtime.sendMessage({
      type: "GET_RECOMMENDATIONS",
      collectionId: state.collectionId,
    });

    if (response?.error) {
      nextError = response.error;
    } else if (response?.response) {
      nextResults = response.response.recommendations ?? [];
      nextTiming = response.response.timing ?? null;
      nextCorpusSize = response.response.corpus_size ?? 0;
    } else {
      nextError = "Unexpected response from service worker.";
    }
  } catch (err: unknown) {
    nextError = err instanceof Error ? err.message : String(err);
  }

  // Discard stale responses (popup closed/reopened, or another fetch raced ahead).
  if (fetchId !== activeFetchId || state.mode !== "popup") return;

  state.results = nextResults;
  state.timing = nextTiming;
  state.corpusSize = nextCorpusSize;
  state.error = nextError;
  state.loading = false;
  if (nextResults.length > 0) {
    state.selectedIndex = 0;
  }

  updatePopup();

  // Hydrate cited badges from the document so they survive popup re-opens.
  if (!nextError && nextResults.length > 0) {
    hydrateCitedFromDocument(fetchId, nextResults.map((r) => r.paper_id));
  }
}

async function hydrateCitedFromDocument(fetchId: number, paperIds: string[]): Promise<void> {
  if (detectEditorType() !== "googledocs") return;
  try {
    const resp = await chrome.runtime.sendMessage({
      type: "GDOCS_SCAN_CITATIONS",
      trackedPaperIds: paperIds,
    });
    if (fetchId !== activeFetchId || state.mode !== "popup") return;
    const found: string[] | undefined = resp?.data?.found;
    if (!Array.isArray(found)) return;
    let changed = false;
    for (const id of found) {
      if (!state.citedPaperIds.has(id)) {
        state.citedPaperIds.add(id);
        changed = true;
      }
    }
    if (changed) updatePopup();
  } catch {
    // Best-effort hydration; ignore failures.
  }
}

// ─── Citation Insertion ───

async function insertSelectedCitation(): Promise<void> {
  const rec = state.results[state.selectedIndex];
  if (!rec) return;

  const editorType = detectEditorType();

  try {
    if (editorType === "googledocs") {
      // Get citation format from settings
      const settingsResp = await chrome.runtime.sendMessage({ type: "GET_SETTINGS" });
      const template = settingsResp?.settings?.googleDocsCitationFormat ?? "(${first_author}, ${year})";
      const citationText = formatCitation(rec, template);
      const paperUrl = rec.doi
        ? `https://doi.org/${rec.doi}`
        : `https://inciteref.com/paper/${rec.paper_id}`;

      const response = await chrome.runtime.sendMessage({
        type: "GDOCS_INSERT_CITATION",
        paperId: rec.paper_id,
        text: citationText,
        paperUrl,
      });

      if (response?.success === false) {
        // Fallback: copy to clipboard
        await navigator.clipboard.writeText(citationText);
        showToast("Copied to clipboard");
      } else {
        showToast(`Inserted: ${citationText}`);
      }
    } else {
      // Overleaf: use legacy content script insertion
      const response = await chrome.runtime.sendMessage({
        type: "INSERT_CITATION_REQUEST",
        recommendation: rec,
      });

      if (response?.method === "clipboard") {
        showToast("Copied — paste with Cmd/Ctrl+V");
      } else if (response?.success !== false) {
        showToast("Citation inserted");
      } else {
        const key = rec.bibtex_key ?? rec.paper_id;
        await navigator.clipboard.writeText(key);
        showToast("Copied to clipboard");
      }
    }

    // Mark as cited
    state.citedPaperIds.add(rec.paper_id);
    updatePopup();
  } catch {
    showToast("Insert failed");
  }
}

// ─── Toast ───

function showToast(message: string): void {
  // Remove existing toast
  const existing = shadow.querySelector(".overlay-toast");
  if (existing) existing.remove();

  const toast = document.createElement("div");
  toast.className = "overlay-toast";
  toast.textContent = message;
  shadow.appendChild(toast);
  setTimeout(() => toast.remove(), 2500);
}
