/**
 * Panel entry point — DOM setup, event wiring, and initialization.
 *
 * All domain logic lives in sub-modules:
 * - panel-state.ts: shared mutable state
 * - panel-recommendations.ts: fetching, health, collections, result rendering
 * - panel-citations.ts: insert, multi-insert, reconciliation, badges
 * - panel-bibliography.ts: bibliography rendering, export, GDocs actions
 */
import { CitationTracker } from "@incite/shared";
import { ChromeCitationStorage, getDocKeyFromActiveTab } from "../shared/citation-storage";
import {
  setTracker,
  setCurrentEditorType,
  setSelectedCollectionId,
} from "./panel-state";
import {
  initRecommendationsDom,
  checkHealth,
  loadPanelSettings,
  detectEditorType,
  getRecommendations,
  getRecommendationsForText,
} from "./panel-recommendations";
import { initCitationsDom } from "./panel-citations";
import { renderBibliography } from "./panel-bibliography";

// --- DOM references ---
const content = document.getElementById("content")!;
const btnRecommend = document.getElementById("btn-recommend") as HTMLButtonElement;
const statusDot = document.getElementById("status-dot")!;
const manualInput = document.getElementById("manual-input")!;
const btnToggleManual = document.getElementById("btn-toggle-manual") as HTMLButtonElement;
const manualText = document.getElementById("manual-text") as HTMLTextAreaElement;
const btnManualSubmit = document.getElementById("btn-manual-submit") as HTMLButtonElement;
const collectionFilter = document.getElementById("collection-filter")!;
const collectionSelect = document.getElementById("collection-select") as HTMLSelectElement;

// --- Wire DOM refs into sub-modules ---
initRecommendationsDom({
  content,
  btnRecommend,
  btnManualSubmit,
  statusDot,
  manualInput,
  manualText,
  collectionFilter,
  collectionSelect,
});
initCitationsDom({ content });

// --- Event listeners ---

btnRecommend.addEventListener("click", () => getRecommendations());

btnToggleManual.addEventListener("click", () => {
  manualInput.classList.toggle("collapsed");
  if (!manualInput.classList.contains("collapsed")) {
    manualText.focus();
  }
});

btnManualSubmit.addEventListener("click", () => {
  const text = manualText.value.trim();
  if (text.length > 0) {
    getRecommendationsForText(text);
  }
});

// Listen for hotkey trigger from service worker
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === "TRIGGER_FROM_HOTKEY") {
    getRecommendations();
    sendResponse({ ack: true });
  }
  return false;
});

// Signal that the panel is ready
chrome.runtime.sendMessage({ type: "PANEL_READY" }).catch((err) => {
  console.error("PANEL_READY message failed:", err);
});

// Collection filter
collectionSelect.addEventListener("change", () => {
  setSelectedCollectionId(collectionSelect.value || null);
  chrome.storage.sync.set({ incite_collection_id: collectionSelect.value || null });
});

// Load persisted collection selection
chrome.storage.sync.get("incite_collection_id", (result) => {
  setSelectedCollectionId(result.incite_collection_id ?? null);
});

// --- Initialization ---

checkHealth();
loadPanelSettings();
initTracker();
detectEditorType().then((et) => { setCurrentEditorType(et); });

// --- Tracker initialization ---

async function initTracker(): Promise<void> {
  const docKey = await getDocKeyFromActiveTab();
  if (!docKey) return;
  const storage = new ChromeCitationStorage();
  const t = new CitationTracker(storage, docKey);
  await t.load();
  setTracker(t);
  renderBibliography();
}

// --- Utilities ---

/** Show a brief toast notification. */
export function showToast(message: string): void {
  const existing = document.querySelector(".toast");
  if (existing) existing.remove();

  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 2500);
}
