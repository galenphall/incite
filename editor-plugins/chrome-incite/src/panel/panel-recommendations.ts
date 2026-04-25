/**
 * Recommendation fetching, health checks, collection management,
 * and result rendering for the side panel.
 */
import type { Recommendation, RecommendResponse } from "@incite/shared";
import { escapeHtml, renderResultCardHTML } from "@incite/shared";
import type { EditorType } from "../shared/types";
import {
  CHROME_CLASS_MAP,
  isLoading,
  setIsLoading,
  selectedRecs,
  tracker,
  panelSettings,
  setPanelSettings,
  collections,
  setCollections,
  selectedCollectionId,
  setSelectedCollectionId,
} from "./panel-state";
import { insertCitation, updateSelectionBar, insertMultiCitation } from "./panel-citations";
import { renderBibliography } from "./panel-bibliography";
import { showToast } from "./panel";

// --- DOM references (set by init) ---
let content: HTMLElement;
let btnRecommend: HTMLButtonElement;
let btnManualSubmit: HTMLButtonElement;
let statusDot: HTMLElement;
let manualInput: HTMLElement;
let manualText: HTMLTextAreaElement;
let collectionFilter: HTMLElement;
let collectionSelect: HTMLSelectElement;

/** Initialize DOM references. Must be called before any other function. */
export function initRecommendationsDom(refs: {
  content: HTMLElement;
  btnRecommend: HTMLButtonElement;
  btnManualSubmit: HTMLButtonElement;
  statusDot: HTMLElement;
  manualInput: HTMLElement;
  manualText: HTMLTextAreaElement;
  collectionFilter: HTMLElement;
  collectionSelect: HTMLSelectElement;
}): void {
  content = refs.content;
  btnRecommend = refs.btnRecommend;
  btnManualSubmit = refs.btnManualSubmit;
  statusDot = refs.statusDot;
  manualInput = refs.manualInput;
  manualText = refs.manualText;
  collectionFilter = refs.collectionFilter;
  collectionSelect = refs.collectionSelect;
}

// --- Settings & detection ---

export async function loadPanelSettings(): Promise<void> {
  try {
    const response = await chrome.runtime.sendMessage({ type: "GET_SETTINGS" });
    if (response?.settings) {
      setPanelSettings({
        showParagraphs: response.settings.showParagraphs ?? true,
        showAbstracts: response.settings.showAbstracts ?? false,
        googleDocsCitationFormat: response.settings.googleDocsCitationFormat ?? "(${first_author}, ${year})",
      });
    }
  } catch (err) {
    console.error("Failed to load panel settings:", err);
  }
}

/** Detect editor type from the active tab URL. */
export async function detectEditorType(): Promise<EditorType> {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const url = tab?.url ?? "";
    if (url.includes("docs.google.com/document")) return "googledocs";
    if (url.includes("overleaf.com/project")) return "overleaf";
  } catch {
    // Ignore
  }
  return "unknown";
}

// --- Health & collections ---

export async function checkHealth(): Promise<void> {
  try {
    const response = await chrome.runtime.sendMessage({ type: "CHECK_HEALTH" });
    if (response?.response) {
      statusDot.className = "status-dot connected";
      statusDot.title = `Connected -- ${response.response.corpus_size ?? "?"} papers`;
      fetchCollections();
    } else {
      statusDot.className = "status-dot error";
      statusDot.title = response?.error ?? "Not connected";
    }
  } catch {
    statusDot.className = "status-dot error";
    statusDot.title = "Not connected";
  }
}

export async function fetchCollections(): Promise<void> {
  try {
    const response = await chrome.runtime.sendMessage({ type: "GET_COLLECTIONS" });
    setCollections(response?.collections ?? []);
    renderCollectionDropdown();
  } catch {
    // Collections are optional
  }
}

export function renderCollectionDropdown(): void {
  // Only show in cloud mode when collections exist
  if (collections.length === 0) {
    collectionFilter.style.display = "none";
    return;
  }

  collectionSelect.innerHTML = '<option value="">All papers</option>';
  for (const c of collections) {
    const opt = document.createElement("option");
    opt.value = String(c.id);
    opt.textContent = `${c.name} (${c.item_count})`;
    if (selectedCollectionId === String(c.id)) {
      opt.selected = true;
    }
    collectionSelect.appendChild(opt);
  }
  collectionFilter.style.display = "";
}

// --- Core recommendation functions ---

export async function getRecommendations(): Promise<void> {
  if (isLoading) return;
  setIsLoading(true);
  btnRecommend.disabled = true;
  showLoading();

  try {
    const response = await chrome.runtime.sendMessage({ type: "GET_RECOMMENDATIONS", collectionId: selectedCollectionId });
    if (response?.error) {
      showExtractionError(response.error);
    } else if (response?.response) {
      await showResults(response.response, response.query);
    } else {
      showError("Unexpected response from service worker.");
    }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    showError(message);
  } finally {
    setIsLoading(false);
    btnRecommend.disabled = false;
  }
}

export async function getRecommendationsForText(text: string): Promise<void> {
  if (isLoading) return;
  setIsLoading(true);
  btnManualSubmit.disabled = true;
  btnRecommend.disabled = true;
  showLoading();

  try {
    const response = await chrome.runtime.sendMessage({ type: "GET_RECOMMENDATIONS_FOR_TEXT", text, collectionId: selectedCollectionId });
    if (response?.error) {
      showError(response.error);
    } else if (response?.response) {
      await showResults(response.response);
    } else {
      showError("Unexpected response from service worker.");
    }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    showError(message);
  } finally {
    setIsLoading(false);
    btnManualSubmit.disabled = false;
    btnRecommend.disabled = false;
  }
}

// --- Rendering ---

export function showLoading(): void {
  content.innerHTML = `
    <div class="loading">
      <div class="spinner"></div>
      <p>Searching your library...</p>
    </div>
  `;
}

export function showError(message: string): void {
  content.innerHTML = `<div class="error-state">${escapeHtml(message)}</div>`;
}

/** Show an extraction error and auto-expand the manual input fallback. */
export function showExtractionError(message: string): void {
  content.innerHTML = `<div class="error-state">${escapeHtml(message)}</div>`;
  // Auto-expand manual input when extraction fails
  manualInput.classList.remove("collapsed");
  manualText.focus();
}

export async function showResults(response: RecommendResponse, queryText?: string): Promise<void> {
  await loadPanelSettings();
  const recs = response.recommendations;
  selectedRecs.clear();

  if (!recs || recs.length === 0) {
    content.innerHTML = `<div class="empty-state"><p>No matching papers found.</p></div>`;
    return;
  }

  let html = "";

  // Show query context for debugging
  if (queryText) {
    const truncated = queryText.length > 300 ? queryText.slice(0, 300) + "..." : queryText;
    html += `<div class="query-context"><details><summary class="timing">Query context</summary><p style="font-size:11px;color:var(--fg-muted);white-space:pre-wrap;margin:4px 0;">${escapeHtml(truncated)}</p></details></div>`;
  }

  // Results header with timing and clear button
  html += `<div class="results-header">`;
  if (response.timing?.total_ms) {
    html += `<span class="timing">${recs.length} results in ${Math.round(response.timing.total_ms)}ms — ${response.corpus_size} papers</span>`;
  }
  html += `<button id="btn-clear-results" class="btn-clear-results" title="Clear results">✕</button>`;
  html += `</div>`;

  // Selection bar (hidden by default)
  html += `<div id="selection-bar" class="selection-bar" style="display:none;">`;
  html += `<span id="selection-count">0 selected</span>`;
  html += `<button id="btn-insert-selected" class="btn-small btn-insert">Insert Selected</button>`;
  html += `<button id="btn-clear-selected" class="btn-small">Clear</button>`;
  html += `</div>`;

  for (const rec of recs) {
    const isCited = tracker?.isTracked(rec.paper_id) ?? false;
    html += renderResultCardHTML(rec, {
      showParagraphs: panelSettings.showParagraphs,
      showAbstracts: panelSettings.showAbstracts,
      isCited,
    }, CHROME_CLASS_MAP);
  }

  content.innerHTML = html;

  // Collapse manual input on successful results
  manualInput.classList.add("collapsed");

  // Attach evidence toggle listeners
  content.querySelectorAll("[data-action='toggle-evidence']").forEach((btn) => {
    btn.addEventListener("click", () => {
      const evidenceContent = btn.nextElementSibling as HTMLElement | null;
      if (evidenceContent) {
        const expanded = evidenceContent.classList.toggle("expanded");
        btn.innerHTML = expanded ? "Hide evidence &#9652;" : "Show evidence &#9662;";
      }
    });
  });

  // Attach event listeners for insert buttons
  content.querySelectorAll("[data-action='insert']").forEach((btn) => {
    btn.addEventListener("click", () => {
      const recData = btn.getAttribute("data-rec");
      if (recData) {
        const recommendation = JSON.parse(recData) as Recommendation;
        insertCitation(recommendation);
      }
    });
  });

  // Attach event listeners for copy buttons
  content.querySelectorAll("[data-action='copy']").forEach((btn) => {
    btn.addEventListener("click", () => {
      const text = btn.getAttribute("data-copy");
      if (text) {
        navigator.clipboard.writeText(text).then(() => showToast("Copied!"));
      }
    });
  });

  // Attach event listeners for checkboxes
  content.querySelectorAll<HTMLInputElement>("[data-action='select']").forEach((cb) => {
    cb.addEventListener("change", () => {
      const recData = cb.getAttribute("data-rec");
      if (!recData) return;
      const rec = JSON.parse(recData) as Recommendation;
      if (cb.checked) {
        selectedRecs.set(rec.paper_id, rec);
      } else {
        selectedRecs.delete(rec.paper_id);
      }
      updateSelectionBar();
    });
  });

  // Selection bar buttons
  document.getElementById("btn-insert-selected")?.addEventListener("click", () => {
    insertMultiCitation();
  });

  document.getElementById("btn-clear-selected")?.addEventListener("click", () => {
    selectedRecs.clear();
    content.querySelectorAll<HTMLInputElement>("[data-action='select']").forEach((cb) => {
      cb.checked = false;
    });
    updateSelectionBar();
  });

  // Clear results button
  document.getElementById("btn-clear-results")?.addEventListener("click", () => {
    selectedRecs.clear();
    content.innerHTML = `<div class="empty-state"><p>Select text in your document, then click <strong>Get Recommendations</strong>.</p></div>`;
  });

  // Render bibliography below results
  renderBibliography();
}
