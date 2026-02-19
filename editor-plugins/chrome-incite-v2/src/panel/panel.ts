import type { Recommendation, RecommendResponse, TrackedCitation, UIClassMap } from "@incite/shared";
import {
  CitationTracker,
  exportBibTeX,
  exportRIS,
  exportFormattedText,
  escapeHtml,
  escapeAttr,
  renderResultCardHTML,
  renderBibliographyHTML,
} from "@incite/shared";
import { ChromeCitationStorage, getDocKeyFromActiveTab } from "../shared/citation-storage";

// --- Chrome-specific class map ---

const CHROME_CLASS_MAP: UIClassMap = {
  resultCard: "result-card",
  resultHeader: "result-header",
  resultHeaderLeft: "result-header-left",
  selectCheckbox: "select-checkbox",
  rankBadge: "rank-badge",
  citedBadge: "cited-badge",
  confidenceBadge: "confidence-badge",
  confidenceHigh: "confidence-high",
  confidenceMid: "confidence-medium",
  confidenceLow: "confidence-low",
  resultTitle: "result-title",
  resultMeta: "result-meta",
  evidence: "evidence",
  evidenceSecondary: "evidence-secondary",
  evidenceScore: "evidence-score",
  resultAbstract: "result-abstract",
  resultActions: "result-actions",
  insertBtn: "btn-small btn-insert",
  copyBtn: "btn-small",
  bibSection: "bibliography-section",
  bibToggle: "bib-toggle",
  bibContent: "bib-content",
  bibExportBar: "bib-export-bar",
  bibList: "bib-list",
  bibItem: "bib-item",
  bibItemText: "bib-item-text",
  bibItemAuthors: "bib-item-authors",
  bibItemTitle: "bib-item-title",
  bibRemove: "bib-remove",
};

// --- State ---
let isLoading = false;
const selectedRecs = new Map<string, Recommendation>();
let tracker: CitationTracker | null = null;
let panelSettings: { showParagraphs: boolean; showAbstracts: boolean } = {
  showParagraphs: true,
  showAbstracts: false,
};

async function loadPanelSettings() {
  try {
    const response = await chrome.runtime.sendMessage({ type: "GET_SETTINGS" });
    if (response?.settings) {
      panelSettings = {
        showParagraphs: response.settings.showParagraphs ?? true,
        showAbstracts: response.settings.showAbstracts ?? false,
      };
    }
  } catch (err) {
    console.error("Failed to load panel settings:", err);
  }
}

// --- DOM references ---
const content = document.getElementById("content")!;
const btnRecommend = document.getElementById("btn-recommend") as HTMLButtonElement;
const statusDot = document.getElementById("status-dot")!;
const manualInput = document.getElementById("manual-input")!;
const btnToggleManual = document.getElementById("btn-toggle-manual") as HTMLButtonElement;
const manualText = document.getElementById("manual-text") as HTMLTextAreaElement;
const btnManualSubmit = document.getElementById("btn-manual-submit") as HTMLButtonElement;

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

// Check health, load settings, and initialize tracker on load
checkHealth();
loadPanelSettings();
initTracker();

// --- Tracker initialization ---

async function initTracker() {
  const docKey = await getDocKeyFromActiveTab();
  if (!docKey) return;
  const storage = new ChromeCitationStorage();
  tracker = new CitationTracker(storage, docKey);
  await tracker.load();
  renderBibliography();
}

// --- Core functions ---

async function getRecommendations() {
  if (isLoading) return;
  isLoading = true;
  btnRecommend.disabled = true;
  showLoading();

  try {
    const response = await chrome.runtime.sendMessage({ type: "GET_RECOMMENDATIONS" });
    if (response?.error) {
      showExtractionError(response.error);
    } else if (response?.response) {
      await showResults(response.response);
    } else {
      showError("Unexpected response from service worker.");
    }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    showError(message);
  } finally {
    isLoading = false;
    btnRecommend.disabled = false;
  }
}

async function getRecommendationsForText(text: string) {
  if (isLoading) return;
  isLoading = true;
  btnManualSubmit.disabled = true;
  btnRecommend.disabled = true;
  showLoading();

  try {
    const response = await chrome.runtime.sendMessage({ type: "GET_RECOMMENDATIONS_FOR_TEXT", text });
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
    isLoading = false;
    btnManualSubmit.disabled = false;
    btnRecommend.disabled = false;
  }
}

async function checkHealth() {
  try {
    const response = await chrome.runtime.sendMessage({ type: "CHECK_HEALTH" });
    if (response?.response) {
      statusDot.className = "status-dot connected";
      statusDot.title = `Connected -- ${response.response.corpus_size ?? "?"} papers`;
    } else {
      statusDot.className = "status-dot error";
      statusDot.title = response?.error ?? "Not connected";
    }
  } catch {
    statusDot.className = "status-dot error";
    statusDot.title = "Not connected";
  }
}

// --- Rendering ---

function showLoading() {
  content.innerHTML = `
    <div class="loading">
      <div class="spinner"></div>
      <p>Searching your library...</p>
    </div>
  `;
}

function showError(message: string) {
  content.innerHTML = `<div class="error-state">${escapeHtml(message)}</div>`;
}

/**
 * Show an extraction error and auto-expand the manual input fallback.
 */
function showExtractionError(message: string) {
  content.innerHTML = `<div class="error-state">${escapeHtml(message)}</div>`;
  // Auto-expand manual input when extraction fails
  manualInput.classList.remove("collapsed");
  manualText.focus();
}

async function showResults(response: RecommendResponse) {
  await loadPanelSettings();
  const recs = response.recommendations;
  selectedRecs.clear();

  if (!recs || recs.length === 0) {
    content.innerHTML = `<div class="empty-state"><p>No matching papers found.</p></div>`;
    return;
  }

  let html = "";

  // Timing info
  if (response.timing?.total_ms) {
    html += `<div class="timing">${recs.length} results in ${Math.round(response.timing.total_ms)}ms -- ${response.corpus_size} papers</div>`;
  }

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

  // Render bibliography below results
  renderBibliography();
}

function updateSelectionBar() {
  const bar = document.getElementById("selection-bar");
  const count = document.getElementById("selection-count");
  if (!bar || !count) return;

  if (selectedRecs.size > 0) {
    bar.style.display = "flex";
    count.textContent = `${selectedRecs.size} selected`;
    content.classList.add("has-selection");
  } else {
    bar.style.display = "none";
    content.classList.remove("has-selection");
  }
}

async function insertCitation(rec: Recommendation) {
  try {
    const response = await chrome.runtime.sendMessage({
      type: "INSERT_CITATION_REQUEST",
      recommendation: rec,
    });
    if (response?.success === false) {
      showToast("Could not insert -- copied to clipboard");
      const key = rec.bibtex_key ?? rec.paper_id;
      await navigator.clipboard.writeText(key);
    } else if (response?.method === "clipboard") {
      showToast("Copied -- paste with Cmd/Ctrl+V");
    } else {
      showToast("Citation inserted");
    }
    // Track the citation
    if (tracker) {
      await tracker.track([rec]);
      refreshCitedBadges();
      renderBibliography();
    }
  } catch {
    showToast("Insert failed");
  }
}

async function insertMultiCitation() {
  const recs = Array.from(selectedRecs.values());
  if (recs.length === 0) return;

  try {
    const response = await chrome.runtime.sendMessage({
      type: "INSERT_MULTI_CITATION_REQUEST",
      recommendations: recs,
    });
    if (response?.success === false) {
      showToast("Could not insert -- copied to clipboard");
    } else if (response?.method === "clipboard") {
      showToast(`${recs.length} citations copied -- paste with Cmd/Ctrl+V`);
    } else {
      showToast(`${recs.length} citations inserted`);
    }
    // Track all citations
    if (tracker) {
      await tracker.track(recs);
      refreshCitedBadges();
      renderBibliography();
    }
    // Clear selection
    selectedRecs.clear();
    content.querySelectorAll<HTMLInputElement>("[data-action='select']").forEach((cb) => {
      cb.checked = false;
    });
    updateSelectionBar();
  } catch {
    showToast("Insert failed");
  }
}

/** Update "Cited" badges on result cards without re-rendering everything. */
function refreshCitedBadges() {
  if (!tracker) return;
  content.querySelectorAll<HTMLInputElement>("[data-action='select']").forEach((cb) => {
    const recData = cb.getAttribute("data-rec");
    if (!recData) return;
    const rec = JSON.parse(recData) as Recommendation;
    const card = cb.closest(".result-card");
    if (!card) return;
    const headerLeft = card.querySelector(".result-header-left");
    if (!headerLeft) return;
    const existingBadge = headerLeft.querySelector(".cited-badge");
    if (tracker!.isTracked(rec.paper_id) && !existingBadge) {
      const badge = document.createElement("span");
      badge.className = "cited-badge";
      badge.textContent = "Cited";
      headerLeft.appendChild(badge);
    }
  });
}

// --- Bibliography section ---

function renderBibliography() {
  // Remove existing bibliography section
  document.getElementById("bibliography-section")?.remove();

  if (!tracker || tracker.count === 0) return;

  const citations = tracker.getAll();
  const bibHtml = renderBibliographyHTML(citations, CHROME_CLASS_MAP);

  // Wrap in a container with the id for removal on re-render
  const wrapper = document.createElement("div");
  wrapper.id = "bibliography-section";
  wrapper.innerHTML = bibHtml;
  const bibElement = wrapper.firstElementChild as HTMLElement;

  // Append after content area
  document.body.appendChild(bibElement);

  // Attach bibliography event listeners
  bibElement.querySelector(`.${CHROME_CLASS_MAP.bibToggle}`)?.addEventListener("click", () => {
    const bibContent = bibElement.querySelector(`.${CHROME_CLASS_MAP.bibContent}`) as HTMLElement | null;
    const toggle = bibElement.querySelector(`.${CHROME_CLASS_MAP.bibToggle}`);
    if (!bibContent || !toggle) return;
    const isVisible = bibContent.style.display !== "none";
    bibContent.style.display = isVisible ? "none" : "block";
    toggle.classList.toggle("expanded", !isVisible);
  });

  // Export button listeners
  bibElement.querySelectorAll("[data-action='bib-export']").forEach((btn) => {
    btn.addEventListener("click", () => {
      const format = btn.getAttribute("data-format");
      if (!tracker) return;
      const allCitations = tracker.getAll();
      if (format === "bibtex") {
        const text = exportBibTeX(allCitations);
        copyAndDownload(text, "references.bib", "BibTeX copied & downloaded");
      } else if (format === "ris") {
        const text = exportRIS(allCitations);
        copyAndDownload(text, "references.ris", "RIS copied & downloaded");
      } else if (format === "apa") {
        const text = exportFormattedText(allCitations);
        navigator.clipboard.writeText(text).then(() => showToast("APA text copied"));
      }
    });
  });

  bibElement.querySelectorAll("[data-action='bib-remove']").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const paperId = btn.getAttribute("data-paper-id");
      if (!paperId || !tracker) return;
      await tracker.remove(paperId);
      refreshCitedBadges();
      renderBibliography();
    });
  });
}

function copyAndDownload(text: string, filename: string, toastMsg: string) {
  navigator.clipboard.writeText(text).then(() => {
    // Also trigger a download
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    showToast(toastMsg);
  });
}

function showToast(message: string) {
  const existing = document.querySelector(".toast");
  if (existing) existing.remove();

  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 2500);
}
