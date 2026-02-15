import type { Recommendation, RecommendResponse, TrackedCitation } from "@incite/shared";
import { CitationTracker, exportBibTeX, exportRIS, exportFormattedText } from "@incite/shared";
import { ChromeCitationStorage, getDocKeyFromActiveTab } from "../shared/citation-storage";

// --- State ---
let isLoading = false;
const selectedRecs = new Map<string, Recommendation>();
let tracker: CitationTracker | null = null;

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
chrome.runtime.sendMessage({ type: "PANEL_READY" }).catch(() => {});

// Check health and initialize tracker on load
checkHealth();
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
      showResults(response.response);
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
      showResults(response.response);
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

function showResults(response: RecommendResponse) {
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
    html += renderResultCard(rec);
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
  } else {
    bar.style.display = "none";
  }
}

function renderResultCard(rec: Recommendation): string {
  const confidence = rec.confidence ?? rec.score;
  const confClass = confidence >= 0.55 ? "confidence-high" : confidence >= 0.35 ? "confidence-medium" : "confidence-low";
  const confLabel = `${Math.round(confidence * 100)}%`;
  const isCited = tracker?.isTracked(rec.paper_id) ?? false;

  let html = `<div class="result-card">`;

  // Header: checkbox + rank + badges
  html += `<div class="result-header">`;
  html += `<div class="result-header-left">`;
  html += `<input type="checkbox" class="select-checkbox" data-action="select" data-rec='${escapeAttr(JSON.stringify(rec))}' />`;
  html += `<span class="rank-badge">#${rec.rank}</span>`;
  if (isCited) {
    html += `<span class="cited-badge">Cited</span>`;
  }
  html += `</div>`;
  html += `<span class="confidence-badge ${confClass}">${confLabel}</span>`;
  html += `</div>`;

  // Title
  html += `<div class="result-title">${escapeHtml(rec.title)}</div>`;

  // Authors + year
  const meta: string[] = [];
  if (rec.authors && rec.authors.length > 0) {
    const names = rec.authors.slice(0, 3).join(", ");
    meta.push(rec.authors.length > 3 ? names + " et al." : names);
  }
  if (rec.year) meta.push(`(${rec.year})`);
  if (meta.length > 0) {
    html += `<div class="result-meta">${escapeHtml(meta.join(" "))}</div>`;
  }

  // Evidence paragraphs
  if (rec.matched_paragraphs && rec.matched_paragraphs.length > 0) {
    for (let i = 0; i < rec.matched_paragraphs.length; i++) {
      const snippet = rec.matched_paragraphs[i];
      const text = snippet.text.length > 300 ? snippet.text.slice(0, 300) + "..." : snippet.text;
      const cls = i === 0 ? "evidence" : "evidence evidence-secondary";
      const badge = snippet.score != null
        ? `<span class="evidence-score">${Math.round(snippet.score * 100)}%</span> `
        : "";
      html += `<div class="${cls}">${badge}${escapeHtml(text)}</div>`;
    }
  } else if (rec.matched_paragraph) {
    const text = rec.matched_paragraph.length > 300
      ? rec.matched_paragraph.slice(0, 300) + "..."
      : rec.matched_paragraph;
    html += `<div class="evidence">${escapeHtml(text)}</div>`;
  }

  // Abstract
  if (rec.abstract) {
    html += `<div class="result-abstract">${escapeHtml(rec.abstract)}</div>`;
  }

  // Actions
  const recJson = escapeAttr(JSON.stringify(rec));
  const bibtexKey = rec.bibtex_key ?? rec.paper_id;
  html += `<div class="result-actions">`;
  html += `<button class="btn-small btn-insert" data-action="insert" data-rec='${recJson}'>Insert</button>`;
  html += `<button class="btn-small" data-action="copy" data-copy="${escapeAttr(bibtexKey)}">Copy Key</button>`;
  html += `</div>`;

  html += `</div>`;
  return html;
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

  let html = `<div id="bibliography-section" class="bibliography-section">`;
  html += `<button class="bib-toggle" id="bib-toggle">`;
  html += `Bibliography (${citations.length} citation${citations.length !== 1 ? "s" : ""})`;
  html += `<span class="toggle-arrow">&#9662;</span>`;
  html += `</button>`;

  html += `<div class="bib-content" id="bib-content" style="display:none;">`;

  // Export buttons
  html += `<div class="bib-export-bar">`;
  html += `<button class="btn-small" id="bib-export-bibtex">BibTeX</button>`;
  html += `<button class="btn-small" id="bib-export-ris">RIS</button>`;
  html += `<button class="btn-small" id="bib-export-apa">APA</button>`;
  html += `</div>`;

  // Citation list
  html += `<div class="bib-list">`;
  for (const cite of citations) {
    const authorStr = cite.authors.length > 0
      ? cite.authors.length > 2
        ? cite.authors[0].split(" ").pop() + " et al."
        : cite.authors.map((a) => a.split(" ").pop()).join(" & ")
      : "";
    const yearStr = cite.year != null ? ` (${cite.year})` : "";

    html += `<div class="bib-item" data-paper-id="${escapeAttr(cite.paper_id)}">`;
    html += `<div class="bib-item-text">`;
    html += `<span class="bib-item-authors">${escapeHtml(authorStr + yearStr)}</span> `;
    html += `<span class="bib-item-title">${escapeHtml(cite.title)}</span>`;
    html += `</div>`;
    html += `<button class="bib-remove" data-action="bib-remove" data-paper-id="${escapeAttr(cite.paper_id)}" title="Remove">&times;</button>`;
    html += `</div>`;
  }
  html += `</div>`;

  html += `</div>`; // bib-content
  html += `</div>`; // bibliography-section

  // Append after content area
  document.body.appendChild(createElementFromHTML(html));

  // Attach bibliography event listeners
  document.getElementById("bib-toggle")?.addEventListener("click", () => {
    const bibContent = document.getElementById("bib-content");
    const toggle = document.getElementById("bib-toggle");
    if (!bibContent || !toggle) return;
    const isVisible = bibContent.style.display !== "none";
    bibContent.style.display = isVisible ? "none" : "block";
    toggle.classList.toggle("expanded", !isVisible);
  });

  document.getElementById("bib-export-bibtex")?.addEventListener("click", () => {
    const text = exportBibTeX(tracker!.getAll());
    copyAndDownload(text, "references.bib", "BibTeX copied & downloaded");
  });

  document.getElementById("bib-export-ris")?.addEventListener("click", () => {
    const text = exportRIS(tracker!.getAll());
    copyAndDownload(text, "references.ris", "RIS copied & downloaded");
  });

  document.getElementById("bib-export-apa")?.addEventListener("click", () => {
    const text = exportFormattedText(tracker!.getAll());
    navigator.clipboard.writeText(text).then(() => showToast("APA text copied"));
  });

  document.querySelectorAll("[data-action='bib-remove']").forEach((btn) => {
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

function createElementFromHTML(html: string): HTMLElement {
  const wrapper = document.createElement("div");
  wrapper.innerHTML = html.trim();
  return wrapper.firstElementChild as HTMLElement;
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

// --- Utilities ---

function escapeHtml(text: string): string {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function escapeAttr(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/'/g, "&#39;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
