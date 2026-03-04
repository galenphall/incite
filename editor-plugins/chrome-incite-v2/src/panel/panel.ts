import type { Recommendation, RecommendResponse, TrackedCitation, UIClassMap, Collection } from "@incite/shared";
import {
  CitationTracker,
  exportBibTeX,
  exportRIS,
  exportFormattedText,
  escapeHtml,
  escapeAttr,
  renderResultCardHTML,
  renderBibliographyHTML,
  formatCitation,
} from "@incite/shared";
import { ChromeCitationStorage, getDocKeyFromActiveTab } from "../shared/citation-storage";
import type { EditorType } from "../shared/types";

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
  evidenceToggle: "evidence-toggle",
  evidenceContent: "evidence-content",
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
let panelSettings: { showParagraphs: boolean; showAbstracts: boolean; googleDocsCitationFormat: string } = {
  showParagraphs: true,
  showAbstracts: false,
  googleDocsCitationFormat: "(${first_author}, ${year})",
};
let currentEditorType: EditorType = "unknown";

async function loadPanelSettings() {
  try {
    const response = await chrome.runtime.sendMessage({ type: "GET_SETTINGS" });
    if (response?.settings) {
      panelSettings = {
        showParagraphs: response.settings.showParagraphs ?? true,
        showAbstracts: response.settings.showAbstracts ?? false,
        googleDocsCitationFormat: response.settings.googleDocsCitationFormat ?? "(${first_author}, ${year})",
      };
    }
  } catch (err) {
    console.error("Failed to load panel settings:", err);
  }
}

/** Detect editor type from the active tab URL. */
async function detectEditorType(): Promise<EditorType> {
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

let collections: Collection[] = [];
let selectedCollectionId: string | null = null;

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
  selectedCollectionId = collectionSelect.value || null;
  chrome.storage.sync.set({ incite_collection_id: selectedCollectionId });
});

// Load persisted collection selection
chrome.storage.sync.get("incite_collection_id", (result) => {
  selectedCollectionId = result.incite_collection_id ?? null;
});

// Check health, load settings, detect editor type, and initialize tracker on load
checkHealth();
loadPanelSettings();
initTracker();
detectEditorType().then((et) => { currentEditorType = et; });

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

async function fetchCollections() {
  try {
    const response = await chrome.runtime.sendMessage({ type: "GET_COLLECTIONS" });
    collections = response?.collections ?? [];
    renderCollectionDropdown();
  } catch {
    // Collections are optional
  }
}

function renderCollectionDropdown() {
  const settings = panelSettings as { showParagraphs: boolean; showAbstracts: boolean };
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

async function showResults(response: RecommendResponse, queryText?: string) {
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
    if (currentEditorType === "googledocs") {
      // Google Docs: use REST API insertion
      const citationText = formatCitation(rec, panelSettings.googleDocsCitationFormat);
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
        showToast(response?.error ?? "Insert failed");
        // Fallback: copy to clipboard
        await navigator.clipboard.writeText(citationText);
        showToast("Could not insert -- copied to clipboard");
      } else {
        showToast(`Inserted: ${citationText}`);
      }
    } else {
      // Overleaf/other: use legacy content script insertion
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
    if (currentEditorType === "googledocs") {
      // Google Docs: build segments for grouped citation insertion
      const template = panelSettings.googleDocsCitationFormat;
      const segments: { text: string; paperUrl: string; paperId: string; offsetInFullText: number }[] = [];

      // Build the full grouped citation text with segment tracking
      const parts: string[] = [];
      for (const rec of recs) {
        let citText = formatCitation(rec, template);
        // Strip outer parentheses from individual citations for grouping
        // e.g., "(Smith, 2024)" → "Smith, 2024" so we can wrap the group
        if (parts.length > 0 || recs.length > 1) {
          citText = citText.replace(/^\((.+)\)$/, "$1");
        }
        const paperUrl = rec.doi
          ? `https://doi.org/${rec.doi}`
          : `https://inciteref.com/paper/${rec.paper_id}`;
        segments.push({
          text: citText,
          paperUrl,
          paperId: rec.paper_id,
          offsetInFullText: 0, // Will be calculated below
        });
        parts.push(citText);
      }

      // Join with "; " separator and wrap in parentheses for grouped citations
      const fullText = parts.length > 1 ? `(${parts.join("; ")})` : formatCitation(recs[0], template);

      // Calculate offsets within the full text
      let offset = fullText.startsWith("(") ? 1 : 0; // Account for opening paren
      for (let i = 0; i < segments.length; i++) {
        segments[i].offsetInFullText = offset;
        offset += segments[i].text.length;
        if (i < segments.length - 1) offset += 2; // "; " separator
      }

      const response = await chrome.runtime.sendMessage({
        type: "GDOCS_INSERT_MULTI_CITATION",
        fullText,
        segments,
      });
      if (response?.success === false) {
        showToast(response?.error ?? "Insert failed");
      } else {
        showToast(`${recs.length} citations inserted`);
      }
    } else {
      // Overleaf/other: use legacy content script insertion
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

/** Send GDOCS_REFRESH_CITATIONS message and reconcile tracker with document state. */
async function refreshAndReconcile(refreshText: boolean) {
  if (!tracker) return;
  const trackedIds = tracker.getAll().map((c) => c.paper_id);
  const response = await chrome.runtime.sendMessage({
    type: "GDOCS_REFRESH_CITATIONS",
    trackedPaperIds: trackedIds,
    refreshText,
  });
  if (!response?.success || !response.data) {
    showToast(response?.error ?? "Refresh failed");
    return;
  }
  const data = response.data as {
    foundPaperIds: string[];
    orphanedPaperIds: string[];
    untrackedPaperIds: string[];
    paperMetadata: Array<{
      canonical_id: string;
      title: string;
      abstract: string;
      authors: string[];
      year: number | null;
      doi: string;
      journal: string;
    }>;
    duplicatesFixed: number;
    citationsRefreshed: number;
  };

  // Remove orphaned citations (in tracker but not in document)
  for (const id of data.orphanedPaperIds) {
    await tracker.remove(id);
  }

  // Add untracked citations (in document but not in tracker)
  if (data.untrackedPaperIds.length > 0) {
    const metaMap = new Map(data.paperMetadata.map((p) => [p.canonical_id, p]));
    const toTrack = data.untrackedPaperIds
      .map((id) => {
        const meta = metaMap.get(id);
        return {
          paper_id: id,
          rank: 0,
          score: 0,
          title: meta?.title ?? id,
          authors: meta?.authors,
          year: meta?.year ?? undefined,
          doi: meta?.doi,
          journal: meta?.journal,
          abstract: meta?.abstract,
        };
      });
    await tracker.track(toTrack);
  }

  // Re-render UI
  refreshCitedBadges();
  renderBibliography(true);

  // Show summary toast
  const parts: string[] = [`${data.foundPaperIds.length} citations`];
  if (data.orphanedPaperIds.length > 0) parts.push(`${data.orphanedPaperIds.length} orphaned removed`);
  if (data.untrackedPaperIds.length > 0) parts.push(`${data.untrackedPaperIds.length} added to tracker`);
  if (data.duplicatesFixed > 0) parts.push(`${data.duplicatesFixed} duplicates fixed`);
  if (data.citationsRefreshed > 0) parts.push(`${data.citationsRefreshed} reformatted`);
  showToast(parts.join(", "));
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

function renderBibliography(keepOpen = false) {
  // Check if bibliography was expanded before re-render
  const existingToggle = document.querySelector(`#bibliography-section .${CHROME_CLASS_MAP.bibToggle}`);
  const wasExpanded = keepOpen && existingToggle?.classList.contains("expanded");

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

  // Append the wrapper (which has the id for cleanup) after content area
  wrapper.appendChild(bibElement);
  document.body.appendChild(wrapper);

  // Restore expanded state if it was open before
  if (wasExpanded) {
    const bibContent = bibElement.querySelector(`.${CHROME_CLASS_MAP.bibContent}`) as HTMLElement | null;
    const toggle = bibElement.querySelector(`.${CHROME_CLASS_MAP.bibToggle}`);
    if (bibContent && toggle) {
      bibContent.style.display = "block";
      toggle.classList.add("expanded");
    }
  }

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
      renderBibliography(true);
    });
  });

  // --- Google Docs-specific bibliography actions ---
  if (currentEditorType === "googledocs") {
    const gdocsBar = document.createElement("div");
    gdocsBar.className = "gdocs-bib-actions";
    gdocsBar.innerHTML = `
      <button class="btn-small btn-insert" data-action="gdocs-insert-bib">Insert Bibliography</button>
      <button class="btn-small" data-action="gdocs-refresh">Refresh</button>
      <button class="btn-small" data-action="gdocs-clean">Clean Links</button>
    `;

    // Insert after the export bar
    const exportBar = bibElement.querySelector(`.${CHROME_CLASS_MAP.bibExportBar}`);
    if (exportBar) {
      exportBar.after(gdocsBar);
    } else {
      const bibContent = bibElement.querySelector(`.${CHROME_CLASS_MAP.bibContent}`);
      bibContent?.prepend(gdocsBar);
    }

    gdocsBar.querySelector("[data-action='gdocs-insert-bib']")?.addEventListener("click", async () => {
      if (!tracker) return;
      const citations = tracker.getAll();
      const entries = citations.map((c) => ({
        paperId: c.paper_id,
        formatted: `${c.authors?.join(", ") ?? "Unknown"} (${c.year ?? "n.d."}). ${c.title}.${c.journal ? ` ${c.journal}.` : ""}${c.doi ? ` https://doi.org/${c.doi}` : ""}`,
        url: c.doi ? `https://doi.org/${c.doi}` : undefined,
      }));
      const response = await chrome.runtime.sendMessage({
        type: "GDOCS_INSERT_BIBLIOGRAPHY",
        entries,
      });
      if (response?.success) {
        showToast("Bibliography inserted");
      } else {
        showToast(response?.error ?? "Failed to insert bibliography");
      }
    });

    gdocsBar.querySelector("[data-action='gdocs-refresh']")?.addEventListener("click", async () => {
      if (!tracker) return;
      showToast("Refreshing citations...");
      await refreshAndReconcile(false);
    });


    gdocsBar.querySelector("[data-action='gdocs-clean']")?.addEventListener("click", async () => {
      const response = await chrome.runtime.sendMessage({ type: "GDOCS_CLEAN" });
      if (response?.success) {
        const data = response.data as { cleaned: number } | undefined;
        showToast(`Cleaned ${data?.cleaned ?? 0} inCite markers`);
      } else {
        showToast(response?.error ?? "Clean failed");
      }
    });
  }
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
