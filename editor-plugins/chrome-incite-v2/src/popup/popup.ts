import { loadSettings } from "../shared/settings";
import type { ChromeExtensionSettings } from "../shared/types";

// --- Types from service worker messages ---

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
}

interface Collection {
  id: string;
  name: string;
  color?: string | null;
  item_count: number;
}

interface Tag {
  id: string;
  name: string;
  color?: string | null;
}

interface LibraryCheckResult {
  doi?: string | null;
  title?: string;
  in_library: boolean;
  canonical_id?: string | null;
  collections?: string[];
  tags?: string[];
}

// --- State ---

type PopupState =
  | { kind: "loading" }
  | { kind: "no-paper" }
  | { kind: "not-signed-in" }
  | { kind: "single-paper"; paper: PaperMetadata; check: LibraryCheckResult | null }
  | { kind: "multi-paper"; papers: PaperMetadata[]; checks: LibraryCheckResult[] }
  | { kind: "already-saved"; paper: PaperMetadata; check: LibraryCheckResult }
  | { kind: "saving" }
  | { kind: "success"; savedCount: number; collectionName: string }
  | { kind: "error"; message: string };

let state: PopupState = { kind: "loading" };
let collections: Collection[] = [];
let selectedCollectionId: string | null = null;
let selectedTags: string[] = [];
let tagSuggestions: Tag[] = [];
let showTagInput = false;
let selectedPaperIndices = new Set<number>();
let settings: ChromeExtensionSettings;

const root = document.getElementById("popup-root")!;

// --- Initialization ---

(async function init() {
  settings = await loadSettings();
  render();

  if (!settings.apiToken) {
    state = { kind: "not-signed-in" };
    render();
    return;
  }

  // Get detected papers from service worker
  try {
    const response = await chrome.runtime.sendMessage({ type: "GET_DETECTED_PAPERS" });

    if (!response || !response.papers || response.papers.length === 0) {
      state = { kind: "no-paper" };
      render();
      return;
    }

    // Load collections in parallel with library check
    const [collectionsResp, checkResp] = await Promise.all([
      chrome.runtime.sendMessage({ type: "GET_COLLECTIONS" }),
      chrome.runtime.sendMessage({ type: "CHECK_LIBRARY", papers: response.papers }),
    ]);

    if (collectionsResp?.collections) {
      collections = collectionsResp.collections;
    }

    // Restore last-used collection
    const stored = await chrome.storage.local.get("lastCollectionId");
    if (stored.lastCollectionId) {
      selectedCollectionId = stored.lastCollectionId;
    }

    const papers: PaperMetadata[] = response.papers;
    const checks: LibraryCheckResult[] = checkResp?.results ?? [];

    if (response.type === "single" && papers.length === 1) {
      const check = checks[0] ?? null;
      if (check?.in_library) {
        state = { kind: "already-saved", paper: papers[0], check };
        // Pre-populate tags from the existing library item
        if (check.tags?.length) {
          selectedTags = [...check.tags];
          showTagInput = true;
        }
      } else {
        state = { kind: "single-paper", paper: papers[0], check };
      }
    } else {
      // Multi-paper: pre-select papers not in library
      papers.forEach((_, i) => {
        if (!checks[i]?.in_library) {
          selectedPaperIndices.add(i);
        }
      });
      state = { kind: "multi-paper", papers, checks };
    }
  } catch (err) {
    state = { kind: "error", message: err instanceof Error ? err.message : "Failed to load" };
  }

  render();
})();

// --- Rendering ---

function render() {
  switch (state.kind) {
    case "loading":
      root.innerHTML = renderLoading();
      break;
    case "no-paper":
      root.innerHTML = renderNoPaper();
      break;
    case "not-signed-in":
      root.innerHTML = renderNotSignedIn();
      document.getElementById("open-options")?.addEventListener("click", (e) => {
        e.preventDefault();
        chrome.runtime.openOptionsPage();
      });
      break;
    case "single-paper":
      root.innerHTML = renderSinglePaper(state.paper, state.check);
      bindSinglePaperEvents();
      break;
    case "multi-paper":
      root.innerHTML = renderMultiPaper(state.papers, state.checks);
      bindMultiPaperEvents();
      break;
    case "already-saved":
      root.innerHTML = renderAlreadySaved(state.paper, state.check);
      bindAlreadySavedEvents();
      break;
    case "saving":
      root.innerHTML = renderSaving();
      break;
    case "success":
      root.innerHTML = renderSuccess(state.savedCount, state.collectionName);
      bindSuccessEvents();
      break;
    case "error":
      root.innerHTML = renderError(state.message);
      bindErrorEvents();
      break;
  }
}

function renderLoading(): string {
  return `
    <div class="popup-header">
      <h1>Save to inCite</h1>
    </div>
    <div class="state-loading">
      <div class="spinner"></div>
      <p>Detecting paper...</p>
    </div>
  `;
}

function renderNoPaper(): string {
  return `
    <div class="popup-header">
      <h1>Save to inCite</h1>
    </div>
    <div class="empty-message">
      <p>No paper detected on this page.</p>
      <p>Try visiting a paper on arXiv, PubMed, Google Scholar, or a journal website.</p>
    </div>
  `;
}

function renderNotSignedIn(): string {
  const url = settings.cloudUrl || "https://inciteref.com";
  return `
    <div class="popup-header">
      <h1>Save to inCite</h1>
    </div>
    <div class="empty-message">
      <p>Connect to your inCite account to get started.</p>
      <ol class="setup-steps">
        <li>Copy your API key from <a href="${url}/web/settings" target="_blank">inciteref.com/web/settings</a></li>
        <li>Paste it in <a href="#" id="open-options">extension options</a> (or right-click the extension icon → Options)</li>
      </ol>
    </div>
  `;
}

function renderPaperCard(paper: PaperMetadata): string {
  const authors = paper.authors?.join(", ") ?? "";
  const yearJournal = [paper.year, paper.journal].filter(Boolean).join(" \u00B7 ");
  const doi = paper.doi ? `<div class="paper-doi">DOI: ${escapeHtml(paper.doi)}</div>` : "";

  return `
    <div class="paper-card">
      <div class="paper-title">${escapeHtml(paper.title)}</div>
      ${authors ? `<div class="paper-authors">${escapeHtml(authors)}</div>` : ""}
      ${yearJournal ? `<div class="paper-meta">${escapeHtml(yearJournal)}</div>` : ""}
      ${doi}
    </div>
  `;
}

function renderCollectionPicker(): string {
  const options = collections.map((c) => {
    const selected = c.id === selectedCollectionId ? "selected" : "";
    return `<option value="${escapeHtml(c.id)}" ${selected}>${escapeHtml(c.name)}</option>`;
  }).join("");

  return `
    <div class="form-group">
      <label class="form-label">Collection</label>
      <select class="form-select" id="collection-select">
        <option value="">My Library</option>
        ${options}
      </select>
    </div>
  `;
}

function renderTagInput(): string {
  if (!showTagInput) {
    return `<button class="tags-toggle" id="toggle-tags">+ Add tags</button>`;
  }

  const chips = selectedTags.map((t, i) => `
    <span class="tag-chip">
      ${escapeHtml(t)}
      <button class="tag-chip-remove" data-tag-index="${i}">&times;</button>
    </span>
  `).join("");

  const dropdown = tagSuggestions.length > 0 ? `
    <div class="tag-dropdown" id="tag-dropdown">
      ${tagSuggestions.map((t, i) => `
        <div class="tag-option" data-tag-name="${escapeHtml(t.name)}" data-index="${i}">${escapeHtml(t.name)}</div>
      `).join("")}
    </div>
  ` : "";

  return `
    <div class="form-group tag-autocomplete">
      <label class="form-label">Tags</label>
      <div class="tag-input-wrapper" id="tag-wrapper">
        ${chips}
        <input type="text" class="tag-text-input" id="tag-input" placeholder="Type to add..." autocomplete="off">
      </div>
      ${dropdown}
    </div>
  `;
}

function renderSinglePaper(paper: PaperMetadata, _check: LibraryCheckResult | null): string {
  return `
    <div class="popup-header">
      <h1>Save to inCite</h1>
    </div>
    <div class="popup-state">
      ${renderPaperCard(paper)}
      ${renderCollectionPicker()}
      ${renderTagInput()}
      <div class="popup-actions">
        <button class="btn-secondary" id="btn-cancel">Cancel</button>
        <button class="btn-primary" id="btn-save">Save</button>
      </div>
    </div>
  `;
}

function renderMultiPaper(papers: PaperMetadata[], checks: LibraryCheckResult[]): string {
  const saveable = papers.filter((_, i) => !checks[i]?.in_library).length;
  const selected = selectedPaperIndices.size;

  const items = papers.map((p, i) => {
    const inLibrary = checks[i]?.in_library;
    const checked = selectedPaperIndices.has(i) && !inLibrary;
    const cls = inLibrary ? "multi-paper-item in-library" : "multi-paper-item";
    const yearStr = p.year ? ` (${p.year})` : "";
    const badge = inLibrary ? `<span class="multi-paper-badge">In library</span>` : "";

    return `
      <div class="${cls}" data-index="${i}">
        <input type="checkbox" ${checked ? "checked" : ""} ${inLibrary ? "disabled" : ""} data-index="${i}">
        <div class="multi-paper-info">
          <div class="multi-paper-title">${escapeHtml(p.title)}${yearStr}</div>
          ${p.authors ? `<div class="multi-paper-meta">${escapeHtml(p.authors.slice(0, 3).join(", "))}${p.authors.length > 3 ? " et al." : ""}</div>` : ""}
        </div>
        ${badge}
      </div>
    `;
  }).join("");

  return `
    <div class="popup-header">
      <h1>Save to inCite</h1>
    </div>
    <div class="popup-state">
      <div class="multi-header">
        <span class="multi-count">${papers.length} papers found</span>
        <button class="btn-select-all" id="btn-select-all">Select all (${saveable})</button>
      </div>
      <div class="multi-paper-list">${items}</div>
      ${renderCollectionPicker()}
      <div class="popup-actions">
        <button class="btn-secondary" id="btn-cancel">Cancel</button>
        <button class="btn-primary" id="btn-save" ${selected === 0 ? "disabled" : ""}>Save ${selected} selected</button>
      </div>
    </div>
  `;
}

function renderAlreadySaved(paper: PaperMetadata, check: LibraryCheckResult): string {
  const collectionStr = check.collections?.length
    ? `In: ${check.collections.join(", ")}`
    : "";
  const tagStr = check.tags?.length
    ? `Tags: ${check.tags.join(", ")}`
    : "";

  const url = settings.cloudUrl || "https://inciteref.com";

  return `
    <div class="popup-header">
      <h1>Already in your library</h1>
    </div>
    <div class="popup-state">
      <div class="already-saved">
        <div class="already-saved-title">${escapeHtml(paper.title)}</div>
        ${collectionStr ? `<div class="already-saved-detail">${escapeHtml(collectionStr)}</div>` : ""}
        ${tagStr ? `<div class="already-saved-detail">${escapeHtml(tagStr)}</div>` : ""}
      </div>
      ${renderCollectionPicker()}
      ${renderTagInput()}
      <div class="popup-actions">
        <a href="${url}/library" target="_blank" class="btn-secondary" style="text-decoration: none; text-align: center;">View in Library</a>
        <button class="btn-primary" id="btn-update">Update</button>
        <button class="btn-secondary" id="btn-close">Close</button>
      </div>
    </div>
  `;
}

function renderSaving(): string {
  return `
    <div class="popup-header">
      <h1>Save to inCite</h1>
    </div>
    <div class="state-loading">
      <div class="spinner"></div>
      <p>Saving to library...</p>
    </div>
  `;
}

function renderSuccess(savedCount: number, collectionName: string): string {
  const paperWord = savedCount === 1 ? "paper" : "papers";
  return `
    <div class="popup-header">
      <h1>Saved to inCite</h1>
    </div>
    <div class="success-state">
      <div class="success-icon">&#10003;</div>
      <div class="success-title">${savedCount} ${paperWord} saved</div>
      <div class="success-detail">Added to "${escapeHtml(collectionName)}"</div>
      <div class="success-hint">This paper will now appear in your recommendations when relevant.</div>
      <div class="popup-actions" style="justify-content: center; margin-top: 16px;">
        <button class="btn-primary" id="btn-done">Done</button>
      </div>
    </div>
  `;
}

function renderError(message: string): string {
  return `
    <div class="popup-header">
      <h1>Save to inCite</h1>
    </div>
    <div class="error-state">
      <div class="error-message">${escapeHtml(message)}</div>
      <div class="popup-actions" style="justify-content: center;">
        <button class="btn-secondary" id="btn-retry">Retry</button>
        <button class="btn-secondary" id="btn-close">Close</button>
      </div>
    </div>
  `;
}

// --- Event Binding ---

function bindSinglePaperEvents() {
  document.getElementById("btn-cancel")?.addEventListener("click", () => window.close());

  document.getElementById("btn-save")?.addEventListener("click", async () => {
    if (state.kind !== "single-paper") return;
    await savePapers([state.paper]);
  });

  bindCollectionEvents();
  bindTagEvents();
}

function bindMultiPaperEvents() {
  document.getElementById("btn-cancel")?.addEventListener("click", () => window.close());

  document.getElementById("btn-select-all")?.addEventListener("click", () => {
    if (state.kind !== "multi-paper") return;
    const { papers, checks } = state;
    const allSelected = papers.every((_, i) => checks[i]?.in_library || selectedPaperIndices.has(i));

    if (allSelected) {
      selectedPaperIndices.clear();
    } else {
      papers.forEach((_, i) => {
        if (!checks[i]?.in_library) selectedPaperIndices.add(i);
      });
    }
    render();
  });

  document.querySelectorAll<HTMLInputElement>(".multi-paper-item input[type='checkbox']").forEach((cb) => {
    cb.addEventListener("change", () => {
      const index = parseInt(cb.dataset.index!, 10);
      if (cb.checked) {
        selectedPaperIndices.add(index);
      } else {
        selectedPaperIndices.delete(index);
      }
      // Update button text
      const btn = document.getElementById("btn-save");
      if (btn) {
        btn.textContent = `Save ${selectedPaperIndices.size} selected`;
        (btn as HTMLButtonElement).disabled = selectedPaperIndices.size === 0;
      }
    });
  });

  // Click row to toggle checkbox
  document.querySelectorAll<HTMLDivElement>(".multi-paper-item:not(.in-library)").forEach((row) => {
    row.addEventListener("click", (e) => {
      if ((e.target as HTMLElement).tagName === "INPUT") return;
      const cb = row.querySelector("input[type='checkbox']") as HTMLInputElement;
      cb.checked = !cb.checked;
      cb.dispatchEvent(new Event("change"));
    });
  });

  document.getElementById("btn-save")?.addEventListener("click", async () => {
    if (state.kind !== "multi-paper") return;
    const papersToSave = state.papers.filter((_, i) => selectedPaperIndices.has(i));
    if (papersToSave.length === 0) return;
    await savePapers(papersToSave);
  });

  bindCollectionEvents();
}

function bindAlreadySavedEvents() {
  document.getElementById("btn-close")?.addEventListener("click", () => window.close());

  document.getElementById("btn-update")?.addEventListener("click", async () => {
    if (state.kind !== "already-saved") return;
    const canonicalId = state.check.canonical_id;
    if (!canonicalId) return;

    // Show saving state
    state = { kind: "saving" };
    render();

    try {
      const response = await chrome.runtime.sendMessage({
        type: "UPDATE_LIBRARY_ITEM",
        canonicalId,
        collectionId: selectedCollectionId,
        tags: selectedTags.length > 0 ? selectedTags : undefined,
      });

      if (response?.error) {
        state = { kind: "error", message: response.error };
      } else {
        const collectionName = collections.find((c) => c.id === selectedCollectionId)?.name ?? "My Library";
        state = { kind: "success", savedCount: 1, collectionName };
      }
    } catch (err) {
      state = { kind: "error", message: err instanceof Error ? err.message : "Update failed" };
    }

    render();

    if (state.kind === "success") {
      setTimeout(() => window.close(), 1500);
    }
  });

  bindCollectionEvents();
  bindTagEvents();
}

function bindSuccessEvents() {
  document.getElementById("btn-done")?.addEventListener("click", () => window.close());
}

function bindErrorEvents() {
  document.getElementById("btn-retry")?.addEventListener("click", () => {
    state = { kind: "loading" };
    render();
    // Re-init
    location.reload();
  });
  document.getElementById("btn-close")?.addEventListener("click", () => window.close());
}

function bindCollectionEvents() {
  document.getElementById("collection-select")?.addEventListener("change", (e) => {
    const select = e.target as HTMLSelectElement;
    selectedCollectionId = select.value || null;
    chrome.storage.local.set({ lastCollectionId: selectedCollectionId });
  });
}

function bindTagEvents() {
  const toggleBtn = document.getElementById("toggle-tags");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      showTagInput = true;
      render();
      document.getElementById("tag-input")?.focus();
    });
  }

  const tagInput = document.getElementById("tag-input") as HTMLInputElement | null;
  if (tagInput) {
    tagInput.addEventListener("input", async () => {
      const query = tagInput.value.trim();
      if (query.length < 1) {
        tagSuggestions = [];
        render();
        document.getElementById("tag-input")?.focus();
        return;
      }
      try {
        const resp = await chrome.runtime.sendMessage({ type: "SEARCH_TAGS", query });
        tagSuggestions = (resp?.tags ?? []).filter(
          (t: Tag) => !selectedTags.includes(t.name)
        );
      } catch {
        tagSuggestions = [];
      }
      render();
      // Restore focus and value after re-render
      const newInput = document.getElementById("tag-input") as HTMLInputElement;
      if (newInput) {
        newInput.value = query;
        newInput.focus();
        newInput.setSelectionRange(query.length, query.length);
      }
    });

    tagInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === ",") {
        e.preventDefault();
        const val = tagInput.value.trim().replace(/,$/, "");
        if (val && !selectedTags.includes(val)) {
          selectedTags.push(val);
          tagSuggestions = [];
          render();
          document.getElementById("tag-input")?.focus();
        }
      }
    });
  }

  // Tag dropdown click
  document.querySelectorAll<HTMLDivElement>(".tag-option").forEach((opt) => {
    opt.addEventListener("click", () => {
      const name = opt.dataset.tagName!;
      if (!selectedTags.includes(name)) {
        selectedTags.push(name);
        tagSuggestions = [];
        render();
        document.getElementById("tag-input")?.focus();
      }
    });
  });

  // Tag chip remove
  document.querySelectorAll<HTMLButtonElement>(".tag-chip-remove").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const idx = parseInt(btn.dataset.tagIndex!, 10);
      selectedTags.splice(idx, 1);
      render();
    });
  });
}

// --- Save Action ---

async function savePapers(papers: PaperMetadata[]) {
  state = { kind: "saving" };
  render();

  try {
    const response = await chrome.runtime.sendMessage({
      type: "SAVE_PAPERS",
      papers,
      collectionId: selectedCollectionId,
      tags: selectedTags.length > 0 ? selectedTags : undefined,
      enrich: true,
    });

    if (response?.error) {
      state = { kind: "error", message: response.error };
    } else {
      const savedCount = (response?.saved?.length ?? 0) + (response?.already_existed?.length ?? 0);
      const collectionName = collections.find((c) => c.id === selectedCollectionId)?.name ?? "My Library";
      state = { kind: "success", savedCount, collectionName };
    }
  } catch (err) {
    state = { kind: "error", message: err instanceof Error ? err.message : "Save failed" };
  }

  render();

  if (state.kind === "success") {
    setTimeout(() => window.close(), 1500);
  }
}

// --- Utilities ---

function escapeHtml(text: string): string {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
