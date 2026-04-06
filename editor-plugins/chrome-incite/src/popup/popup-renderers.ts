import { escapeHtml } from "@incite/shared";
import type { PaperMetadata, LibraryCheckResult } from "./popup-state";
import {
  collections,
  selectedCollectionId,
  selectedTags,
  tagSuggestions,
  showTagInput,
  selectedPaperIndices,
  settings,
} from "./popup-state";

// --- Rendering functions ---

export function renderLoading(): string {
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

export function renderNoPaper(): string {
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

export function renderNotSignedIn(): string {
  const url = settings.cloudUrl || "https://inciteref.com";
  return `
    <div class="popup-header">
      <h1>Save to inCite</h1>
    </div>
    <div class="empty-message">
      <p>Connect to your inCite account to get started.</p>
      <ol class="setup-steps">
        <li>Copy your API key from <a href="${url}/web/settings" target="_blank">inciteref.com/web/settings</a></li>
        <li>Paste it in <a href="#" id="open-options">extension options</a> (or right-click the extension icon \u2192 Options)</li>
      </ol>
    </div>
  `;
}

export function renderPaperCard(paper: PaperMetadata): string {
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

export function renderCollectionPicker(): string {
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

export function renderTagInput(): string {
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

export function renderSinglePaper(paper: PaperMetadata, _check: LibraryCheckResult | null): string {
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

export function renderMultiPaper(papers: PaperMetadata[], checks: LibraryCheckResult[]): string {
  const isExactMatch = (i: number) => checks[i]?.in_library && checks[i]?.match_type !== "fuzzy_title";
  const isFuzzyMatch = (i: number) => checks[i]?.in_library && checks[i]?.match_type === "fuzzy_title";
  const saveable = papers.filter((_, i) => !isExactMatch(i)).length;
  const selected = selectedPaperIndices.size;

  const items = papers.map((p, i) => {
    const exact = isExactMatch(i);
    const fuzzy = isFuzzyMatch(i);
    const checked = selectedPaperIndices.has(i) && !exact;
    const cls = exact ? "multi-paper-item in-library" : fuzzy ? "multi-paper-item fuzzy-match" : "multi-paper-item";
    const yearStr = p.year ? ` (${p.year})` : "";
    const badge = exact
      ? `<span class="multi-paper-badge">In library</span>`
      : fuzzy
        ? `<span class="multi-paper-badge likely">Likely in library</span>`
        : "";

    return `
      <div class="${cls}" data-index="${i}">
        <input type="checkbox" ${checked ? "checked" : ""} ${exact ? "disabled" : ""} data-index="${i}">
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

export function renderAlreadySaved(paper: PaperMetadata, check: LibraryCheckResult): string {
  const collectionStr = check.collections?.length
    ? `In: ${check.collections.map((c) => c.name).join(", ")}`
    : "";
  const tagStr = check.tags?.length
    ? `Tags: ${check.tags.map((t) => t.name).join(", ")}`
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

export function renderLikelySaved(paper: PaperMetadata, check: LibraryCheckResult): string {
  const collectionStr = check.collections?.length
    ? `In: ${check.collections.map((c) => c.name).join(", ")}`
    : "";
  const tagStr = check.tags?.length
    ? `Tags: ${check.tags.map((t) => t.name).join(", ")}`
    : "";

  const url = settings.cloudUrl || "https://inciteref.com";

  return `
    <div class="popup-header">
      <h1>Likely already in your library</h1>
    </div>
    <div class="popup-state">
      <div class="likely-saved">
        <div class="likely-saved-comparison">
          <div class="likely-saved-label">This page:</div>
          <div class="likely-saved-value">${escapeHtml(paper.title)}</div>
          <div class="likely-saved-label">In your library:</div>
          <div class="likely-saved-value">${escapeHtml(check.library_title ?? check.title ?? "")}</div>
        </div>
        ${collectionStr ? `<div class="likely-saved-detail">${escapeHtml(collectionStr)}</div>` : ""}
        ${tagStr ? `<div class="likely-saved-detail">${escapeHtml(tagStr)}</div>` : ""}
      </div>
      ${renderCollectionPicker()}
      ${renderTagInput()}
      <div class="popup-actions">
        <a href="${url}/library" target="_blank" class="btn-secondary" style="text-decoration: none; text-align: center;">View in Library</a>
        <button class="btn-secondary" id="btn-update">Update</button>
        <button class="btn-primary" id="btn-save-new">Save as New</button>
      </div>
    </div>
  `;
}

export function renderSaving(): string {
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

export function renderSuccess(savedCount: number, collectionName: string): string {
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

export function renderError(message: string): string {
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
