/**
 * Pure HTML rendering functions for the overlay rail and command palette popup.
 * Returns HTML strings — no DOM manipulation.
 */
import type { Recommendation } from "@incite/shared";
import { escapeHtml, renderHighlightedTextHTML } from "@incite/shared";
import {
  ICON_RECOMMEND,
  ICON_ADD_PAPER,
  ICON_LIBRARY,
  ICON_EXPAND,
  ICON_CLOSE,
  ICON_SETTINGS,
  ICON_LOGO,
} from "./overlay-icons";

export interface OverlayState {
  mode: "rail" | "popup";
  results: Recommendation[];
  loading: boolean;
  selectedIndex: number;
  expandedEvidence: Set<number>;
  timing: { total_ms: number } | null;
  corpusSize: number;
  error: string | null;
  collectionId: string | null;
  citedPaperIds: Set<string>;
}

// ─── Rail ───

export function renderRail(): string {
  return `<div class="incite-rail">
    <button class="rail-btn" data-action="recommend" aria-label="Get recommendations" title="Get recommendations">
      ${ICON_RECOMMEND}
    </button>
    <button class="rail-btn" data-action="add-paper" aria-label="Save paper to library" title="Save paper">
      ${ICON_ADD_PAPER}
    </button>
    <button class="rail-btn" data-action="library" aria-label="Browse library" title="Browse library">
      ${ICON_LIBRARY}
    </button>
    <div class="rail-spacer"></div>
    <div class="rail-divider"></div>
    <button class="rail-btn" data-action="settings" aria-label="Settings" title="Settings">
      ${ICON_SETTINGS}
    </button>
    <button class="rail-btn" data-action="expand" aria-label="Expand sidebar" title="Expand sidebar">
      ${ICON_EXPAND}
    </button>
  </div>`;
}

// ─── Popup ───

export function renderPopup(state: OverlayState): string {
  let body: string;

  if (state.loading) {
    body = renderPopupLoading();
  } else if (state.error) {
    body = renderPopupError(state.error);
  } else if (state.results.length === 0) {
    body = renderPopupEmpty();
  } else {
    body = renderPopupResults(state);
  }

  const statusText = renderStatusBar(state);

  return `<div class="incite-popup" tabindex="-1" role="listbox" aria-label="Citation recommendations">
    <div class="popup-header">
      <div class="popup-logo">
        ${ICON_LOGO}
        <span>inCite</span>
      </div>
      <button class="popup-close" data-action="close-popup" aria-label="Close">
        ${ICON_CLOSE}
      </button>
    </div>
    ${statusText}
    <div class="popup-results">
      ${body}
    </div>
    ${renderHintBar()}
  </div>`;
}

// ─── Status Bar ───

function renderStatusBar(state: OverlayState): string {
  if (state.loading) {
    return `<div class="popup-status">Searching...</div>`;
  }
  if (state.error || state.results.length === 0) {
    return "";
  }
  const count = state.results.length;
  const ms = state.timing?.total_ms ? `in ${Math.round(state.timing.total_ms)}ms` : "";
  const corpus = state.corpusSize ? ` \u2014 ${state.corpusSize.toLocaleString()} papers` : "";
  return `<div class="popup-status">${count} result${count !== 1 ? "s" : ""} ${ms}${corpus}</div>`;
}

// ─── Result Rows ───

function renderPopupResults(state: OverlayState): string {
  let html = "";
  for (let i = 0; i < state.results.length; i++) {
    const rec = state.results[i];
    html += renderPopupRow(rec, i, state);
    if (state.expandedEvidence.has(i)) {
      html += renderRowEvidence(rec);
    }
  }
  return html;
}

function renderPopupRow(rec: Recommendation, index: number, state: OverlayState): string {
  const isSelected = index === state.selectedIndex;
  const isCited = state.citedPaperIds.has(rec.paper_id);
  const selectedClass = isSelected ? " selected" : "";
  const confidence = rec.confidence ?? rec.score;

  let html = `<div class="result-row${selectedClass}" data-index="${index}" role="option" aria-selected="${isSelected}">`;

  // Relevance dots
  html += renderRelevanceDots(confidence);

  // Content
  html += `<div class="row-content">`;
  html += `<div class="row-title">${escapeHtml(rec.title)}</div>`;

  const meta: string[] = [];
  if (rec.authors && rec.authors.length > 0) {
    const lastName = rec.authors[0].split(" ").pop() ?? rec.authors[0];
    meta.push(rec.authors.length > 1 ? `${lastName} et al.` : lastName);
  }
  if (rec.year) meta.push(String(rec.year));
  if (meta.length > 0) {
    html += `<div class="row-meta">${escapeHtml(meta.join(", "))}</div>`;
  }
  html += `</div>`;

  // Badges
  html += `<div class="row-badges">`;
  if (isCited) {
    html += `<span class="cited-badge" aria-label="Already cited">CITED</span>`;
  }
  html += `<span class="insert-hint">\u21b5</span>`;
  html += `</div>`;

  html += `</div>`;
  return html;
}

// ─── Relevance Dots ───

export function renderRelevanceDots(score: number): string {
  const filled = score >= 0.55 ? 5 : score >= 0.45 ? 4 : score >= 0.35 ? 3 : score >= 0.25 ? 2 : 1;
  let html = `<span class="relevance-dots" aria-label="Relevance: ${filled} out of 5">`;
  for (let i = 0; i < 5; i++) {
    html += `<span class="dot${i < filled ? " filled" : ""}"></span>`;
  }
  html += `</span>`;
  return html;
}

// ─── Evidence ───

function renderRowEvidence(rec: Recommendation): string {
  const snippets = rec.matched_paragraphs ?? (rec.matched_paragraph ? [{ text: rec.matched_paragraph, score: 0 }] : []);
  if (snippets.length === 0) return "";

  let html = `<div class="row-evidence">`;
  for (let i = 0; i < snippets.length; i++) {
    const snippet = snippets[i];
    const cls = i === 0 ? "evidence" : "evidence evidence-secondary";
    const badge = snippet.score > 0
      ? `<span class="evidence-score">${Math.round(snippet.score * 100)}%</span> `
      : "";
    html += `<div class="${cls}">${badge}${renderHighlightedTextHTML(snippet.text, 250)}</div>`;
  }
  html += `</div>`;
  return html;
}

// ─── Loading Skeleton ───

function renderPopupLoading(): string {
  let html = "";
  for (let i = 0; i < 5; i++) {
    html += `<div class="skeleton-row">
      <div class="skeleton-dots">${"<span class=\"skeleton-dot\"></span>".repeat(5)}</div>
      <div class="skeleton-lines">
        <div class="skeleton-line skeleton-title"></div>
        <div class="skeleton-line skeleton-meta"></div>
      </div>
    </div>`;
  }
  return html;
}

// ─── Empty & Error ───

function renderPopupEmpty(): string {
  return `<div class="popup-empty">
    <p>No matching papers found.</p>
    <p>Select text in your document and try again.</p>
  </div>`;
}

function renderPopupError(message: string): string {
  return `<div class="popup-error">${escapeHtml(message)}</div>`;
}

// ─── Hint Bar ───

function renderHintBar(): string {
  return `<div class="popup-hints">
    <span><kbd>\u2191</kbd><kbd>\u2193</kbd> navigate</span>
    <span><kbd>Enter</kbd> insert</span>
    <span><kbd>Tab</kbd> evidence</span>
    <span><kbd>Esc</kbd> close</span>
  </div>`;
}
