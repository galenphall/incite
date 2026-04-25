/**
 * Shared mutable state for the side panel.
 *
 * Centralized here so sub-modules can import state directly
 * without threading a context object through every function call.
 * All state is module-scoped -- esbuild bundles everything into a
 * single IIFE, so these are effectively private globals.
 */
import type { Recommendation, Collection, UIClassMap } from "@incite/shared";
import { CitationTracker } from "@incite/shared";
import type { EditorType } from "../shared/types";

// --- Chrome-specific class map for shared rendering functions ---

export const CHROME_CLASS_MAP: UIClassMap = {
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

// --- Mutable state ---

export let isLoading = false;
export function setIsLoading(v: boolean): void {
  isLoading = v;
}

export const selectedRecs = new Map<string, Recommendation>();

export let tracker: CitationTracker | null = null;
export function setTracker(t: CitationTracker | null): void {
  tracker = t;
}

export let panelSettings = {
  showParagraphs: true,
  showAbstracts: false,
  googleDocsCitationFormat: "(${first_author}, ${year})",
};
export function setPanelSettings(s: typeof panelSettings): void {
  panelSettings = s;
}

export let currentEditorType: EditorType = "unknown";
export function setCurrentEditorType(et: EditorType): void {
  currentEditorType = et;
}

export let collections: Collection[] = [];
export function setCollections(c: Collection[]): void {
  collections = c;
}

export let selectedCollectionId: string | null = null;
export function setSelectedCollectionId(id: string | null): void {
  selectedCollectionId = id;
}
