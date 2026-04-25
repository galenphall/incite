/**
 * Citation insertion, multi-citation grouping, document reconciliation,
 * and cited-badge updates for the side panel.
 */
import type { Recommendation } from "@incite/shared";
import { formatCitation } from "@incite/shared";
import {
  selectedRecs,
  tracker,
  panelSettings,
  currentEditorType,
} from "./panel-state";
import { renderBibliography } from "./panel-bibliography";
import { showToast } from "./panel";

// --- DOM references (set by init) ---
let content: HTMLElement;

/** Initialize DOM references. Must be called before any other function. */
export function initCitationsDom(refs: { content: HTMLElement }): void {
  content = refs.content;
}

// --- Selection bar ---

export function updateSelectionBar(): void {
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

// --- Single citation insertion ---

export async function insertCitation(rec: Recommendation): Promise<void> {
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

// --- Multi-citation insertion ---

export async function insertMultiCitation(): Promise<void> {
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
        // e.g., "(Smith, 2024)" -> "Smith, 2024" so we can wrap the group
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

// --- Document reconciliation ---

/**
 * Reconcile the panel's citation tracker with the actual document state.
 *
 * The tracker (chrome.storage.local) and the document can drift apart:
 * - **Orphaned citations**: The user manually deleted a citation from the doc,
 *   but the tracker still lists it. These are removed from the tracker.
 * - **Untracked citations**: The user pasted a citation from another doc, or
 *   a citation was inserted before the tracker existed. These are added to
 *   the tracker by fetching metadata from the API.
 *
 * Also handles copy-paste duplicates (citations with duplicate hidden bookmark
 * ranges) and optional citation text reformatting to match the current format.
 */
export async function refreshAndReconcile(refreshText: boolean): Promise<void> {
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

  // Remove orphaned citations (in tracker but deleted from document by user)
  for (const id of data.orphanedPaperIds) {
    await tracker.remove(id);
  }

  // Add untracked citations (in document but not in tracker — e.g. pasted from another doc)
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
  const summaryParts: string[] = [`${data.foundPaperIds.length} citations`];
  if (data.orphanedPaperIds.length > 0) summaryParts.push(`${data.orphanedPaperIds.length} orphaned removed`);
  if (data.untrackedPaperIds.length > 0) summaryParts.push(`${data.untrackedPaperIds.length} added to tracker`);
  if (data.duplicatesFixed > 0) summaryParts.push(`${data.duplicatesFixed} duplicates fixed`);
  if (data.citationsRefreshed > 0) summaryParts.push(`${data.citationsRefreshed} reformatted`);
  showToast(summaryParts.join(", "));
}

// --- Badge updates ---

/** Update "Cited" badges on result cards without re-rendering everything. */
export function refreshCitedBadges(): void {
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
