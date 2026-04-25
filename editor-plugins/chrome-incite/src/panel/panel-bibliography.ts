/**
 * Bibliography rendering, export actions, and Google Docs-specific
 * bibliography operations for the side panel.
 */
import {
  exportBibTeX,
  exportRIS,
  exportFormattedText,
  renderBibliographyHTML,
} from "@incite/shared";
import {
  CHROME_CLASS_MAP,
  tracker,
  currentEditorType,
} from "./panel-state";
import { refreshCitedBadges, refreshAndReconcile } from "./panel-citations";
import { showToast } from "./panel";

// --- Bibliography rendering ---

export function renderBibliography(keepOpen = false): void {
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
      const bibCitations = tracker.getAll();
      const entries = bibCitations.map((c) => ({
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

// --- Export helpers ---

export function copyAndDownload(text: string, filename: string, toastMsg: string): void {
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
