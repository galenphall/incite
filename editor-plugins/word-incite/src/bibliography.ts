/**
 * Bibliography rendering for the Word add-in.
 *
 * Renders the bibliography section with export buttons (BibTeX, RIS, APA)
 * and a per-citation remove button.
 */

import { exportBibTeX, exportRIS, exportFormattedText, CitationTracker } from "@incite/shared";
import type { TrackedCitation } from "@incite/shared";

/** Helper to get a DOM element by id. */
function getEl(id: string): HTMLElement {
	return document.getElementById(id)!;
}

/**
 * Render the bibliography section into the DOM.
 * Creates the section if it doesn't exist, appends after #results.
 */
export function renderBibliography(
	tracker: CitationTracker,
	onRemove: (paperId: string) => Promise<void>,
	onInsert: () => Promise<void>,
): void {
	let section = document.getElementById("bibliographySection");
	if (!section) {
		section = document.createElement("div");
		section.id = "bibliographySection";
		// Append after the results area
		const results = getEl("results");
		results.parentElement!.appendChild(section);
	}
	section.innerHTML = "";

	const citations = tracker.getAll();
	if (citations.length === 0) {
		section.style.display = "none";
		return;
	}
	section.style.display = "";

	// Collapsible header
	const header = document.createElement("div");
	header.className = "mc-bib-header";
	header.textContent = `Bibliography (${citations.length} citation${citations.length !== 1 ? "s" : ""})`;
	let expanded = true;
	const body = document.createElement("div");

	header.addEventListener("click", () => {
		expanded = !expanded;
		body.style.display = expanded ? "" : "none";
		header.classList.toggle("collapsed", !expanded);
	});
	section.appendChild(header);

	// Export buttons
	const exportRow = document.createElement("div");
	exportRow.className = "mc-bib-export";

	const bibtexBtn = createExportButton("BibTeX", () => exportBibTeX(citations));
	exportRow.appendChild(bibtexBtn);

	const risBtn = createExportButton("RIS", () => exportRIS(citations));
	exportRow.appendChild(risBtn);

	const apaBtn = createExportButton("Copy APA", () => exportFormattedText(citations));
	exportRow.appendChild(apaBtn);

	const insertBtn = document.createElement("button");
	insertBtn.className = "mc-copy-btn";
	insertBtn.textContent = "Insert";
	insertBtn.addEventListener("click", async () => {
		await onInsert();
		insertBtn.textContent = "Inserted!";
		setTimeout(() => {
			insertBtn.textContent = "Insert";
		}, 1500);
	});
	exportRow.appendChild(insertBtn);

	body.appendChild(exportRow);

	// Citation list
	for (const cite of citations) {
		const row = document.createElement("div");
		row.className = "mc-bib-entry";

		const info = document.createElement("span");
		info.className = "mc-bib-info";
		const authorStr =
			cite.authors.length > 0
				? cite.authors[0].split(",")[0].split(" ").pop() ?? ""
				: "";
		const yearStr = cite.year != null ? ` (${cite.year})` : "";
		info.textContent = `${authorStr}${yearStr} - ${cite.title}`;
		row.appendChild(info);

		const removeBtn = document.createElement("button");
		removeBtn.className = "mc-remove-btn";
		removeBtn.textContent = "x";
		removeBtn.title = "Remove from bibliography";
		removeBtn.addEventListener("click", () => onRemove(cite.paper_id));
		row.appendChild(removeBtn);

		body.appendChild(row);
	}

	section.appendChild(body);
}

/** Create a clipboard-copy export button. */
function createExportButton(
	label: string,
	getContent: () => string,
): HTMLButtonElement {
	const btn = document.createElement("button");
	btn.className = "mc-copy-btn";
	btn.textContent = label;
	btn.addEventListener("click", () => {
		navigator.clipboard.writeText(getContent()).then(() => {
			btn.textContent = "Copied!";
			setTimeout(() => {
				btn.textContent = label;
			}, 1500);
		});
	});
	return btn;
}
