/**
 * Results rendering for the Word add-in.
 *
 * Uses shared renderResultCardHTML() for card content, then wires up
 * DOM event listeners for insert/copy/select actions.
 */

import type { Recommendation } from "@incite/shared";
import type { UIClassMap } from "@incite/shared";
import { renderResultCardHTML, renderHighlightedTextHTML } from "@incite/shared";
import { CitationTracker } from "@incite/shared";

/** Word-specific CSS class map matching taskpane.css. */
export const WORD_CLASS_MAP: Partial<UIClassMap> = {
	resultCard: "mc-result",
	resultHeader: "mc-result-header",
	resultHeaderLeft: "mc-result-header-left",
	selectCheckbox: "mc-select-checkbox",
	rankBadge: "mc-rank",
	citedBadge: "mc-cited-badge",
	confidenceBadge: "mc-confidence",
	confidenceHigh: "mc-confidence-high",
	confidenceMid: "mc-confidence-mid",
	confidenceLow: "mc-confidence-low",
	resultTitle: "mc-title",
	resultMeta: "mc-meta",
	evidence: "mc-paragraph",
	resultActions: "mc-actions",
	insertBtn: "mc-insert-btn",
	copyBtn: "mc-copy-btn",
};

export interface ResultsCallbacks {
	onInsert: (rec: Recommendation) => void;
	onCopy: (rec: Recommendation, btn: HTMLButtonElement) => void;
	onSelect: (rec: Recommendation, selected: boolean) => void;
	onInsertSelected: () => void;
	onClearSelected: () => void;
}

/**
 * Render the selection bar at the top of the results container.
 */
export function renderSelectionBar(
	container: HTMLElement,
	selectedCount: number,
	callbacks: Pick<ResultsCallbacks, "onInsertSelected" | "onClearSelected">,
): void {
	const existing = document.getElementById("selectionBar");
	if (existing) existing.remove();

	if (selectedCount === 0) return;

	const bar = document.createElement("div");
	bar.id = "selectionBar";
	bar.className = "mc-selection-bar";

	const label = document.createElement("span");
	label.textContent = `${selectedCount} selected`;
	bar.appendChild(label);

	const insertBtn = document.createElement("button");
	insertBtn.className = "mc-insert-btn";
	insertBtn.textContent = "Insert Selected";
	insertBtn.addEventListener("click", () => callbacks.onInsertSelected());
	bar.appendChild(insertBtn);

	const clearBtn = document.createElement("button");
	clearBtn.className = "mc-copy-btn";
	clearBtn.textContent = "Clear";
	clearBtn.addEventListener("click", () => callbacks.onClearSelected());
	bar.appendChild(clearBtn);

	container.insertBefore(bar, container.firstChild);
}

/**
 * Render all recommendation results into the container.
 * Uses shared renderResultCardHTML for card content, then wires DOM events.
 */
export function renderResults(
	results: Recommendation[],
	container: HTMLElement,
	options: {
		showParagraphs: boolean;
		selectedRecs: Map<string, Recommendation>;
		tracker?: CitationTracker;
	},
	callbacks: ResultsCallbacks,
): void {
	container.innerHTML = "";

	// Toggle has-selection class to hide individual insert buttons
	if (options.selectedRecs.size > 0) {
		container.classList.add("has-selection");
	} else {
		container.classList.remove("has-selection");
	}

	// Selection bar at the top
	if (options.selectedRecs.size > 0) {
		renderSelectionBar(container, options.selectedRecs.size, callbacks);
	}

	for (const rec of results) {
		const cardHTML = renderResultCardHTML(
			rec,
			{
				showParagraphs: options.showParagraphs,
				isCited: options.tracker?.isTracked(rec.paper_id) ?? false,
			},
			WORD_CLASS_MAP,
		);

		const wrapper = document.createElement("div");
		wrapper.innerHTML = cardHTML;
		const card = wrapper.firstElementChild as HTMLElement;

		// Wire up checkbox
		const checkbox = card.querySelector(`.mc-select-checkbox`) as HTMLInputElement | null;
		if (checkbox) {
			checkbox.checked = options.selectedRecs.has(rec.paper_id);
			checkbox.addEventListener("change", () => {
				callbacks.onSelect(rec, checkbox.checked);
			});
		}

		// Wire up insert button
		const insertBtn = card.querySelector(`.mc-insert-btn`) as HTMLButtonElement | null;
		if (insertBtn) {
			insertBtn.textContent = "+ Insert";
			insertBtn.addEventListener("click", () => callbacks.onInsert(rec));
		}

		// Wire up copy button
		const copyBtn = card.querySelector(`.mc-copy-btn`) as HTMLButtonElement | null;
		if (copyBtn) {
			copyBtn.textContent = "Copy";
			copyBtn.addEventListener("click", () => callbacks.onCopy(rec, copyBtn));
		}

		container.appendChild(card);
	}
}
