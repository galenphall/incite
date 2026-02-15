import { ItemView, WorkspaceLeaf, setIcon } from "obsidian";
import type { Recommendation, InCiteSettings, TrackedCitation } from "./types";

export const VIEW_TYPE_INCITE = "incite-sidebar";

export class InCiteSidebarView extends ItemView {
	private results: Recommendation[] = [];
	private settings: InCiteSettings;
	private onInsert: (rec: Recommendation) => void;
	private onInsertMulti: (recs: Recommendation[]) => void;
	private isTracked: (paperId: string) => boolean;
	private loading = false;
	private errorMessage: string | null = null;
	selectedRecs: Map<string, Recommendation> = new Map();

	// Bibliography callbacks
	private trackedCitations: TrackedCitation[] = [];
	private onRemoveCitation: ((paperId: string) => void) | null = null;
	private onExportBibliography: ((format: string) => void) | null = null;
	private bibExpanded = false;

	constructor(
		leaf: WorkspaceLeaf,
		settings: InCiteSettings,
		onInsert: (rec: Recommendation) => void,
		onInsertMulti: (recs: Recommendation[]) => void,
		isTracked: (paperId: string) => boolean
	) {
		super(leaf);
		this.settings = settings;
		this.onInsert = onInsert;
		this.onInsertMulti = onInsertMulti;
		this.isTracked = isTracked;
	}

	getViewType(): string {
		return VIEW_TYPE_INCITE;
	}

	getDisplayText(): string {
		return "inCite";
	}

	getIcon(): string {
		return "book-open";
	}

	async onOpen(): Promise<void> {
		this.render();
	}

	async onClose(): Promise<void> {
		// nothing to clean up
	}

	/** Update settings reference (called when settings change). */
	updateSettings(settings: InCiteSettings): void {
		this.settings = settings;
	}

	/** Set bibliography data and callbacks. */
	setBibliography(
		citations: TrackedCitation[],
		onRemove: (paperId: string) => void,
		onExport: (format: string) => void
	): void {
		this.trackedCitations = citations;
		this.onRemoveCitation = onRemove;
		this.onExportBibliography = onExport;
		this.render();
	}

	/** Update tracked citations list (re-renders). */
	updateTrackedCitations(citations: TrackedCitation[]): void {
		this.trackedCitations = citations;
		this.render();
	}

	/** Show loading spinner. */
	setLoading(): void {
		this.loading = true;
		this.errorMessage = null;
		this.render();
	}

	/** Display error message. */
	setError(message: string): void {
		this.loading = false;
		this.errorMessage = message;
		this.results = [];
		this.render();
	}

	/** Update displayed results. */
	setResults(results: Recommendation[]): void {
		this.loading = false;
		this.errorMessage = null;
		this.results = results;
		this.selectedRecs.clear();
		this.render();
	}

	private render(): void {
		const container = this.containerEl.children[1];
		container.empty();

		container.createEl("div", { cls: "mayacite-header" }, (header) => {
			header.createEl("h4", { text: "inCite" });
		});

		if (this.loading) {
			container.createEl("div", { cls: "mayacite-loading" }, (el) => {
				el.createEl("span", { text: "Searching..." });
			});
			return;
		}

		if (this.errorMessage) {
			container.createEl("div", { cls: "mayacite-error" }, (el) => {
				el.createEl("span", { text: this.errorMessage! });
			});
			return;
		}

		if (this.results.length === 0) {
			container.createEl("div", { cls: "mayacite-empty" }, (el) => {
				el.createEl("span", {
					text: "No results yet. Place your cursor and press Cmd/Ctrl+Shift+C.",
				});
			});
			this.renderBibliography(container);
			return;
		}

		// Selection action row (visible when selections > 0)
		this.renderSelectionBar(container);

		const list = container.createEl("div", { cls: "mayacite-results" });

		for (const rec of this.results) {
			const item = list.createEl("div", { cls: "mayacite-result" });

			// Header row: checkbox + rank + confidence badge
			item.createEl("div", { cls: "mayacite-result-rank" }, (el) => {
				// Selection checkbox
				const checkbox = el.createEl("input", {
					type: "checkbox",
					cls: "mayacite-select-checkbox",
				});
				(checkbox as HTMLInputElement).checked = this.selectedRecs.has(rec.paper_id);
				checkbox.addEventListener("change", () => {
					if ((checkbox as HTMLInputElement).checked) {
						this.selectedRecs.set(rec.paper_id, rec);
					} else {
						this.selectedRecs.delete(rec.paper_id);
					}
					this.renderSelectionBarUpdate();
				});

				el.createEl("span", {
					cls: "mayacite-rank-number",
					text: `${rec.rank}.`,
				});
				const conf = rec.confidence ?? 0;
				const confClass =
					conf >= 0.55
						? "mayacite-confidence-high"
						: conf >= 0.35
							? "mayacite-confidence-mid"
							: "mayacite-confidence-low";
				el.createEl("span", {
					cls: `mayacite-confidence ${confClass}`,
					text: conf.toFixed(2),
					attr: { title: `Confidence: ${conf.toFixed(3)}` },
				});
			});

			// Title
			item.createEl("div", { cls: "mayacite-result-title", text: rec.title });

			// Authors + year
			if (rec.authors || rec.year) {
				const meta: string[] = [];
				if (rec.authors && rec.authors.length > 0) {
					const names = rec.authors.slice(0, 3).join(", ");
					meta.push(
						rec.authors.length > 3 ? `${names} et al.` : names
					);
				}
				if (rec.year) {
					meta.push(`(${rec.year})`);
				}
				item.createEl("div", {
					cls: "mayacite-result-meta",
					text: meta.join(" "),
				});
			}

			// Matched paragraph evidence (with **bold** highlighting)
			if (this.settings.showParagraphs) {
				if (rec.matched_paragraphs?.length) {
					rec.matched_paragraphs.forEach((snippet, idx) => {
						const blockquote = item.createEl("blockquote", {
							cls: idx === 0 ? "mayacite-paragraph" : "mayacite-paragraph mayacite-paragraph-secondary",
						});
						if (snippet.score != null) {
							blockquote.createEl("span", {
								cls: "mayacite-evidence-score",
								text: `${(snippet.score * 100).toFixed(0)}%`,
							});
						}
						this.renderHighlightedText(snippet.text, blockquote, 350);
					});
				} else if (rec.matched_paragraph) {
					const blockquote = item.createEl("blockquote", {
						cls: "mayacite-paragraph",
					});
					this.renderHighlightedText(rec.matched_paragraph, blockquote, 350);
				}
			}

			// Insert button + cited badge
			const btnRow = item.createEl("div", { cls: "mayacite-result-actions" });
			const insertBtn = btnRow.createEl("button", {
				cls: "mayacite-insert-btn",
				text: "Insert",
			});
			setIcon(insertBtn.createSpan({ cls: "mayacite-insert-icon" }), "plus");
			insertBtn.addEventListener("click", () => {
				this.onInsert(rec);
			});

			if (this.isTracked(rec.paper_id)) {
				btnRow.createEl("span", {
					cls: "mayacite-cited-badge",
					text: "Cited",
					attr: { title: "Already cited in this document" },
				});
			}
		}

		this.renderBibliography(container);
	}

	/** Render the selection action bar. */
	private renderSelectionBar(container: Element): void {
		const bar = container.createEl("div", {
			cls: "mayacite-selection-bar",
			attr: { style: this.selectedRecs.size > 0 ? "" : "display:none" },
		});
		bar.dataset.selectionBar = "true";

		bar.createEl("span", {
			cls: "mayacite-selection-count",
			text: `${this.selectedRecs.size} selected`,
		});

		const insertBtn = bar.createEl("button", {
			cls: "mayacite-insert-btn",
			text: "Insert Selected",
		});
		insertBtn.addEventListener("click", () => {
			const recs = Array.from(this.selectedRecs.values());
			if (recs.length > 0) {
				this.onInsertMulti(recs);
				this.selectedRecs.clear();
				this.render();
			}
		});

		const clearBtn = bar.createEl("button", {
			cls: "mayacite-clear-btn",
			text: "Clear",
		});
		clearBtn.addEventListener("click", () => {
			this.selectedRecs.clear();
			this.render();
		});
	}

	/** Update just the selection bar without full re-render. */
	private renderSelectionBarUpdate(): void {
		const container = this.containerEl.children[1];
		const bar = container.querySelector("[data-selection-bar]") as HTMLElement | null;
		if (!bar) return;

		if (this.selectedRecs.size > 0) {
			bar.style.display = "";
			const countEl = bar.querySelector(".mayacite-selection-count");
			if (countEl) countEl.textContent = `${this.selectedRecs.size} selected`;
		} else {
			bar.style.display = "none";
		}
	}

	/** Render the collapsible bibliography section. */
	private renderBibliography(container: Element): void {
		if (this.trackedCitations.length === 0 && !this.onExportBibliography) return;

		const section = container.createEl("div", { cls: "mayacite-bibliography" });

		const header = section.createEl("div", {
			cls: "mayacite-bib-header",
		});
		const toggleIcon = header.createEl("span", { cls: "mayacite-bib-toggle" });
		setIcon(toggleIcon, this.bibExpanded ? "chevron-down" : "chevron-right");
		header.createEl("span", {
			text: `Bibliography (${this.trackedCitations.length} citations)`,
		});

		header.addEventListener("click", () => {
			this.bibExpanded = !this.bibExpanded;
			this.render();
		});

		if (!this.bibExpanded) return;

		// Export buttons
		if (this.onExportBibliography && this.trackedCitations.length > 0) {
			const exportRow = section.createEl("div", { cls: "mayacite-bib-export" });
			for (const fmt of ["BibTeX", "RIS", "Text"]) {
				const btn = exportRow.createEl("button", {
					cls: "mayacite-export-btn",
					text: fmt,
				});
				btn.addEventListener("click", () => {
					this.onExportBibliography!(fmt.toLowerCase());
				});
			}
		}

		// Citation list
		if (this.trackedCitations.length === 0) {
			section.createEl("div", {
				cls: "mayacite-bib-empty",
				text: "No citations tracked yet. Insert a citation to begin.",
			});
			return;
		}

		const list = section.createEl("div", { cls: "mayacite-bib-list" });
		for (const cite of this.trackedCitations) {
			const row = list.createEl("div", { cls: "mayacite-bib-item" });
			const authorStr = cite.authors.length > 0
				? cite.authors[0].split(/\s+/).pop() ?? ""
				: "";
			const label = cite.year
				? `${authorStr} (${cite.year})`
				: authorStr;
			row.createEl("span", {
				cls: "mayacite-bib-label",
				text: label,
				attr: { title: cite.title },
			});
			row.createEl("span", {
				cls: "mayacite-bib-title",
				text: cite.title,
			});
			if (this.onRemoveCitation) {
				const removeBtn = row.createEl("button", {
					cls: "mayacite-bib-remove",
					attr: { title: "Remove from bibliography" },
				});
				setIcon(removeBtn, "x");
				removeBtn.addEventListener("click", () => {
					this.onRemoveCitation!(cite.paper_id);
				});
			}
		}
	}

	/**
	 * Render text with **bold** markers converted to <strong> elements.
	 * Centers the view around the highlighted sentence, truncating before/after as needed.
	 */
	private renderHighlightedText(
		text: string,
		container: HTMLElement,
		maxLength: number
	): void {
		// Find the highlighted portion
		const startBold = text.indexOf("**");
		const endBold = text.indexOf("**", startBold + 2);

		// No valid highlight found - just truncate normally
		if (startBold === -1 || endBold === -1) {
			const truncated = text.length > maxLength;
			const displayText = truncated ? text.slice(0, maxLength) + "..." : text;
			container.appendText(displayText);
			return;
		}

		// Calculate window around the highlight
		const highlightEnd = endBold + 2;
		const highlightLength = highlightEnd - startBold;
		const availableContext = maxLength - highlightLength;
		const contextBefore = Math.floor(availableContext * 0.3); // 30% before
		const contextAfter = availableContext - contextBefore; // 70% after

		// Determine slice boundaries
		let sliceStart = Math.max(0, startBold - contextBefore);
		let sliceEnd = Math.min(text.length, highlightEnd + contextAfter);

		// Extract the display portion
		let displayText = text.slice(sliceStart, sliceEnd);

		// Add ellipsis indicators
		const prefixEllipsis = sliceStart > 0;
		const suffixEllipsis = sliceEnd < text.length;

		if (prefixEllipsis) {
			// Try to start at a word boundary
			const spaceIdx = displayText.indexOf(" ");
			if (spaceIdx > 0 && spaceIdx < 20) {
				displayText = displayText.slice(spaceIdx + 1);
			}
			displayText = "..." + displayText;
		}
		if (suffixEllipsis) {
			displayText = displayText + "...";
		}

		// Split on **text** patterns, keeping the delimiters
		const parts = displayText.split(/(\*\*.+?\*\*)/g);
		for (const part of parts) {
			if (part.startsWith("**") && part.endsWith("**")) {
				// Bold text - strip markers and render as <strong>
				const boldText = part.slice(2, -2);
				container.createEl("strong", { text: boldText });
			} else if (part) {
				// Regular text
				container.appendText(part);
			}
		}
	}
}
