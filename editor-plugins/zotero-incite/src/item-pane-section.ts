import { renderResultCardHTML, escapeHtml } from "@incite/shared";
import type { Recommendation, RenderResultOptions } from "@incite/shared";
import { InCiteClient } from "./api-client";
import { loadClientConfig, loadDisplaySettings } from "./prefs";
import { PLUGIN_ID, SECTION_ID } from "./types";

/**
 * Build a query string from a Zotero item, matching the canonical
 * format_paper_embedding_text() format: "Title. Authors. Year. Abstract"
 */
function buildQueryFromItem(item: Zotero.Item): string | null {
	if (!item.isRegularItem()) return null;

	const title = item.getField("title");
	if (!title) return null;

	const parts: string[] = [title];

	// Authors: format as "Smith", "Smith and Jones", or "Smith et al."
	const creators = item.getCreators().filter(c => c.creatorType === "author");
	if (creators.length === 1) {
		parts.push(creators[0].lastName ?? "");
	} else if (creators.length === 2) {
		parts.push(`${creators[0].lastName ?? ""} and ${creators[1].lastName ?? ""}`);
	} else if (creators.length > 2) {
		parts.push(`${creators[0].lastName ?? ""} et al.`);
	}

	const year = item.getField("date")?.replace(/^(\d{4}).*/, "$1");
	if (year) parts.push(year);

	const journal = item.getField("publicationTitle") || item.getField("journalAbbreviation");
	if (journal) parts.push(journal);

	const abstract = item.getField("abstractNote");
	if (abstract) parts.push(abstract);

	return parts.filter(Boolean).join(". ");
}

/** Handle click events on result cards via event delegation. */
function handleCardClick(e: MouseEvent, body: HTMLElement): void {
	const target = e.target as HTMLElement;
	const btn = target.closest("[data-action]") as HTMLElement | null;
	if (!btn) return;

	const action = btn.dataset.action;
	if (action === "copy") {
		const key = btn.dataset.copy ?? "";
		// Use the Zotero clipboard approach
		const clipboardHelper = Components.classes["@mozilla.org/widget/clipboardhelper;1"]
			?.getService(Components.interfaces.nsIClipboardHelper);
		if (clipboardHelper) {
			clipboardHelper.copyString(key);
		}
		btn.textContent = "Copied!";
		setTimeout(() => { btn.textContent = "Copy Key"; }, 1500);
	}
}

/** Render recommendation results into the pane body. */
function renderResults(body: HTMLElement, recs: Recommendation[], showParagraphs: boolean): void {
	const opts: RenderResultOptions = { showParagraphs, showAbstracts: false };
	let html = "";
	for (const rec of recs) {
		html += renderResultCardHTML(rec, opts);
	}
	body.innerHTML = html;

	// Attach click handler via delegation
	body.addEventListener("click", (e) => handleCardClick(e as MouseEvent, body));
}

/** Register the item pane section in Zotero. */
export function registerItemPaneSection(): void {
	Zotero.ItemPaneManager.registerSection({
		paneID: SECTION_ID,
		pluginID: PLUGIN_ID,
		header: {
			l10nID: "incite-section-header",
			icon: "chrome://zotero/skin/16/universal/book.svg",
		},
		sidenav: {
			l10nID: "incite-sidenav-label",
			icon: "chrome://zotero/skin/16/universal/book.svg",
		},
		onItemChange: ({ item, setEnabled }) => {
			// Only show for regular items (not notes/attachments)
			setEnabled(item.isRegularItem());
		},
		onRender: ({ body }) => {
			body.innerHTML = `<div class="incite-loading">Loading recommendations…</div>`;
		},
		onAsyncRender: async ({ body, item }) => {
			const query = buildQueryFromItem(item);
			if (!query) {
				body.innerHTML = `<div class="incite-message">Select a paper to see recommendations.</div>`;
				return;
			}

			try {
				const config = loadClientConfig();
				const { k, authorBoost, showParagraphs } = loadDisplaySettings();
				const client = new InCiteClient(config);

				const response = await client.recommend(query, k, authorBoost);

				if (!response.recommendations || response.recommendations.length === 0) {
					body.innerHTML = `<div class="incite-message">No recommendations found.</div>`;
					return;
				}

				renderResults(body, response.recommendations, showParagraphs);
			} catch (err) {
				const config = loadClientConfig();
				const mode = config.apiMode;
				const msg = mode === "local"
					? `Could not connect to local inCite server. Make sure <code>incite serve</code> is running.`
					: `Could not connect to inCite cloud. Check your API token in settings.`;
				body.innerHTML = `<div class="incite-error">${msg}<br><small>${escapeHtml(String(err))}</small></div>`;
			}
		},
	});
}

/** Unregister the item pane section. */
export function unregisterItemPaneSection(): void {
	try {
		Zotero.ItemPaneManager.unregisterSection(SECTION_ID);
	} catch {
		// Ignore if already unregistered
	}
}
