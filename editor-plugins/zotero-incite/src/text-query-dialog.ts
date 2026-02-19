import { renderResultCardHTML, escapeHtml } from "@incite/shared";
import type { RenderResultOptions } from "@incite/shared";
import { InCiteClient } from "./api-client";
import { loadClientConfig, loadDisplaySettings } from "./prefs";

const DIALOG_ID = "incite-query-dialog";
const MENU_ID = "incite-tools-menu";

/** Create and show the text query dialog. */
function showDialog(win: Window): void {
	const doc = win.document;

	// Remove existing dialog if present
	doc.getElementById(DIALOG_ID)?.remove();

	const overlay = doc.createElement("div");
	overlay.id = DIALOG_ID;
	overlay.className = "incite-dialog-overlay";

	overlay.innerHTML = `
		<div class="incite-dialog">
			<div class="incite-dialog-header">
				<h3>inCite: Find Related Papers</h3>
				<button class="incite-dialog-close" data-action="close">&times;</button>
			</div>
			<div class="incite-dialog-body">
				<textarea class="incite-dialog-input" rows="5"
					placeholder="Paste a writing passage to find related papers from your library…"></textarea>
				<button class="incite-dialog-search" data-action="search">Search</button>
				<div class="incite-dialog-results"></div>
			</div>
		</div>
	`;

	doc.documentElement.appendChild(overlay);

	// Focus textarea
	const textarea = overlay.querySelector(".incite-dialog-input") as HTMLTextAreaElement;
	textarea?.focus();

	// Event handlers
	overlay.addEventListener("click", async (e) => {
		const target = e.target as HTMLElement;
		const action = target.closest("[data-action]") as HTMLElement | null;
		if (!action) return;

		const act = action.dataset.action;
		if (act === "close") {
			overlay.remove();
		} else if (act === "search") {
			await performSearch(overlay, textarea.value.trim());
		} else if (act === "copy") {
			const key = action.dataset.copy ?? "";
			const clipboardHelper = Components.classes["@mozilla.org/widget/clipboardhelper;1"]
				?.getService(Components.interfaces.nsIClipboardHelper);
			if (clipboardHelper) {
				clipboardHelper.copyString(key);
			}
			action.textContent = "Copied!";
			setTimeout(() => { action.textContent = "Copy Key"; }, 1500);
		}
	});

	// Close on Escape
	overlay.addEventListener("keydown", (e) => {
		if ((e as KeyboardEvent).key === "Escape") overlay.remove();
	});

	// Ctrl/Cmd+Enter to search
	textarea.addEventListener("keydown", async (e) => {
		if ((e as KeyboardEvent).key === "Enter" && (e.metaKey || e.ctrlKey)) {
			await performSearch(overlay, textarea.value.trim());
		}
	});
}

/** Run the recommendation query and render results. */
async function performSearch(container: HTMLElement, query: string): Promise<void> {
	const resultsEl = container.querySelector(".incite-dialog-results") as HTMLElement;
	if (!query) {
		resultsEl.innerHTML = `<div class="incite-message">Please enter some text to search.</div>`;
		return;
	}

	resultsEl.innerHTML = `<div class="incite-loading">Searching…</div>`;

	try {
		const config = loadClientConfig();
		const { k, authorBoost, showParagraphs } = loadDisplaySettings();
		const client = new InCiteClient(config);

		const response = await client.recommend(query, k, authorBoost);

		if (!response.recommendations || response.recommendations.length === 0) {
			resultsEl.innerHTML = `<div class="incite-message">No recommendations found.</div>`;
			return;
		}

		const opts: RenderResultOptions = { showParagraphs, showAbstracts: false };
		let html = "";
		for (const rec of response.recommendations) {
			html += renderResultCardHTML(rec, opts);
		}
		resultsEl.innerHTML = html;
	} catch (err) {
		const config = loadClientConfig();
		const mode = config.apiMode;
		const msg = mode === "local"
			? `Could not connect to local inCite server. Make sure <code>incite serve</code> is running.`
			: `Could not connect to inCite cloud. Check your API token in settings.`;
		resultsEl.innerHTML = `<div class="incite-error">${msg}<br><small>${escapeHtml(String(err))}</small></div>`;
	}
}

/** Add the "inCite: Find Related Papers..." menu item to the Tools menu. */
export function registerToolsMenu(win: Window): void {
	const doc = win.document;
	const toolsMenu = doc.getElementById("menu_ToolsPopup");
	if (!toolsMenu) return;

	const menuItem = doc.createXULElement("menuitem");
	menuItem.id = MENU_ID;
	menuItem.setAttribute("data-l10n-id", "incite-tools-menu-label");
	menuItem.setAttribute("accesskey", "I");
	menuItem.addEventListener("command", () => showDialog(win));
	toolsMenu.appendChild(menuItem);

	// Register keyboard shortcut: Cmd/Ctrl+Shift+I
	const keyset = doc.getElementById("mainKeyset") ?? doc.createXULElement("keyset");
	const key = doc.createXULElement("key");
	key.id = "incite-shortcut-key";
	key.setAttribute("key", "I");
	key.setAttribute("modifiers", "accel,shift");
	key.addEventListener("command", () => showDialog(win));
	keyset.appendChild(key);
	if (!keyset.parentNode) {
		doc.documentElement.appendChild(keyset);
	}
}

/** Remove the menu item and shortcut. */
export function unregisterToolsMenu(win: Window): void {
	const doc = win.document;
	doc.getElementById(MENU_ID)?.remove();
	doc.getElementById("incite-shortcut-key")?.remove();
	doc.getElementById(DIALOG_ID)?.remove();
}
