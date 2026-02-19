import { registerPreferencesPane } from "./prefs";
import { registerItemPaneSection, unregisterItemPaneSection } from "./item-pane-section";
import { registerToolsMenu, unregisterToolsMenu } from "./text-query-dialog";

const STYLE_ID = "incite-styles";

/** Inject the plugin stylesheet into a window. */
function injectStyles(win: Window): void {
	const doc = win.document;
	if (doc.getElementById(STYLE_ID)) return;

	const link = doc.createElement("link");
	link.id = STYLE_ID;
	link.rel = "stylesheet";
	link.href = rootURI + "content/style.css";
	doc.documentElement.appendChild(link);
}

/** Remove the plugin stylesheet from a window. */
function removeStyles(win: Window): void {
	win.document.getElementById(STYLE_ID)?.remove();
}

export const hooks = {
	onStartup(): void {
		registerPreferencesPane();
		registerItemPaneSection();
	},

	onMainWindowLoad(win: Window): void {
		injectStyles(win);
		registerToolsMenu(win);
	},

	onMainWindowUnload(win: Window): void {
		removeStyles(win);
		unregisterToolsMenu(win);
	},

	onShutdown(): void {
		unregisterItemPaneSection();
	},
};
