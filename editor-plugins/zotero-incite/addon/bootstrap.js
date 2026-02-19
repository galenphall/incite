/* eslint-disable no-undef */
// Zotero 7 bootstrap lifecycle — loads the bundled plugin code.

var chromeHandle;

function install(data, reason) {}

async function startup({ id, version, resourceURI, rootURI }) {
	// Load the bundled IIFE script which sets globalThis.InciteZotero
	Services.scriptloader.loadSubScript(rootURI + "content/scripts/index.js");

	InciteZotero.hooks.onStartup();

	// Register each open main window
	for (const win of Zotero.getMainWindows()) {
		if (win.ZoteroPane) {
			InciteZotero.hooks.onMainWindowLoad(win);
		}
	}
}

function onMainWindowLoad({ window }) {
	InciteZotero.hooks.onMainWindowLoad(window);
}

function onMainWindowUnload({ window }) {
	InciteZotero.hooks.onMainWindowUnload(window);
}

function shutdown({ id, version, resourceURI, rootURI }, reason) {
	InciteZotero.hooks.onShutdown();

	// Unload from all open windows
	for (const win of Zotero.getMainWindows()) {
		InciteZotero.hooks.onMainWindowUnload(win);
	}
}

function uninstall(data, reason) {}
