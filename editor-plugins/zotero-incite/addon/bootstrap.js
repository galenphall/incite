/* eslint-disable no-undef */
// Zotero 7 bootstrap lifecycle — loads the bundled plugin code.

var chromeHandle;

function install(data, reason) {}

async function startup({ id, version, resourceURI, rootURI: _rootURI }) {
	// Expose rootURI globally so the bundled plugin code can access it
	globalThis.rootURI = _rootURI;

	// Register chrome so chrome://zotero-incite/content/ URLs resolve
	var aomStartup = Components.classes[
		"@mozilla.org/addons/addon-manager-startup;1"
	].getService(Components.interfaces.amIAddonManagerStartup);
	var manifestURI = Services.io.newURI(_rootURI + "manifest.json");
	chromeHandle = aomStartup.registerChrome(manifestURI, [
		["content", "zotero-incite", _rootURI + "content/"],
	]);

	// Load the bundled IIFE script which sets globalThis.InciteZotero
	Services.scriptloader.loadSubScript(_rootURI + "content/scripts/index.js");

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

	if (chromeHandle) {
		chromeHandle.destruct();
		chromeHandle = null;
	}
}

function uninstall(data, reason) {}
