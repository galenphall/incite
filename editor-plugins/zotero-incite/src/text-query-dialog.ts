const MENU_ID = "incite-tools-menu";

/** Open (or focus) the control center in a Zotero-native window. */
function openControlCenter(): void {
	try {
		const existing = Services.wm.getMostRecentWindow("incite:control-center");
		if (existing) {
			existing.focus();
			return;
		}
		const win = Zotero.getMainWindows()[0];
		if (!win) {
			Zotero.debug("inCite: no main window found");
			return;
		}
		const url = "chrome://zotero-incite/content/control-center.xhtml";
		Zotero.debug(`inCite: opening control center: ${url}`);
		win.openDialog(
			url,
			"incite-control-center",
			"chrome,centerscreen,resizable,dialog=no"
		);
	} catch (e) {
		Zotero.debug(`inCite: error opening control center: ${e}`);
	}
}

export function registerToolsMenu(win: Window): void {
	const doc = win.document;
	const toolsMenu = doc.getElementById("menu_ToolsPopup");
	if (!toolsMenu) return;

	const menuItem = doc.createXULElement("menuitem");
	menuItem.id = MENU_ID;
	menuItem.setAttribute("label", "inCite Control Center");
	menuItem.setAttribute("accesskey", "I");
	menuItem.addEventListener("command", openControlCenter);
	toolsMenu.appendChild(menuItem);

	const keyset = doc.getElementById("mainKeyset") ?? doc.createXULElement("keyset");
	const key = doc.createXULElement("key");
	key.id = "incite-shortcut-key";
	key.setAttribute("key", "I");
	key.setAttribute("modifiers", "accel,shift");
	key.addEventListener("command", openControlCenter);
	keyset.appendChild(key);
	if (!keyset.parentNode) {
		doc.documentElement.appendChild(keyset);
	}
}

export function unregisterToolsMenu(win: Window): void {
	const doc = win.document;
	doc.getElementById(MENU_ID)?.remove();
	doc.getElementById("incite-shortcut-key")?.remove();
}
