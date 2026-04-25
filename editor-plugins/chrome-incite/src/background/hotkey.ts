/**
 * Hotkey listener injection for the save-paper keyboard shortcut.
 *
 * The Chrome `commands` API only supports a limited set of modifier+key combos.
 * To allow user-configurable hotkeys, we inject a keydown listener into every
 * page that reads the hotkey from chrome.storage.sync and fires a message
 * back to the service worker on match.
 */

/**
 * Content function injected into pages to listen for the save-paper hotkey.
 * Reads the hotkey from chrome.storage.sync and sends a message on match.
 * Guarded against double-injection via a window flag.
 */
export function injectedHotkeyListener() {
  const FLAG = "__incite_hotkey_injected__";
  if ((window as unknown as Record<string, boolean>)[FLAG]) return;
  (window as unknown as Record<string, boolean>)[FLAG] = true;

  const STORAGE_KEY = "incite_settings";
  let hotkeyStr = "Alt+Shift+S";

  function parseHk(s: string) {
    const parts = s.split("+").map((p) => p.trim());
    const r = { ctrl: false, alt: false, shift: false, meta: false, key: "" };
    for (const part of parts) {
      const l = part.toLowerCase();
      if (l === "ctrl" || l === "control") r.ctrl = true;
      else if (l === "alt") r.alt = true;
      else if (l === "shift") r.shift = true;
      else if (l === "meta" || l === "cmd" || l === "command") r.meta = true;
      else r.key = l;
    }
    return r;
  }

  let parsed = parseHk(hotkeyStr);

  // Load initial hotkey from storage
  chrome.storage.sync.get(STORAGE_KEY, (result) => {
    const stored = result[STORAGE_KEY];
    if (stored?.savePaperHotkey) {
      hotkeyStr = stored.savePaperHotkey;
      parsed = parseHk(hotkeyStr);
    }
  });

  // Listen for hotkey changes without re-injection
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === "sync" && changes[STORAGE_KEY]?.newValue?.savePaperHotkey) {
      hotkeyStr = changes[STORAGE_KEY].newValue.savePaperHotkey;
      parsed = parseHk(hotkeyStr);
    }
  });

  document.addEventListener("keydown", (event: KeyboardEvent) => {
    if (
      event.ctrlKey === parsed.ctrl &&
      event.altKey === parsed.alt &&
      event.shiftKey === parsed.shift &&
      event.metaKey === parsed.meta &&
      event.key.toLowerCase() === parsed.key
    ) {
      event.preventDefault();
      event.stopPropagation();
      chrome.runtime.sendMessage({ type: "SAVE_PAPERS_HOTKEY" });
    }
  }, true);
}

/** Inject the hotkey listener into a tab. */
export async function injectHotkeyListener(tabId: number): Promise<void> {
  try {
    await chrome.scripting.executeScript({
      target: { tabId },
      func: injectedHotkeyListener,
    });
  } catch {
    // Injection not allowed on this page (chrome://, edge://, etc.)
  }
}
