/** Parsed representation of a hotkey combo like "Alt+Shift+S". */
export interface ParsedHotkey {
  ctrl: boolean;
  alt: boolean;
  shift: boolean;
  meta: boolean;
  key: string; // lowercase key name, e.g. "s", "k", "arrowdown"
}

/**
 * Parse a hotkey string like "Alt+Shift+S" or "Ctrl+K" into components.
 * Modifier order doesn't matter. The last non-modifier token is the key.
 */
export function parseHotkey(hotkeyString: string): ParsedHotkey {
  const parts = hotkeyString.split("+").map((p) => p.trim());
  const result: ParsedHotkey = { ctrl: false, alt: false, shift: false, meta: false, key: "" };

  for (const part of parts) {
    const lower = part.toLowerCase();
    if (lower === "ctrl" || lower === "control") result.ctrl = true;
    else if (lower === "alt") result.alt = true;
    else if (lower === "shift") result.shift = true;
    else if (lower === "meta" || lower === "cmd" || lower === "command") result.meta = true;
    else result.key = lower;
  }

  return result;
}

/** Check if a KeyboardEvent matches a parsed hotkey. */
export function matchesHotkey(event: KeyboardEvent, parsed: ParsedHotkey): boolean {
  if (event.ctrlKey !== parsed.ctrl) return false;
  if (event.altKey !== parsed.alt) return false;
  if (event.shiftKey !== parsed.shift) return false;
  if (event.metaKey !== parsed.meta) return false;
  return event.key.toLowerCase() === parsed.key;
}

/**
 * Format a KeyboardEvent into a display string like "Alt+Shift+S".
 * Used by the key recorder widget. Returns empty string for lone modifier presses.
 */
export function formatHotkey(event: KeyboardEvent): string {
  const modifierKeys = new Set(["control", "alt", "shift", "meta"]);
  if (modifierKeys.has(event.key.toLowerCase())) return "";

  const parts: string[] = [];
  if (event.ctrlKey) parts.push("Ctrl");
  if (event.altKey) parts.push("Alt");
  if (event.shiftKey) parts.push("Shift");
  if (event.metaKey) parts.push("Meta");

  // Require at least one modifier for a valid hotkey
  if (parts.length === 0) return "";

  // Capitalize single-char keys, keep named keys as-is
  const key = event.key.length === 1 ? event.key.toUpperCase() : event.key;
  parts.push(key);

  return parts.join("+");
}
