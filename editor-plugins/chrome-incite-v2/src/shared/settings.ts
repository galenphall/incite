import type { ChromeExtensionSettings } from "./types";
import { DEFAULT_SETTINGS, STORAGE_KEY } from "./constants";

export async function loadSettings(): Promise<ChromeExtensionSettings> {
  const result = await chrome.storage.sync.get(STORAGE_KEY);
  const stored = result[STORAGE_KEY] ?? {};
  return { ...DEFAULT_SETTINGS, ...stored };
}

export async function saveSettings(
  partial: Partial<ChromeExtensionSettings>
): Promise<ChromeExtensionSettings> {
  const current = await loadSettings();
  const updated = { ...current, ...partial };
  await chrome.storage.sync.set({ [STORAGE_KEY]: updated });
  return updated;
}
