import type { ChromeExtensionSettings } from "./types";
import { DEFAULT_SETTINGS, STORAGE_KEY } from "./constants";

export async function loadSettings(): Promise<ChromeExtensionSettings> {
  const result = await chrome.storage.sync.get(STORAGE_KEY);
  const stored = result[STORAGE_KEY] ?? {};
  const merged = { ...DEFAULT_SETTINGS, ...stored };

  // Migrate: if citationStyle was never saved, the stored googleDocsCitationFormat
  // is from the old default (bibtex). Reset it to match the new APA default.
  if (!stored.citationStyle && stored.googleDocsCitationFormat) {
    merged.citationStyle = DEFAULT_SETTINGS.citationStyle;
    merged.googleDocsCitationFormat = DEFAULT_SETTINGS.googleDocsCitationFormat;
  }

  return merged;
}

export async function saveSettings(
  partial: Partial<ChromeExtensionSettings>
): Promise<ChromeExtensionSettings> {
  const current = await loadSettings();
  const updated = { ...current, ...partial };
  await chrome.storage.sync.set({ [STORAGE_KEY]: updated });
  return updated;
}
