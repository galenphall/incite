import { getActiveUrl } from "@incite/shared";
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

/** Cloud-syncable keys that should be pulled from the server. */
const CLOUD_SYNCABLE_KEYS: (keyof ChromeExtensionSettings)[] = [
  "k", "authorBoost", "contextSentences", "insertFormat",
  "autoDetectEnabled", "showParagraphs", "collectionId", "savePaperHotkey",
];

/**
 * Fetch settings from the cloud API and merge cloud-syncable keys into
 * chrome.storage.sync. Non-syncable keys (apiMode, apiToken, etc.) are
 * never overwritten.
 */
export async function syncFromCloud(): Promise<void> {
  const current = await loadSettings();
  if (!current.apiToken) return;

  const baseUrl = getActiveUrl(current);
  try {
    const response = await fetch(`${baseUrl}/api/v1/settings`, {
      headers: {
        Authorization: `Bearer ${current.apiToken}`,
        Accept: "application/json",
      },
    });
    if (!response.ok) return;

    const { settings: cloud } = await response.json();
    if (!cloud || typeof cloud !== "object") return;

    const patch: Partial<ChromeExtensionSettings> = {};
    for (const key of CLOUD_SYNCABLE_KEYS) {
      if (key in cloud && cloud[key] !== undefined) {
        (patch as Record<string, unknown>)[key] = cloud[key];
      }
    }

    if (Object.keys(patch).length > 0) {
      await saveSettings(patch);
    }
  } catch {
    // Silently ignore network errors — cloud sync is best-effort
  }
}
