/**
 * Shared API client singleton and common tab utilities.
 * Used by all handler modules that need to call the inCite API.
 */
import { InCiteClient, FetchTransport } from "@incite/shared";
import type { ClientConfig } from "@incite/shared";
import type { ChromeExtensionSettings, EditorType } from "../shared/types";
import { loadSettings } from "../shared/settings";

let client: InCiteClient | null = null;

export function configFromSettings(settings: ChromeExtensionSettings): ClientConfig {
  return {
    apiMode: settings.apiMode,
    cloudUrl: settings.cloudUrl,
    localUrl: settings.localUrl,
    apiToken: settings.apiToken,
  };
}

/** Get or create the shared InCiteClient, updating config from current settings. */
export async function getClient(): Promise<InCiteClient> {
  const settings = await loadSettings();
  if (!client) {
    client = new InCiteClient(configFromSettings(settings), new FetchTransport());
  } else {
    client.updateConfig(configFromSettings(settings));
  }
  return client;
}

/** Get the currently active tab, or null if none. */
export async function getActiveTab(): Promise<chrome.tabs.Tab | null> {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab ?? null;
}

/** Detect the editor type from a tab URL. */
export function detectEditorType(url: string): EditorType {
  if (url.includes("docs.google.com/document")) return "googledocs";
  if (url.includes("overleaf.com/project")) return "overleaf";
  return "unknown";
}
