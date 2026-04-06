/**
 * Type-safe wrapper around chrome.runtime.sendMessage.
 * Maps each message type string to its expected response type,
 * eliminating the need for unsafe `as` casts at call sites.
 */

import type { RecommendResponse, HealthResponse } from "@incite/shared";
import type { ChromeExtensionSettings } from "../types";

/** Map of message type strings to their response shapes. */
interface MessageResponseMap {
  GET_RECOMMENDATIONS: { response?: RecommendResponse; error?: string; query?: string };
  GET_RECOMMENDATIONS_FOR_TEXT: { response?: RecommendResponse; error?: string };
  CHECK_HEALTH: { response?: HealthResponse; error?: string };
  GET_SETTINGS: { settings: ChromeExtensionSettings };
  SAVE_SETTINGS: { settings: ChromeExtensionSettings };
  INSERT_CITATION_REQUEST: { success: boolean; method?: string };
  INSERT_MULTI_CITATION_REQUEST: { success: boolean; method?: string };
  GDOCS_INSERT_CITATION: { success: boolean; error?: string };
  GDOCS_INSERT_MULTI_CITATION: { success: boolean; error?: string };
  GDOCS_INSERT_BIBLIOGRAPHY: { success: boolean; error?: string };
  GDOCS_SCAN_CITATIONS: { success: boolean; data?: unknown; error?: string };
  GDOCS_CLEAN: { success: boolean; data?: unknown; error?: string };
  GDOCS_REFRESH_CITATIONS: { success: boolean; data?: unknown; error?: string };
  GET_DETECTED_PAPERS: { papers: unknown[]; type: string | null };
  SAVE_PAPERS: { saved?: unknown[]; already_existed?: unknown[]; error?: string };
  CHECK_LIBRARY: { results: unknown[] };
  GET_COLLECTIONS: { collections: unknown[] };
  SEARCH_TAGS: { tags: unknown[] };
  UPDATE_LIBRARY_ITEM: { error?: string };
  PANEL_READY: { ack: boolean };
  SAVE_PAPERS_HOTKEY: { ack: boolean };
  TRIGGER_FROM_HOTKEY: { ack: boolean };
}

/**
 * Send a message to the service worker with type-safe response.
 *
 * @param message - The message object (must include a `type` field matching MessageResponseMap keys)
 * @returns The typed response from the service worker
 */
export async function sendMessage<K extends keyof MessageResponseMap>(
  message: { type: K } & Record<string, unknown>,
): Promise<MessageResponseMap[K]> {
  return chrome.runtime.sendMessage(message);
}
