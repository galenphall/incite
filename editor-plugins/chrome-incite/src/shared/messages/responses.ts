/**
 * Service worker response message types.
 * These are sent back from the service worker in response to panel/popup messages.
 */

import type { RecommendResponse, HealthResponse } from "@incite/shared";
import type { ChromeExtensionSettings } from "../types";

export interface RecommendationsResultMessage {
  type: "RECOMMENDATIONS_RESULT";
  response?: RecommendResponse;
  error?: string;
}

export interface HealthResultMessage {
  type: "HEALTH_RESULT";
  response?: HealthResponse;
  error?: string;
}

export interface SettingsResultMessage {
  type: "SETTINGS_RESULT";
  settings: ChromeExtensionSettings;
}

export interface InsertResultMessage {
  type: "INSERT_RESULT";
  success: boolean;
  method?: string; // "clipboard" | "direct"
}

export type ServiceWorkerResponse =
  | RecommendationsResultMessage
  | HealthResultMessage
  | SettingsResultMessage
  | InsertResultMessage;
