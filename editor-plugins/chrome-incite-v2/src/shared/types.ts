import type { Recommendation, RecommendResponse, HealthResponse, InCiteSettings, ApiMode } from "@incite/shared";

// Re-export shared types used by other Chrome files
export type { ApiMode } from "@incite/shared";

/** Editor type detected from the current tab URL */
export type EditorType = "googledocs" | "overleaf" | "unknown";

/** Chrome extension settings — extends shared settings with Chrome-specific fields. */
export interface ChromeExtensionSettings extends InCiteSettings {
  citationStyle: string;
  googleDocsCitationFormat: string;
  overleafCitationFormat: string;
  showAbstracts: boolean;
}

// --- Message protocol between service worker, panel, and content scripts ---

export interface GetContextMessage {
  type: "GET_CONTEXT";
  requestId: string;
}

export interface ContextResponseMessage {
  type: "CONTEXT_RESPONSE";
  requestId: string;
  text?: string;
  cursorOffset?: number;
  fullText?: string;
  error?: string;
}

export interface InsertCitationMessage {
  type: "INSERT_CITATION";
  citation: string;
  editorType: EditorType;
}

export interface InsertResultMessage {
  type: "INSERT_RESULT";
  success: boolean;
  method?: string; // "clipboard" | "direct"
}

export interface GetRecommendationsMessage {
  type: "GET_RECOMMENDATIONS";
}

export interface CheckHealthMessage {
  type: "CHECK_HEALTH";
}

export interface GetSettingsMessage {
  type: "GET_SETTINGS";
}

export interface SaveSettingsMessage {
  type: "SAVE_SETTINGS";
  settings: Partial<ChromeExtensionSettings>;
}

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

export interface InsertMultiCitationRequestMessage {
  type: "INSERT_MULTI_CITATION_REQUEST";
  recommendations: Recommendation[];
}

export type PanelMessage =
  | GetRecommendationsMessage
  | CheckHealthMessage
  | GetSettingsMessage
  | SaveSettingsMessage
  | { type: "INSERT_CITATION_REQUEST"; recommendation: Recommendation }
  | InsertMultiCitationRequestMessage;

export type ServiceWorkerResponse =
  | RecommendationsResultMessage
  | HealthResultMessage
  | SettingsResultMessage
  | InsertResultMessage;

// Bridge messages for Overleaf (window.postMessage between ISOLATED and MAIN world)
export interface OverleafBridgeMessage {
  source: "incite-isolated" | "incite-main";
  type: "GET_CONTEXT" | "CONTEXT_RESPONSE" | "INSERT_CITATION" | "INSERT_RESULT";
  requestId: string;
  payload?: unknown;
}
