import type { InCiteSettings } from "@incite/shared";

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

// Bridge messages for Overleaf (window.postMessage between ISOLATED and MAIN world)
export interface OverleafBridgeMessage {
  source: "incite-isolated" | "incite-main";
  type: "GET_CONTEXT" | "CONTEXT_RESPONSE" | "INSERT_CITATION" | "INSERT_RESULT";
  requestId: string;
  payload?: unknown;
}
