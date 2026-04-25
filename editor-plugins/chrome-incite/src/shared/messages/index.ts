/**
 * Message type index — re-exports all domain-specific message types.
 */

// Panel messages
export type {
  GetRecommendationsMessage,
  GetRecommendationsForTextMessage,
  CheckHealthMessage,
  GetSettingsMessage,
  SaveSettingsMessage,
  InsertCitationRequestMessage,
  InsertMultiCitationRequestMessage,
} from "./panel";

// Google Docs messages
export type {
  GetCursorContextMessage,
  CursorContextResponseMessage,
  GDocsInsertCitationMessage,
  GDocsInsertMultiCitationMessage,
  GDocsInsertBibliographyMessage,
  GDocsScanCitationsMessage,
  GDocsCleanMessage,
  GDocsRefreshCitationsMessage,
  GDocsResultMessage,
} from "./gdocs";

// Library messages
export type {
  PagePapersDetectedMessage,
  GetDetectedPapersMessage,
  SavePapersMessage,
  CheckLibraryMessage,
  GetCollectionsMessage,
  SearchTagsMessage,
  ExtractPapersMessage,
  UpdateLibraryItemMessage,
  LibraryMessage,
} from "./library";

// Response messages
export type {
  RecommendationsResultMessage,
  HealthResultMessage,
  SettingsResultMessage,
  InsertResultMessage,
  ServiceWorkerResponse,
} from "./responses";

// Overlay messages (rail + command palette)
export type {
  ToggleCommandPaletteMessage,
  OpenSidePanelMessage,
  OverlayMessage,
} from "./overlay";

// Type-safe message sender
export { sendMessage } from "./send";

// --- Content script messages (shared between content scripts and service worker) ---

import type { EditorType } from "../types";

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

// --- Composite union types ---

import type {
  GetRecommendationsMessage,
  GetRecommendationsForTextMessage,
  CheckHealthMessage,
  GetSettingsMessage,
  SaveSettingsMessage,
  InsertCitationRequestMessage,
  InsertMultiCitationRequestMessage,
} from "./panel";

import type {
  GDocsInsertCitationMessage,
  GDocsInsertMultiCitationMessage,
  GDocsInsertBibliographyMessage,
  GDocsScanCitationsMessage,
  GDocsCleanMessage,
  GDocsRefreshCitationsMessage,
} from "./gdocs";

import type { LibraryMessage } from "./library";

import type { OverlayMessage } from "./overlay";

/** All messages the panel can send to the service worker. */
export type PanelMessage =
  | GetRecommendationsMessage
  | CheckHealthMessage
  | GetSettingsMessage
  | SaveSettingsMessage
  | InsertCitationRequestMessage
  | InsertMultiCitationRequestMessage
  | GDocsInsertCitationMessage
  | GDocsInsertMultiCitationMessage
  | GDocsInsertBibliographyMessage
  | GDocsScanCitationsMessage
  | GDocsCleanMessage
  | GDocsRefreshCitationsMessage;

/** Extended panel message — includes panel messages plus lifecycle, library, and overlay messages. */
export type ExtendedPanelMessage =
  | PanelMessage
  | { type: "PANEL_READY" }
  | { type: "SAVE_PAPERS_HOTKEY" }
  | GetRecommendationsForTextMessage
  | LibraryMessage
  | OverlayMessage;
