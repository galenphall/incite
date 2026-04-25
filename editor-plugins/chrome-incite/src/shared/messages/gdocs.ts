/**
 * Google Docs message types.
 * Messages for cursor context extraction and REST API-based document operations.
 */

export interface GetCursorContextMessage {
  type: "GET_CURSOR_CONTEXT";
  requestId: string;
}

export interface CursorContextResponseMessage {
  type: "CURSOR_CONTEXT_RESPONSE";
  requestId: string;
  paragraphText?: string;
  cursorOffset?: number;
  textBefore?: string;
  textAfter?: string;
  error?: string;
}

export interface GDocsInsertCitationMessage {
  type: "GDOCS_INSERT_CITATION";
  paperId: string;
  text: string;
  paperUrl: string;
}

export interface GDocsInsertMultiCitationMessage {
  type: "GDOCS_INSERT_MULTI_CITATION";
  fullText: string;
  segments: { text: string; paperUrl: string; paperId: string; offsetInFullText: number }[];
}

export interface GDocsInsertBibliographyMessage {
  type: "GDOCS_INSERT_BIBLIOGRAPHY";
  entries: { paperId: string; formatted: string; url?: string }[];
}

export interface GDocsScanCitationsMessage {
  type: "GDOCS_SCAN_CITATIONS";
  trackedPaperIds: string[];
}

export interface GDocsCleanMessage {
  type: "GDOCS_CLEAN";
}

export interface GDocsRefreshCitationsMessage {
  type: "GDOCS_REFRESH_CITATIONS";
  trackedPaperIds: string[];
  refreshText: boolean;
}

export interface GDocsResultMessage {
  type: "GDOCS_RESULT";
  success: boolean;
  data?: unknown;
  error?: string;
}
