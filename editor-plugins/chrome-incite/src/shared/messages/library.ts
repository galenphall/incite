/**
 * Library message types.
 * Messages for paper detection, saving, library checks, collections, and tags.
 */

import type { PaperMetadata } from "../../translators/types";

export interface PagePapersDetectedMessage {
  type: "PAGE_PAPERS_DETECTED";
  detection: { type: "single" | "multiple" };
  papers?: PaperMetadata[];
  translatorName: string;
}

export interface GetDetectedPapersMessage {
  type: "GET_DETECTED_PAPERS";
}

export interface SavePapersMessage {
  type: "SAVE_PAPERS";
  papers: PaperMetadata[];
  collectionId?: string | null;
  tags?: string[];
  enrich?: boolean;
}

export interface CheckLibraryMessage {
  type: "CHECK_LIBRARY";
  papers: PaperMetadata[];
}

export interface GetCollectionsMessage {
  type: "GET_COLLECTIONS";
}

export interface SearchTagsMessage {
  type: "SEARCH_TAGS";
  query: string;
}

export interface ExtractPapersMessage {
  type: "EXTRACT_PAPERS";
}

export interface UpdateLibraryItemMessage {
  type: "UPDATE_LIBRARY_ITEM";
  canonicalId: string;
  collectionId?: string | null;
  tags?: string[];
}

export type LibraryMessage =
  | PagePapersDetectedMessage
  | GetDetectedPapersMessage
  | SavePapersMessage
  | CheckLibraryMessage
  | GetCollectionsMessage
  | SearchTagsMessage
  | ExtractPapersMessage
  | UpdateLibraryItemMessage;
