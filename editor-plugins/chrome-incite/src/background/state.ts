/**
 * Shared mutable state for the service worker.
 *
 * Centralized here so handler modules can import state directly
 * without threading a context object through every function call.
 * All state is module-scoped — esbuild bundles everything into a
 * single IIFE, so these are effectively private globals.
 */
import type { PaperMetadata } from "../translators/types";
import { GoogleDocsAPI } from "../shared/gdocs-api";

/** Detection result cached per tab, cleared on navigation. */
export interface DetectedPapersState {
  type: "single" | "multiple";
  papers: PaperMetadata[];
  translatorName: string;
}

/** Cache of detected papers per tab ID. */
export const detectedPapers = new Map<number, DetectedPapersState>();

/** Singleton Google Docs REST API client (handles OAuth internally). */
export const gdocsApi = new GoogleDocsAPI();

/** Cache of last resolved cursor index per document, for citation insertion. */
export const gdocsCursorCache = new Map<string, { index: number; timestamp: number }>();

/** Whether the side panel has signaled it is ready to receive messages. */
export let panelReady = false;

export function setPanelReady(ready: boolean): void {
  panelReady = ready;
}
