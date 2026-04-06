import type { PaperMetadata, LibraryCheckResult, Collection, Tag } from "@incite/shared";
import type { ChromeExtensionSettings } from "../shared/types";

// Re-export shared types used by popup modules
export type { PaperMetadata, LibraryCheckResult, Collection, Tag };

// --- Popup state machine ---

export type PopupState =
  | { kind: "loading" }
  | { kind: "no-paper" }
  | { kind: "not-signed-in" }
  | { kind: "single-paper"; paper: PaperMetadata; check: LibraryCheckResult | null }
  | { kind: "multi-paper"; papers: PaperMetadata[]; checks: LibraryCheckResult[] }
  | { kind: "already-saved"; paper: PaperMetadata; check: LibraryCheckResult }
  | { kind: "likely-saved"; paper: PaperMetadata; check: LibraryCheckResult }
  | { kind: "saving" }
  | { kind: "success"; savedCount: number; collectionName: string }
  | { kind: "error"; message: string };

// --- Mutable state ---

export let state: PopupState = { kind: "loading" };
export let collections: Collection[] = [];
export let selectedCollectionId: string | null = null;
export let selectedTags: string[] = [];
export let tagSuggestions: Tag[] = [];
export let showTagInput = false;
export let selectedPaperIndices = new Set<number>();
export let settings: ChromeExtensionSettings;

// --- State setters ---

export function setState(s: PopupState): void {
  state = s;
}

export function setCollections(c: Collection[]): void {
  collections = c;
}

export function setSelectedCollectionId(id: string | null): void {
  selectedCollectionId = id;
}

export function setSelectedTags(tags: string[]): void {
  selectedTags = tags;
}

export function setTagSuggestions(tags: Tag[]): void {
  tagSuggestions = tags;
}

export function setShowTagInput(show: boolean): void {
  showTagInput = show;
}

export function setSelectedPaperIndices(indices: Set<number>): void {
  selectedPaperIndices = indices;
}

export function setSettings(s: ChromeExtensionSettings): void {
  settings = s;
}

// --- Callbacks (set by entry point, used by event binders) ---

let _render: () => void = () => {};
let _savePapers: (papers: PaperMetadata[]) => Promise<void> = async () => {};

export function setRenderCallback(fn: () => void): void {
  _render = fn;
}

export function setSavePapersCallback(fn: (papers: PaperMetadata[]) => Promise<void>): void {
  _savePapers = fn;
}

export function render(): void {
  _render();
}

export function savePapers(papers: PaperMetadata[]): Promise<void> {
  return _savePapers(papers);
}
