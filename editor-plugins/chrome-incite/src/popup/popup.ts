import { loadSettings } from "../shared/settings";
import type { PaperMetadata, LibraryCheckResult } from "./popup-state";
import {
  state,
  collections,
  selectedCollectionId,
  selectedTags,
  selectedPaperIndices,
  setState,
  setCollections,
  setSelectedCollectionId,
  setSelectedTags,
  setShowTagInput,
  setSelectedPaperIndices,
  setSettings,
  setRenderCallback,
  setSavePapersCallback,
} from "./popup-state";
import {
  renderLoading,
  renderNoPaper,
  renderNotSignedIn,
  renderSinglePaper,
  renderMultiPaper,
  renderAlreadySaved,
  renderLikelySaved,
  renderSaving,
  renderSuccess,
  renderError,
} from "./popup-renderers";
import {
  bindSinglePaperEvents,
  bindMultiPaperEvents,
  bindAlreadySavedEvents,
  bindLikelySavedEvents,
  bindSuccessEvents,
  bindErrorEvents,
} from "./popup-events";

const root = document.getElementById("popup-root")!;

// Wire callbacks so sub-modules can call render() and savePapers()
setRenderCallback(render);
setSavePapersCallback(savePapers);

// --- Initialization ---

(async function init() {
  const settings = await loadSettings();
  setSettings(settings);
  render();

  if (!settings.apiToken) {
    setState({ kind: "not-signed-in" });
    render();
    return;
  }

  // Get detected papers from service worker
  try {
    const response = await chrome.runtime.sendMessage({ type: "GET_DETECTED_PAPERS" });

    if (!response || !response.papers || response.papers.length === 0) {
      setState({ kind: "no-paper" });
      render();
      return;
    }

    // Load collections in parallel with library check
    const [collectionsResp, checkResp] = await Promise.all([
      chrome.runtime.sendMessage({ type: "GET_COLLECTIONS" }),
      chrome.runtime.sendMessage({ type: "CHECK_LIBRARY", papers: response.papers }),
    ]);

    if (collectionsResp?.collections) {
      setCollections(collectionsResp.collections);
    }

    // Restore last-used collection (only if it still exists)
    const stored = await chrome.storage.local.get("lastCollectionId");
    if (stored.lastCollectionId && collections.some((c) => c.id === stored.lastCollectionId)) {
      setSelectedCollectionId(stored.lastCollectionId);
    }

    const papers: PaperMetadata[] = response.papers;
    const checks: LibraryCheckResult[] = checkResp?.results ?? [];

    if (response.type === "single" && papers.length === 1) {
      const check = checks[0] ?? null;
      if (check?.in_library && check.match_type === "fuzzy_title") {
        setState({ kind: "likely-saved", paper: papers[0], check });
        // Pre-populate tags from the existing library item
        if (check.tags?.length) {
          setSelectedTags(check.tags.map((t) => t.name));
          setShowTagInput(true);
        }
      } else if (check?.in_library) {
        setState({ kind: "already-saved", paper: papers[0], check });
        // Pre-populate tags from the existing library item
        if (check.tags?.length) {
          setSelectedTags(check.tags.map((t) => t.name));
          setShowTagInput(true);
        }
      } else {
        setState({ kind: "single-paper", paper: papers[0], check });
      }
    } else {
      // Multi-paper: pre-select papers not exactly in library
      // Fuzzy matches are left unchecked so the user explicitly opts in
      const indices = new Set<number>();
      papers.forEach((_, i) => {
        const isExact = checks[i]?.in_library && checks[i]?.match_type !== "fuzzy_title";
        if (!isExact && !checks[i]?.in_library) {
          indices.add(i);
        }
      });
      setSelectedPaperIndices(indices);
      setState({ kind: "multi-paper", papers, checks });
    }
  } catch (err) {
    setState({ kind: "error", message: err instanceof Error ? err.message : "Failed to load" });
  }

  render();
})();

// --- State machine dispatch ---

function render() {
  switch (state.kind) {
    case "loading":
      root.innerHTML = renderLoading();
      break;
    case "no-paper":
      root.innerHTML = renderNoPaper();
      break;
    case "not-signed-in":
      root.innerHTML = renderNotSignedIn();
      document.getElementById("open-options")?.addEventListener("click", (e) => {
        e.preventDefault();
        chrome.runtime.openOptionsPage();
      });
      break;
    case "single-paper":
      root.innerHTML = renderSinglePaper(state.paper, state.check);
      bindSinglePaperEvents();
      break;
    case "multi-paper":
      root.innerHTML = renderMultiPaper(state.papers, state.checks);
      bindMultiPaperEvents();
      break;
    case "already-saved":
      root.innerHTML = renderAlreadySaved(state.paper, state.check);
      bindAlreadySavedEvents();
      break;
    case "likely-saved":
      root.innerHTML = renderLikelySaved(state.paper, state.check);
      bindLikelySavedEvents();
      break;
    case "saving":
      root.innerHTML = renderSaving();
      break;
    case "success":
      root.innerHTML = renderSuccess(state.savedCount, state.collectionName);
      bindSuccessEvents();
      break;
    case "error":
      root.innerHTML = renderError(state.message);
      bindErrorEvents();
      break;
  }
}

// --- Save Action ---

async function savePapers(papers: PaperMetadata[]) {
  setState({ kind: "saving" });
  render();

  try {
    const response = await chrome.runtime.sendMessage({
      type: "SAVE_PAPERS",
      papers,
      collectionId: selectedCollectionId,
      tags: selectedTags.length > 0 ? selectedTags : undefined,
      enrich: true,
    });

    if (response?.error) {
      setState({ kind: "error", message: response.error });
    } else {
      const savedCount = (response?.saved?.length ?? 0) + (response?.already_existed?.length ?? 0);
      const collectionName = collections.find((c) => c.id === selectedCollectionId)?.name ?? "My Library";
      setState({ kind: "success", savedCount, collectionName });
    }
  } catch (err) {
    setState({ kind: "error", message: err instanceof Error ? err.message : "Save failed" });
  }

  render();

  if (state.kind === "success") {
    setTimeout(() => window.close(), 1500);
  }
}
