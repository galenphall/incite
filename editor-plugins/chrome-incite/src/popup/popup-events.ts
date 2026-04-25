import type { Tag } from "./popup-state";
import {
  state,
  collections,
  selectedCollectionId,
  selectedPaperIndices,
  selectedTags,
  tagSuggestions,
  showTagInput,
  setState,
  setSelectedCollectionId,
  setSelectedTags,
  setTagSuggestions,
  setShowTagInput,
  setSelectedPaperIndices,
  render,
  savePapers,
} from "./popup-state";

// --- Event binding functions ---

export function bindSinglePaperEvents(): void {
  document.getElementById("btn-cancel")?.addEventListener("click", () => window.close());

  document.getElementById("btn-save")?.addEventListener("click", async () => {
    if (state.kind !== "single-paper") return;
    await savePapers([state.paper]);
  });

  bindCollectionEvents();
  bindTagEvents();
}

export function bindMultiPaperEvents(): void {
  document.getElementById("btn-cancel")?.addEventListener("click", () => window.close());

  document.getElementById("btn-select-all")?.addEventListener("click", () => {
    if (state.kind !== "multi-paper") return;
    const { papers, checks } = state;
    const isExact = (idx: number) => checks[idx]?.in_library && checks[idx]?.match_type !== "fuzzy_title";
    const allSelected = papers.every((_, i) => isExact(i) || selectedPaperIndices.has(i));

    if (allSelected) {
      setSelectedPaperIndices(new Set());
    } else {
      const newIndices = new Set<number>();
      papers.forEach((_, i) => {
        if (!isExact(i)) newIndices.add(i);
      });
      setSelectedPaperIndices(newIndices);
    }
    render();
  });

  document.querySelectorAll<HTMLInputElement>(".multi-paper-item input[type='checkbox']").forEach((cb) => {
    cb.addEventListener("change", () => {
      const index = parseInt(cb.dataset.index!, 10);
      if (cb.checked) {
        selectedPaperIndices.add(index);
      } else {
        selectedPaperIndices.delete(index);
      }
      // Update button text
      const btn = document.getElementById("btn-save");
      if (btn) {
        btn.textContent = `Save ${selectedPaperIndices.size} selected`;
        (btn as HTMLButtonElement).disabled = selectedPaperIndices.size === 0;
      }
    });
  });

  // Click row to toggle checkbox
  document.querySelectorAll<HTMLDivElement>(".multi-paper-item:not(.in-library)").forEach((row) => {
    row.addEventListener("click", (e) => {
      if ((e.target as HTMLElement).tagName === "INPUT") return;
      const cb = row.querySelector("input[type='checkbox']") as HTMLInputElement;
      cb.checked = !cb.checked;
      cb.dispatchEvent(new Event("change"));
    });
  });

  document.getElementById("btn-save")?.addEventListener("click", async () => {
    if (state.kind !== "multi-paper") return;
    const papersToSave = state.papers.filter((_, i) => selectedPaperIndices.has(i));
    if (papersToSave.length === 0) return;
    await savePapers(papersToSave);
  });

  bindCollectionEvents();
}

export function bindAlreadySavedEvents(): void {
  document.getElementById("btn-close")?.addEventListener("click", () => window.close());

  document.getElementById("btn-update")?.addEventListener("click", async () => {
    if (state.kind !== "already-saved") return;
    const canonicalId = state.check.canonical_id;
    if (!canonicalId) return;

    // Show saving state
    setState({ kind: "saving" });
    render();

    try {
      const response = await chrome.runtime.sendMessage({
        type: "UPDATE_LIBRARY_ITEM",
        canonicalId,
        collectionId: selectedCollectionId,
        tags: selectedTags.length > 0 ? selectedTags : undefined,
      });

      if (response?.error) {
        setState({ kind: "error", message: response.error });
      } else {
        const collectionName = collections.find((c) => c.id === selectedCollectionId)?.name ?? "My Library";
        setState({ kind: "success", savedCount: 1, collectionName });
      }
    } catch (err) {
      setState({ kind: "error", message: err instanceof Error ? err.message : "Update failed" });
    }

    render();

    if (state.kind === "success") {
      setTimeout(() => window.close(), 1500);
    }
  });

  bindCollectionEvents();
  bindTagEvents();
}

export function bindLikelySavedEvents(): void {
  // "Update" — same as already-saved, uses canonicalId from check
  document.getElementById("btn-update")?.addEventListener("click", async () => {
    if (state.kind !== "likely-saved") return;
    const canonicalId = state.check.canonical_id;
    if (!canonicalId) return;

    setState({ kind: "saving" });
    render();

    try {
      const response = await chrome.runtime.sendMessage({
        type: "UPDATE_LIBRARY_ITEM",
        canonicalId,
        collectionId: selectedCollectionId,
        tags: selectedTags.length > 0 ? selectedTags : undefined,
      });

      if (response?.error) {
        setState({ kind: "error", message: response.error });
      } else {
        const collectionName = collections.find((c) => c.id === selectedCollectionId)?.name ?? "My Library";
        setState({ kind: "success", savedCount: 1, collectionName });
      }
    } catch (err) {
      setState({ kind: "error", message: err instanceof Error ? err.message : "Update failed" });
    }

    render();

    if (state.kind === "success") {
      setTimeout(() => window.close(), 1500);
    }
  });

  // "Save as New" — save the paper as a new entry
  document.getElementById("btn-save-new")?.addEventListener("click", async () => {
    if (state.kind !== "likely-saved") return;
    await savePapers([state.paper]);
  });

  bindCollectionEvents();
  bindTagEvents();
}

export function bindSuccessEvents(): void {
  document.getElementById("btn-done")?.addEventListener("click", () => window.close());
}

export function bindErrorEvents(): void {
  document.getElementById("btn-retry")?.addEventListener("click", () => {
    setState({ kind: "loading" });
    render();
    // Re-init
    location.reload();
  });
  document.getElementById("btn-close")?.addEventListener("click", () => window.close());
}

export function bindCollectionEvents(): void {
  document.getElementById("collection-select")?.addEventListener("change", (e) => {
    const select = e.target as HTMLSelectElement;
    setSelectedCollectionId(select.value || null);
    chrome.storage.local.set({ lastCollectionId: selectedCollectionId });
  });
}

export function bindTagEvents(): void {
  const toggleBtn = document.getElementById("toggle-tags");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      setShowTagInput(true);
      render();
      document.getElementById("tag-input")?.focus();
    });
  }

  const tagInput = document.getElementById("tag-input") as HTMLInputElement | null;
  if (tagInput) {
    tagInput.addEventListener("input", async () => {
      const query = tagInput.value.trim();
      if (query.length < 1) {
        setTagSuggestions([]);
        render();
        document.getElementById("tag-input")?.focus();
        return;
      }
      try {
        const resp = await chrome.runtime.sendMessage({ type: "SEARCH_TAGS", query });
        setTagSuggestions(
          (resp?.tags ?? []).filter((t: Tag) => !selectedTags.includes(t.name))
        );
      } catch {
        setTagSuggestions([]);
      }
      render();
      // Restore focus and value after re-render
      const newInput = document.getElementById("tag-input") as HTMLInputElement;
      if (newInput) {
        newInput.value = query;
        newInput.focus();
        newInput.setSelectionRange(query.length, query.length);
      }
    });

    tagInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === ",") {
        e.preventDefault();
        const val = tagInput.value.trim().replace(/,$/, "");
        if (val && !selectedTags.includes(val)) {
          selectedTags.push(val);
          setTagSuggestions([]);
          render();
          document.getElementById("tag-input")?.focus();
        }
      }
    });
  }

  // Tag dropdown click
  document.querySelectorAll<HTMLDivElement>(".tag-option").forEach((opt) => {
    opt.addEventListener("click", () => {
      const name = opt.dataset.tagName!;
      if (!selectedTags.includes(name)) {
        selectedTags.push(name);
        setTagSuggestions([]);
        render();
        document.getElementById("tag-input")?.focus();
      }
    });
  });

  // Tag chip remove
  document.querySelectorAll<HTMLButtonElement>(".tag-chip-remove").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const idx = parseInt(btn.dataset.tagIndex!, 10);
      selectedTags.splice(idx, 1);
      render();
    });
  });
}
