// MAIN world script — has access to page's JS context (CodeMirror 6)

interface CM6EditorView {
  state: {
    doc: { toString(): string; sliceString(from: number, to: number): string };
    selection: { main: { from: number; to: number; head: number } };
  };
  dispatch(spec: { changes: { from: number; to?: number; insert: string } }): void;
}

/**
 * Try multiple strategies to find the CM6 EditorView instance.
 *
 * 1. .cmView.view on the .cm-editor element (standard CM6 DOM binding)
 * 2. Walk up from .cm-content to find the view instance
 * 3. Overleaf-specific globals (_ide, editor)
 */
function getEditorView(): CM6EditorView | null {
  const cmEditor = document.querySelector(".cm-editor") as HTMLElement | null;
  if (!cmEditor) return null;

  // Strategy 1: Direct .cmView access (standard CM6)
  const cmView = (cmEditor as unknown as Record<string, unknown>).cmView as
    | { view: CM6EditorView }
    | undefined;
  if (cmView?.view) return cmView.view;

  // Strategy 2: Walk up from .cm-content
  const cmContent = document.querySelector(".cm-content") as HTMLElement | null;
  if (cmContent) {
    let el: HTMLElement | null = cmContent;
    while (el && el !== document.body) {
      const view = (el as unknown as Record<string, unknown>).cmView as
        | { view: CM6EditorView }
        | undefined;
      if (view?.view) return view.view;
      el = el.parentElement;
    }
  }

  // Strategy 3: Overleaf-specific globals
  const win = window as unknown as Record<string, unknown>;
  for (const key of ["_ide", "editor"]) {
    const global = win[key] as Record<string, unknown> | undefined;
    if (global) {
      // Try common Overleaf paths
      const view =
        (global.editorManager as Record<string, unknown>)?.currentView ??
        global.editorView ??
        global.view;
      if (view && typeof (view as CM6EditorView).state?.doc?.toString === "function") {
        return view as CM6EditorView;
      }
    }
  }

  return null;
}

/**
 * Get text using window.getSelection() as a last resort.
 * CM6's contenteditable means selection API works for selected text (not cursor context).
 */
function getSelectionFallback(): string | null {
  const selection = window.getSelection();
  const text = selection?.toString().trim();
  return text && text.length > 0 ? text : null;
}

// Retry finding the editor with delay — CM6 may not be initialized yet
let editorReady = false;
let retryCount = 0;
const MAX_RETRIES = 10;
const RETRY_INTERVAL = 1000;

function waitForEditor(): void {
  if (editorReady || retryCount >= MAX_RETRIES) return;
  retryCount++;

  const view = getEditorView();
  if (view) {
    editorReady = true;
    return;
  }

  // Also watch for the editor element to appear in DOM
  setTimeout(waitForEditor, RETRY_INTERVAL);
}

// Start waiting for editor
waitForEditor();

// Also observe DOM for late-loading .cm-editor elements
const observer = new MutationObserver(() => {
  if (!editorReady && getEditorView()) {
    editorReady = true;
    observer.disconnect();
  }
});
observer.observe(document.body, { childList: true, subtree: true });

// Disconnect observer after 30s to avoid unnecessary overhead
setTimeout(() => observer.disconnect(), 30000);

// Listen for messages from ISOLATED world
window.addEventListener("message", (event) => {
  if (event.source !== window) return;
  const data = event.data;
  if (!data || data.source !== "incite-isolated") return;

  const { type, requestId, payload } = data;

  if (type === "GET_CONTEXT") {
    handleGetContext(requestId);
  } else if (type === "INSERT_CITATION") {
    handleInsertCitation(requestId, payload?.citation);
  }
});

function handleGetContext(requestId: string) {
  const view = getEditorView();

  if (view) {
    const { from, to, head } = view.state.selection.main;

    if (from !== to) {
      // Text is selected — return it
      const text = view.state.doc.sliceString(from, to);
      respond(requestId, "CONTEXT_RESPONSE", {
        type: "CONTEXT_RESPONSE",
        requestId,
        text,
      });
    } else {
      // Cursor only — return full text + offset for context extraction in service worker
      const fullText = view.state.doc.toString();
      respond(requestId, "CONTEXT_RESPONSE", {
        type: "CONTEXT_RESPONSE",
        requestId,
        fullText,
        cursorOffset: head,
      });
    }
    return;
  }

  // Fallback: try window.getSelection() for selected text
  const fallbackText = getSelectionFallback();
  if (fallbackText) {
    respond(requestId, "CONTEXT_RESPONSE", {
      type: "CONTEXT_RESPONSE",
      requestId,
      text: fallbackText,
    });
    return;
  }

  respond(requestId, "CONTEXT_RESPONSE", {
    type: "CONTEXT_RESPONSE",
    requestId,
    error:
      "Could not find Overleaf editor. Make sure a document is open. " +
      "If the issue persists, select text and use the manual input in the panel.",
  });
}

function handleInsertCitation(requestId: string, citation: string) {
  const view = getEditorView();
  if (!view || !citation) {
    respond(requestId, "INSERT_RESULT", { success: false });
    return;
  }

  const cursor = view.state.selection.main.head;
  view.dispatch({
    changes: { from: cursor, insert: citation },
  });

  respond(requestId, "INSERT_RESULT", { success: true, method: "direct" });
}

function respond(requestId: string, type: string, payload: unknown) {
  window.postMessage(
    {
      source: "incite-main",
      type,
      requestId,
      payload,
    },
    "*"
  );
}
