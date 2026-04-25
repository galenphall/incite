import { showToast } from "./shared";

/**
 * Google Docs content script.
 *
 * Handles cursor context extraction via the `.docs-texteventtarget-iframe`
 * (Zotero-style probing) and falls back to clipboard-based selection extraction.
 *
 * The texteventtarget iframe is how Google Docs receives text input — it has
 * survived 7+ years including the 2021 canvas migration. Zotero relies on
 * the same element for their Google Docs integration.
 */

// ---------------------------------------------------------------------------
// Message listener
// ---------------------------------------------------------------------------

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  switch (message.type) {
    case "GET_CURSOR_CONTEXT": {
      getCursorContext()
        .then((result) => {
          sendResponse({
            type: "CURSOR_CONTEXT_RESPONSE",
            requestId: message.requestId,
            ...result,
          });
        })
        .catch((err) => {
          sendResponse({
            type: "CURSOR_CONTEXT_RESPONSE",
            requestId: message.requestId,
            error: err?.message ?? "Unknown error",
          });
        });
      return true; // Async response
    }

    case "GET_CONTEXT": {
      extractText().then((result) => {
        sendResponse({ type: "CONTEXT_RESPONSE", requestId: message.requestId, ...result });
      });
      return true;
    }

    case "INSERT_CITATION": {
      // Legacy clipboard insertion — kept as fallback for Overleaf.
      navigator.clipboard
        .writeText(message.citation)
        .then(() => {
          showToast(`Copied "${message.citation}" — paste with Cmd/Ctrl+V`);
          sendResponse({ type: "INSERT_RESULT", success: true, method: "clipboard" });
        })
        .catch((err) => {
          console.error("Failed to copy citation to clipboard:", err);
          sendResponse({ type: "INSERT_RESULT", success: false, error: err?.message });
        });
      return true;
    }

    case "INSERT_PLACEHOLDER": {
      // Zotero-style: insert a placeholder at the cursor via synthetic paste event.
      // The service worker will find this placeholder via REST API and replace it
      // with the real citation text.
      insertPlaceholderAtCursor(message.placeholder)
        .then((success) => sendResponse({ type: "PLACEHOLDER_RESULT", success }))
        .catch((err) => {
          console.error("Placeholder insertion failed:", err);
          sendResponse({ type: "PLACEHOLDER_RESULT", success: false });
        });
      return true;
    }
  }
});

// ---------------------------------------------------------------------------
// Texteventtarget iframe access
// ---------------------------------------------------------------------------

/**
 * Get the contenteditable element inside the Google Docs texteventtarget iframe.
 * This iframe is structurally necessary for Google Docs text input and has been
 * stable for 7+ years across all Google Docs UI changes.
 */
function getTextEventTarget(): HTMLElement | null {
  const iframe = document.querySelector<HTMLIFrameElement>(
    ".docs-texteventtarget-iframe"
  );
  if (!iframe) return null;

  try {
    const doc = iframe.contentDocument;
    if (!doc) return null;
    return doc.querySelector("[contenteditable]");
  } catch {
    // Cross-origin restriction — shouldn't happen for same-origin iframes
    return null;
  }
}

/**
 * Dispatch a synthetic keyboard event to the texteventtarget element.
 * Google Docs listens for these events to handle text editing.
 */
function dispatchKey(
  target: HTMLElement,
  key: string,
  opts: Partial<KeyboardEventInit> = {}
): void {
  const eventInit: KeyboardEventInit = {
    key,
    code: key,
    bubbles: true,
    cancelable: true,
    ...opts,
  };
  target.dispatchEvent(new KeyboardEvent("keydown", eventInit));
  target.dispatchEvent(new KeyboardEvent("keyup", eventInit));
}

/**
 * Read the current content of the texteventtarget's contenteditable.
 * Google Docs populates this element in response to copy events.
 */
function readTargetContent(target: HTMLElement): string {
  return target.textContent ?? "";
}

// ---------------------------------------------------------------------------
// Cursor context extraction (Zotero-style probing)
// ---------------------------------------------------------------------------

interface CursorProbeResult {
  paragraphText?: string;
  cursorOffset?: number;
  textBefore?: string;
  textAfter?: string;
  error?: string;
}

/**
 * Extract cursor context using the Zotero-style texteventtarget probing method.
 *
 * Steps:
 * 1. Access `.docs-texteventtarget-iframe` → contenteditable
 * 2. Clear the contenteditable's innerHTML
 * 3. Dispatch a synthetic `copy` event — Docs populates the contenteditable
 *    with the current selection/cursor context HTML
 * 4. Read the resulting text content
 * 5. Probe left/right of cursor with Shift+Arrow keys to get surrounding text
 * 6. Undo probes by dispatching opposite arrow keys with Shift
 */
async function getCursorContext(): Promise<CursorProbeResult> {
  const target = getTextEventTarget();
  if (!target) {
    return { error: "Could not access Google Docs text input element." };
  }

  // Step 1: Get text at cursor position via copy probe
  target.innerHTML = "";
  dispatchCopyEvent(target);
  await sleep(50);

  const cursorText = readTargetContent(target).trim();
  if (!cursorText) {
    return { error: "No text at cursor position. Place cursor in a paragraph and try again." };
  }

  // Return the selected/cursor text directly.
  // Probing left/right with Shift+Arrow is unreliable in canvas-mode Google Docs
  // and causes duplication when text is already selected.
  return {
    paragraphText: cursorText,
    cursorOffset: 0,
  };
}

/**
 * Probe in a direction from the cursor to extract surrounding text.
 * Uses Shift+Arrow keys to extend selection, then reads the new content.
 */
async function probeDirection(
  target: HTMLElement,
  direction: "left" | "right",
  maxChars: number
): Promise<string> {
  const arrowKey = direction === "left" ? "ArrowLeft" : "ArrowRight";
  const undoKey = direction === "left" ? "ArrowRight" : "ArrowLeft";

  let accumulated = "";
  let prevText = "";
  const steps = Math.min(maxChars, 100); // Cap iterations

  for (let i = 0; i < steps; i++) {
    // Extend selection in the given direction
    dispatchKey(target, arrowKey, { shiftKey: true });
    await sleep(10);

    // Read the extended selection via copy
    target.innerHTML = "";
    dispatchCopyEvent(target);
    await sleep(20);

    const currentText = readTargetContent(target).trim();

    // If text didn't change, we've hit a boundary
    if (currentText === prevText) break;

    // Extract the newly selected characters
    if (direction === "left") {
      // Text grows at the beginning
      const newPart = currentText.slice(
        0,
        currentText.length - (prevText ? prevText.length : 0)
      );
      accumulated = newPart + accumulated;
    } else {
      // Text grows at the end
      const newPart = currentText.slice(prevText ? prevText.length : 0);
      accumulated = accumulated + newPart;
    }

    prevText = currentText;

    // Stop at paragraph boundaries (newline)
    if (accumulated.includes("\n")) {
      accumulated = accumulated.split("\n")[direction === "left" ? accumulated.split("\n").length - 1 : 0];
      break;
    }
  }

  // Undo the selection by pressing the opposite arrow key the same number of times
  // Actually, we just need to collapse the selection — press the arrow without Shift
  dispatchKey(target, undoKey);
  await sleep(10);

  return accumulated;
}

/**
 * Dispatch a synthetic copy event to the texteventtarget.
 * Google Docs responds by populating the contenteditable with
 * the current selection's content.
 */
function dispatchCopyEvent(target: HTMLElement): void {
  const event = new ClipboardEvent("copy", {
    bubbles: true,
    cancelable: true,
    clipboardData: new DataTransfer(),
  });
  target.dispatchEvent(event);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// Placeholder insertion (Zotero-style)
// ---------------------------------------------------------------------------

/**
 * Insert a placeholder string at the current cursor position by dispatching
 * a synthetic paste event to the texteventtarget.
 *
 * This mirrors Zotero's approach: they insert a placeholder at the cursor
 * from the browser side (since neither Apps Script nor the REST API can access
 * the live cursor), then find and replace it via the REST API.
 *
 * The paste event on the texteventtarget is the same mechanism Google Docs
 * uses for real paste operations — it reads clipboardData from the event.
 */
async function insertPlaceholderAtCursor(placeholder: string): Promise<boolean> {
  const target = getTextEventTarget();
  if (!target) return false;

  // Focus the texteventtarget so Google Docs knows where to insert
  target.focus();
  await sleep(50);

  // Note: if text is selected, the paste will replace it. The service worker
  // captures the selected text beforehand and restores it via the REST API.

  // Dispatch a synthetic paste event with the placeholder in clipboardData.
  // Google Docs listens for paste events on the texteventtarget and processes
  // the clipboardData content, inserting it at the current cursor position.
  const dt = new DataTransfer();
  dt.setData("text/plain", placeholder);
  const pasteEvent = new ClipboardEvent("paste", {
    bubbles: true,
    cancelable: true,
    clipboardData: dt,
  });
  target.dispatchEvent(pasteEvent);

  // Wait for Google Docs to process the paste
  await sleep(300);

  return true;
}

// ---------------------------------------------------------------------------
// Legacy text extraction (clipboard-based, for explicit selection)
// ---------------------------------------------------------------------------

/**
 * Multi-strategy text extraction for Google Docs.
 * Used as fallback when the user has explicitly selected text.
 */
async function extractText(): Promise<{ text?: string; error?: string }> {
  // Strategy 1: Direct selection API
  const selection = window.getSelection();
  const selectedText = selection?.toString().trim();
  if (selectedText && selectedText.length > 0) {
    return { text: selectedText };
  }

  // Strategy 2: Clipboard-based extraction
  try {
    const clipboardText = await extractViaClipboard();
    if (clipboardText && clipboardText.length > 0) {
      return { text: clipboardText };
    }
  } catch (err) {
    console.debug("Clipboard extraction failed:", err);
  }

  return {
    error:
      "Could not extract text. Select text in your document and try again, " +
      "or use the manual text input in the panel below.",
  };
}

/**
 * Extract selected text via the clipboard.
 * Saves the current clipboard content, triggers a copy, reads the result, and restores.
 */
async function extractViaClipboard(): Promise<string> {
  let savedClipboard: string | null = null;
  try {
    savedClipboard = await navigator.clipboard.readText();
  } catch {
    // Cannot read clipboard
  }

  document.execCommand("copy");
  await sleep(100);

  const copiedText = await navigator.clipboard.readText();

  if (savedClipboard !== null && savedClipboard !== copiedText) {
    try {
      await navigator.clipboard.writeText(savedClipboard);
    } catch {
      // Clipboard restore failed
    }
  }

  return copiedText?.trim() ?? "";
}
