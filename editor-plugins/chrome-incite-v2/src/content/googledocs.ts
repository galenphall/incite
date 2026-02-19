import { showToast } from "./shared";

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  switch (message.type) {
    case "GET_CONTEXT": {
      extractText().then((result) => {
        sendResponse({ type: "CONTEXT_RESPONSE", requestId: message.requestId, ...result });
      });
      return true; // Async response
    }

    case "INSERT_CITATION": {
      // Google Docs uses a custom canvas editor — clipboard is the most reliable insertion method
      navigator.clipboard.writeText(message.citation).then(() => {
        showToast(`Copied "${message.citation}" -- paste with Cmd/Ctrl+V`);
        sendResponse({ type: "INSERT_RESULT", success: true, method: "clipboard" });
      }).catch((err) => {
        console.error("Failed to copy citation to clipboard:", err);
        sendResponse({ type: "INSERT_RESULT", success: false, error: err?.message });
      });
      return true; // Async response
    }
  }
});

/**
 * Multi-strategy text extraction for Google Docs.
 *
 * Strategy 1: window.getSelection() — works in non-canvas mode and some configurations.
 * Strategy 2: Clipboard extraction — Google Docs intercepts execCommand('copy') even in
 *   canvas mode, so we can save clipboard -> copy -> read -> restore.
 * Strategy 3: Return an error prompting the user to use manual paste in the panel.
 */
async function extractText(): Promise<{ text?: string; error?: string }> {
  // Strategy 1: Direct selection API
  const selection = window.getSelection();
  const selectedText = selection?.toString().trim();
  if (selectedText && selectedText.length > 0) {
    return { text: selectedText };
  }

  // Strategy 2: Clipboard-based extraction
  // Google Docs intercepts copy commands internally, so execCommand('copy') works
  // even in canvas mode where getSelection() returns empty.
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
  // Save current clipboard content
  let savedClipboard: string | null = null;
  try {
    savedClipboard = await navigator.clipboard.readText();
  } catch (err) {
    console.debug("Could not read clipboard to save:", err);
  }

  // Trigger copy — Google Docs intercepts this even in canvas mode
  document.execCommand("copy");

  // Brief delay for Docs to process the copy
  await new Promise((resolve) => setTimeout(resolve, 100));

  // Read the copied text
  const copiedText = await navigator.clipboard.readText();

  // Restore previous clipboard if we saved it and it's different
  if (savedClipboard !== null && savedClipboard !== copiedText) {
    try {
      await navigator.clipboard.writeText(savedClipboard);
    } catch (err) {
      console.debug("Clipboard restore failed:", err);
    }
  }

  return copiedText?.trim() ?? "";
}
