/**
 * Overleaf ISOLATED world content script.
 *
 * Chrome extensions run content scripts in the ISOLATED world by default,
 * which cannot access the page's JavaScript context (and therefore cannot
 * access Overleaf's CodeMirror 6 EditorView). To bridge this gap:
 *
 *   Service Worker <-> ISOLATED world (this file) <-> MAIN world (overleaf-main.ts)
 *                  chrome.runtime.sendMessage      window.postMessage
 *
 * This script receives messages from the service worker via chrome.runtime,
 * forwards them to the MAIN world script via window.postMessage, and relays
 * responses back. The MAIN world script (declared with "world": "MAIN" in
 * manifest.json) has direct access to the CM6 EditorView for text extraction
 * and citation insertion.
 *
 * Includes retry logic because CM6 may not be fully initialized when the
 * MAIN world script first loads (especially on slow connections).
 */
import { showToast } from "./shared";

// Map of pending requests waiting for responses from MAIN world
const pendingRequests = new Map<string, (response: unknown) => void>();

const BRIDGE_TIMEOUT_MS = 10000;

// Listen for messages from service worker
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === "GET_CONTEXT" || message.type === "INSERT_CITATION") {
    sendWithRetry(message, sendResponse);
    return true; // Async response
  }
});

/**
 * Send message to MAIN world with one retry if the first attempt times out.
 * CM6 may not be initialized when the MAIN world script first runs.
 */
function sendWithRetry(
  message: { type: string; requestId?: string },
  sendResponse: (response: unknown) => void,
  isRetry = false
) {
  const requestId = message.requestId ?? crypto.randomUUID();

  // Forward to MAIN world via window.postMessage
  window.postMessage(
    {
      source: "incite-isolated",
      type: message.type,
      requestId,
      payload: message,
    },
    "*"
  );

  // Wait for response from MAIN world
  const timeout = setTimeout(() => {
    pendingRequests.delete(requestId);

    if (!isRetry) {
      // Retry once — CM6 might not have been ready
      sendWithRetry(message, sendResponse, true);
      return;
    }

    // Second attempt also failed
    if (message.type === "GET_CONTEXT") {
      sendResponse({
        type: "CONTEXT_RESPONSE",
        requestId,
        error:
          "Overleaf editor did not respond. Try refreshing the page, " +
          "or use the manual text input in the panel.",
      });
    } else {
      sendResponse({ type: "INSERT_RESULT", success: false });
    }
  }, BRIDGE_TIMEOUT_MS);

  pendingRequests.set(requestId, (response) => {
    clearTimeout(timeout);
    sendResponse(response);
  });
}

// Listen for responses from MAIN world
window.addEventListener("message", (event) => {
  if (event.source !== window) return;
  const data = event.data;
  if (!data || data.source !== "incite-main") return;

  const resolve = pendingRequests.get(data.requestId);
  if (resolve) {
    pendingRequests.delete(data.requestId);

    if (data.type === "CONTEXT_RESPONSE") {
      resolve(data.payload);
    } else if (data.type === "INSERT_RESULT") {
      if (data.payload?.success) {
        showToast("Citation inserted");
      }
      resolve(data.payload);
    }
  }
});
