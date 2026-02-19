import { findTranslator } from "../translators/registry";
import type { PaperMetadata, DetectionResult } from "../translators/types";

// Run detection on page load
(function() {
  const translator = findTranslator(location.href);
  const detection = translator.detect(document);

  if (detection) {
    // Pre-extract papers immediately so they're ready when popup opens
    let papers: PaperMetadata[] | undefined;
    if (detection.type === "single") {
      const paper = translator.extractSingle(document);
      papers = paper ? [paper] : undefined;
    } else {
      papers = translator.extractMultiple(document);
    }

    chrome.runtime.sendMessage({
      type: "PAGE_PAPERS_DETECTED",
      detection,
      papers,
      translatorName: translator.name,
    });
  }

  // Listen for extraction requests from popup (in case content changed)
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type === "EXTRACT_PAPERS") {
      const t = findTranslator(location.href);
      const d = t.detect(document);
      if (!d) {
        sendResponse({ papers: [], type: null });
        return;
      }

      let extracted: PaperMetadata[];
      if (d.type === "single") {
        const paper = t.extractSingle(document);
        extracted = paper ? [paper] : [];
      } else {
        extracted = t.extractMultiple(document);
      }
      sendResponse({ papers: extracted, type: d.type });
    }
    return true; // keep channel open
  });
})();
