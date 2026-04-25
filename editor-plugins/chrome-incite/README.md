# inCite Chrome Extension

## Overview

The inCite Chrome extension provides citation recommendations while writing in Google Docs and Overleaf. It extracts context from the user's cursor position, queries the inCite API for relevant papers, and inserts formatted citations directly into the document. On academic sites (arXiv, PubMed, Semantic Scholar, Google Scholar, bioRxiv/medRxiv), it detects paper metadata and offers one-click save-to-library.

## Architecture

The extension uses Chrome Manifest V3 with the **service worker** as the central message hub. All communication flows through `chrome.runtime.sendMessage` to the service worker, which dispatches to domain-specific handler modules (recommendations, library, detection, Google Docs REST API). The service worker manages tab state, context menus, and cloud settings sync.

**Content scripts** provide editor integration. For Google Docs, the content script probes the hidden `texteventtarget` iframe to extract cursor context and insert placeholders (since Google Docs renders in a canvas with no DOM access to editor content). For Overleaf, two content scripts bridge the ISOLATED/MAIN world gap: the ISOLATED world script relays messages via `window.postMessage` to the MAIN world script, which has direct access to CodeMirror 6's `EditorView` API. On academic sites, content scripts run **translators** that extract paper metadata from page DOM and meta tags.

The **side panel** displays citation recommendations and handles citation insertion, bibliography rendering, and document reconciliation. The **popup** provides save-to-library UI on academic sites. Both share a common library (`@incite/shared`) for API client, citation formatting, and citation tracking.

## Directory Structure

```
src/
  background/                      Service worker + handler modules
    service-worker.ts              Entry point: lifecycle, tab management, context menus
    message-router.ts              Central message dispatch to domain handlers
    state.ts                       Shared mutable state (detected papers, GDocs cache)
    client.ts                      API client singleton, tab utilities
    hotkey.ts                      Configurable hotkey injection
    recommendation-handlers.ts     Queries, health check, citation insertion
    library-handlers.ts            Save-to-library, collections, tags, PDF upload
    detection-handlers.ts          Paper detection with inline page-context extraction
    gdocs-handlers.ts              Google Docs REST API operations (cursor, citations, bib)
  content/                         Content scripts
    googledocs.ts                  Google Docs cursor context via texteventtarget probing
    overleaf-main.ts               Overleaf MAIN world (CodeMirror 6 API access)
    overleaf-isolated.ts           Overleaf ISOLATED world (bridge to MAIN)
    translator-runner.ts           Runs paper detection on academic sites
    shared.ts                      Toast notification utility
  panel/                           Side panel (recommendations UI)
    panel.ts                       Entry point, initialization, event wiring
    panel-state.ts                 Shared panel state
    panel-recommendations.ts       Result display, health, collections
    panel-citations.ts             Citation insertion, refresh, selection
    panel-bibliography.ts          Bibliography rendering and export
  popup/                           Popup (save-to-library UI)
    popup.ts                       Entry point, state machine, init
    popup-state.ts                 Shared popup state and types
    popup-renderers.ts             All render* functions
    popup-events.ts                All bind*Events functions
  options/                         Extension options page
    options.ts                     API token, citation format, hotkey settings
  shared/                          Shared utilities
    types.ts                       Core types (settings, editor type)
    settings.ts                    Chrome storage + cloud sync
    constants.ts                   Default settings
    citation-storage.ts            Citation tracking via chrome.storage.local
    hotkey.ts                      Hotkey parsing/matching (for options page)
    gdocs-api.ts                   Google Docs REST API client (OAuth)
    gdocs-index-resolver.ts        Cursor position -> document index resolution
    gdocs-citation-inserter.ts     Citation/bibliography insertion via REST API
    pdf-uploader.ts                PDF download + upload with retry queue
    messages/                      Message protocol types (by domain)
      panel.ts                     Panel -> service worker messages
      gdocs.ts                     Google Docs REST API messages
      library.ts                   Save-to-library messages
      responses.ts                 Service worker response types
      send.ts                      Type-safe message sender
      index.ts                     Re-exports and composite union types
  translators/                     Paper metadata extractors
    types.ts                       Translator interface, PaperMetadata type
    utils.ts                       Shared getMeta/getAllMeta/extractYear utilities
    registry.ts                    Translator selection by URL
    generic.ts                     Fallback structured text extractor
    arxiv.ts                       arXiv abstract/abs pages
    pubmed.ts                      PubMed article pages
    semantic-scholar.ts            Semantic Scholar paper pages
    google-scholar.ts              Google Scholar results pages
    biorxiv.ts                     bioRxiv and medRxiv preprint pages
```

## Message Flow

Three primary communication patterns:

### Recommendations (writing in Google Docs / Overleaf)

```
Panel -> GET_RECOMMENDATIONS -> Service Worker -> GET_CONTEXT -> Content Script
Content Script -> CONTEXT_RESPONSE -> Service Worker -> inCite API -> Panel
```

The panel requests recommendations; the service worker asks the content script for cursor context (text around the cursor); the content script extracts it (texteventtarget probe for GDocs, CodeMirror API for Overleaf); the service worker sends it to the API and returns results to the panel.

### Save to Library (browsing academic sites)

```
Popup -> GET_DETECTED_PAPERS -> Service Worker -> (cache or inject detector)
Popup -> SAVE_PAPERS -> Service Worker -> inCite API -> Popup
```

When the popup opens, it asks for detected papers. The service worker checks its per-tab cache; if empty, it injects a detection script via `chrome.scripting.executeScript`. The user selects papers and triggers save, which the service worker forwards to the API.

### Paper Detection (automatic, on academic sites)

```
Content Script -> PAGE_PAPERS_DETECTED -> Service Worker -> Badge Update
```

Translator content scripts run automatically on matched URLs, extract paper metadata from the page, and send it to the service worker. The service worker caches the result per tab and updates the toolbar badge.

## Build & Development

```bash
npm install          # Install dependencies
npm run build        # Bundle with esbuild (output: dist/)
npm run watch        # Rebuild on file changes
npm test             # Run vitest tests
npm run typecheck    # TypeScript type checking
```

**Loading unpacked:** Chrome -> Extensions -> Developer mode -> Load unpacked -> select the `chrome-incite/` directory (uses `dist/` for bundled scripts).

**Shared library:** The extension depends on `@incite/shared` (linked via `file:../shared`). If you modify the shared library, rebuild it before rebuilding the extension.

## Adding a Translator

Translators extract paper metadata from specific academic websites. To add support for a new site:

1. Create `src/translators/{site}.ts`
2. Import `getMeta`, `getAllMeta`, `extractYear` from `./utils`
3. Implement the `Translator` interface:
   - `name`: human-readable name (e.g., `"springer"`)
   - `urlPatterns`: array of `RegExp` matching the site's paper page URLs
   - `detect(document)`: return `{ type: "single" }` or `{ type: "multiple" }` if papers are found, `null` otherwise
   - `extractSingle(document)`: extract metadata for a single-paper page
   - `extractMultiple(document)`: extract metadata for a search/listing page
4. Register in `src/translators/registry.ts`: add to the `TRANSLATORS` array, ordered by specificity (most specific first; generic is always the fallback)
5. Add URL patterns to `manifest.json` content scripts if the translator needs a dedicated content script (rather than relying on the injected generic detector)

## Key Design Decisions

- **Texteventtarget copy trick:** Google Docs renders in a canvas -- there is no DOM access to editor content. The content script accesses the hidden `.docs-texteventtarget-iframe`, clears its contenteditable, dispatches a synthetic `copy` event, and reads the text that Google Docs populates in response. This is the same technique Zotero uses and has been stable for 7+ years including the 2021 canvas migration.

- **Placeholder-based citation insertion:** The Google Docs REST API cannot access the live cursor position. We use a Zotero-inspired approach: the content script inserts a unique placeholder at the cursor via a synthetic paste event on the texteventtarget, the service worker finds it in the document via the REST API, deletes it (restoring any text it replaced), and inserts the real citation at the resolved position.

- **ISOLATED/MAIN world bridge:** Overleaf's CodeMirror 6 API is only accessible from the MAIN execution world, but Chrome extensions run content scripts in the ISOLATED world by default. Two content scripts bridge the gap: `overleaf-isolated.ts` (ISOLATED world) relays messages via `window.postMessage` to `overleaf-main.ts` (MAIN world), which has direct access to the `EditorView` instance. The ISOLATED script includes retry logic because CM6 may not be initialized when the MAIN script first loads.

- **Inline detection extraction:** `detection-handlers.ts` contains a ~200-line function that runs in page context via `chrome.scripting.executeScript`. It cannot import modules -- all extraction logic (DOM traversal, meta tag reading, structured text parsing) must be self-contained within the function body. This duplication is inherent to Chrome's extension security model for scripts injected into arbitrary pages.

- **Tab-mode switching:** The service worker configures each tab as either "writing" (Google Docs/Overleaf -> side panel for recommendations) or "academic" (arXiv, PubMed, etc. -> popup for save-to-library). This is done by toggling `chrome.sidePanel.setOptions` and `chrome.action.setPopup` per tab, so clicking the toolbar icon opens the appropriate UI.
