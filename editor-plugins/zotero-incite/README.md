# inCite for Zotero

A Zotero 7 plugin that shows citation recommendations from your own library.

## Features

- **Item pane section**: Select a paper in Zotero → see related papers from your library in the right panel
- **Text query**: Paste a writing passage via Tools → "inCite: Find Related Papers..." (or Cmd/Ctrl+Shift+I) → get recommendations
- **Local & cloud modes**: Works with `incite serve` locally or with the cloud service at inciteref.com

## Requirements

- Zotero 7+
- Either a local inCite server (`incite serve --embedder minilm-ft`) or a cloud account at inciteref.com

## Installation

1. Download the latest `.xpi` from [Releases](https://github.com/galenphall/incite/releases)
2. In Zotero: Tools → Add-ons → gear icon → Install Add-on From File
3. Select the `.xpi` file
4. Restart Zotero

## Configuration

Go to Zotero → Settings → inCite to configure:

- **API Mode**: Local (default) or Cloud
- **Server URLs**: Local server address or cloud URL
- **API Token**: Required for cloud mode
- **Number of results**: How many recommendations to show (default: 10)
- **Author boost**: Weight for author matching (default: 1.0)
- **Show evidence paragraphs**: Display matching text snippets

## Development

```bash
npm install
npm run dev      # development build (with sourcemaps)
npm run build    # production build (minified)
npm run release  # build + package .xpi
```

The plugin uses the `@incite/shared` library for API client, types, and UI rendering functions shared across all inCite editor plugins.

## Architecture

- `src/index.ts` — Entry point, exports lifecycle hooks
- `src/hooks.ts` — Startup/shutdown: registers pane section, prefs, menu items
- `src/item-pane-section.ts` — Core feature: recommendations in the Zotero item pane
- `src/text-query-dialog.ts` — Text input modal for finding related papers
- `src/api-client.ts` — `ZoteroTransport` wrapping `Zotero.HTTP.request()`
- `src/prefs.ts` — Read/write Zotero preferences
- `addon/` — Zotero 7 WebExtension manifest, bootstrap, XHTML prefs, icons
