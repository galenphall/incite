# inCite for Zotero

A Zotero 7 plugin that connects your Zotero library to the InCite cloud service or hosts InCite locally.

## Features

- **Cloud mode**: Sign in and upload your library to inciteref.com — no Python required
- **Local mode**: The plugin installs InCite, indexes your library, and starts the server — you just need Python 3 and pip
- **Library sync**: Upload papers and PDFs directly from Zotero, or sync via the Zotero cloud API

## Requirements

- Zotero 7+
- **Cloud mode**: An account at inciteref.com
- **Local mode**: Python 3 and pip installed on your machine (the plugin handles the rest)

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
- `src/item-pane-section.ts` — Item pane UI section
- `src/text-query-dialog.ts` — Text query input modal
- `src/api-client.ts` — `ZoteroTransport` wrapping `Zotero.HTTP.request()`
- `src/prefs.ts` — Read/write Zotero preferences
- `addon/` — Zotero 7 WebExtension manifest, bootstrap, XHTML prefs, icons
