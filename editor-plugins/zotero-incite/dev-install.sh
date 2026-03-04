#!/usr/bin/env bash
# Build the inCite Zotero plugin XPI.
# Usage: ./dev-install.sh
#
# After building, install in Zotero via:
#   Tools → Add-ons → gear icon → Install Add-on From File → select the XPI
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ADDON_DIR="$SCRIPT_DIR/addon"
XPI_NAME="zotero-incite.xpi"

echo "==> Building TypeScript..."
cd "$SCRIPT_DIR"
npm run build

echo "==> Packaging XPI..."
cd "$ADDON_DIR"
rm -f "$SCRIPT_DIR/$XPI_NAME"
zip -r "$SCRIPT_DIR/$XPI_NAME" . -x '*.DS_Store'

echo ""
echo "==> XPI ready: $SCRIPT_DIR/$XPI_NAME"
echo ""
echo "Install in Zotero:"
echo "  Tools → Add-ons → gear icon → Install Add-on From File"
echo "  Select: $SCRIPT_DIR/$XPI_NAME"
