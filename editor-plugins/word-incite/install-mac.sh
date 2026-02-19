#!/bin/bash
# Install the inCite Word add-in on macOS by sideloading the manifest.
#
# Usage: bash editor-plugins/word-incite/install-mac.sh
#
# This copies manifest.xml into Word's sideload directory so the add-in
# appears in the Home ribbon. The add-in UI is hosted at inciteref.com —
# no local server needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MANIFEST="$SCRIPT_DIR/manifest.xml"
WEF_DIR="$HOME/Library/Containers/com.microsoft.Word/Data/Documents/wef"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest.xml not found at $MANIFEST"
    exit 1
fi

echo "Installing inCite Word add-in..."

mkdir -p "$WEF_DIR"
cp "$MANIFEST" "$WEF_DIR/manifest.xml"

echo ""
echo "Done! Manifest installed to:"
echo "  $WEF_DIR/manifest.xml"
echo ""
echo "Next steps:"
echo "  1. Quit Word completely (Cmd+Q)"
echo "  2. Reopen Word"
echo "  3. Look for the 'inCite' button in the Home tab ribbon"
echo "  4. Click it, enter your API token, and start getting recommendations"
echo ""
echo "To uninstall, delete: $WEF_DIR/manifest.xml"
