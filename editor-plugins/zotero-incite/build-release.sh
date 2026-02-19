#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VERSION=$(node -e "console.log(require('./package.json').version)")
XPI_NAME="zotero-incite-${VERSION}.xpi"

echo "=== Building inCite for Zotero v${VERSION} ==="

# Install dependencies (including shared lib)
echo "Installing dependencies..."
npm install

# Build production bundle
echo "Building production bundle..."
npm run build

# Package .xpi (just a zip of the addon/ directory)
echo "Packaging ${XPI_NAME}..."
cd addon
zip -r "../${XPI_NAME}" . -x "*.DS_Store" -x "__MACOSX/*"
cd ..

echo "=== Built: ${XPI_NAME} ==="
echo "Install in Zotero: Tools → Add-ons → gear icon → Install from File"
