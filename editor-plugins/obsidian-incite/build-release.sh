#!/usr/bin/env bash
# Build the inCite Obsidian plugin and create a release zip.
#
# Usage: ./build-release.sh
# Output: obsidian-incite-release.zip (in this directory)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHARED_DIR="$(cd "$SCRIPT_DIR/../shared" && pwd)"
OUTPUT_ZIP="$SCRIPT_DIR/obsidian-incite-release.zip"

echo "=== Building inCite Obsidian Plugin ==="

# Step 1: Install shared package (no build step needed — esbuild bundles TS directly)
echo "  Installing @incite/shared..."
cd "$SHARED_DIR"
npm install --silent 2>/dev/null
echo "  Shared package ready."

# Step 2: Build plugin
echo "  Building obsidian-incite..."
cd "$SCRIPT_DIR"
npm install --silent 2>/dev/null
npm run build --silent 2>/dev/null
echo "  Plugin built."

# Step 3: Verify output files exist
for f in main.js styles.css manifest.json; do
    if [ ! -f "$SCRIPT_DIR/$f" ]; then
        echo "Error: $f not found after build."
        exit 1
    fi
done

# Step 4: Create release zip
rm -f "$OUTPUT_ZIP"
cd "$SCRIPT_DIR"
zip -j "$OUTPUT_ZIP" main.js styles.css manifest.json
echo ""
echo "=== Release zip created: $OUTPUT_ZIP ==="
echo ""
echo "Install instructions:"
echo "  1. Open your Obsidian vault folder"
echo "  2. Navigate to .obsidian/plugins/"
echo "  3. Create a folder called 'incite'"
echo "  4. Extract the zip contents into that folder"
echo "  5. Restart Obsidian"
echo "  6. Settings > Community Plugins > Enable 'inCite'"
echo "  7. In plugin settings, set the API URL to http://127.0.0.1:8230"
