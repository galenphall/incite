#!/usr/bin/env bash
set -euo pipefail

# Verify incite installs and works on a clean machine via Docker.
# Usage: verify_clean_install.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_NAME="incite-verify"

echo "Building verification image..."
docker build \
    -f "$SCRIPT_DIR/Dockerfile.verify" \
    -t "$IMAGE_NAME" \
    "$PROJECT_DIR"

echo ""
echo "Running verification container..."
docker run --rm "$IMAGE_NAME"

echo ""
echo "Clean-machine verification passed."
