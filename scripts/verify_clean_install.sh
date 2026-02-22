#!/usr/bin/env bash
set -euo pipefail

# Verify incite installs and works on a clean machine via Docker.
# Syncs to a temp directory first to simulate the public repo.
# Usage: verify_clean_install.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_NAME="incite-verify"
STAGING_DIR=$(mktemp -d)
trap 'rm -rf "$STAGING_DIR"' EXIT

# Sync allowed files to staging (simulates public repo)
echo "Syncing public-repo files to staging directory..."
mkdir -p "$STAGING_DIR" && git -C "$STAGING_DIR" init -q
bash "$SCRIPT_DIR/sync_to_public.sh" "$STAGING_DIR"

echo ""
echo "Building verification image from synced files..."
docker build \
    -f "$STAGING_DIR/scripts/Dockerfile.verify" \
    -t "$IMAGE_NAME" \
    "$STAGING_DIR"

echo ""
echo "Running verification container..."
docker run --rm "$IMAGE_NAME"

echo ""
echo "Clean-machine verification passed."
