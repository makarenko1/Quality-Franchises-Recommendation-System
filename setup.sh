#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_NAME="repo_data.zip"
ARCHIVE_PATH="$REPO_ROOT/$ARCHIVE_NAME"

# ── 1. Update the repo ────────────────────────────────────────────────────────
echo "==> Updating repository..."
git -C "$REPO_ROOT" fetch
git -C "$REPO_ROOT" pull

# ── 2. Google Drive file ID ───────────────────────────────────────────────────
GDRIVE_FILE_ID="${GDRIVE_FILE_ID:-1cjxdEiBxPIDaF56qllh_8A5u5bGqXBLr}"

# ── 3. Ensure gdown is available ─────────────────────────────────────────────
if ! command -v gdown &>/dev/null; then
    echo "==> Installing gdown (Google Drive downloader)..."
    pip install gdown --quiet
fi

# ── 4. Download archive from Google Drive ────────────────────────────────────
echo "==> Downloading $ARCHIVE_NAME from Google Drive..."
rm -f "$ARCHIVE_PATH"
gdown "https://drive.google.com/uc?id=${GDRIVE_FILE_ID}" -O "$ARCHIVE_PATH"

# ── 5. Unpack into repo root ──────────────────────────────────────────────────
echo "==> Unpacking archive..."
unzip -o "$ARCHIVE_PATH" -d "$REPO_ROOT"
rm "$ARCHIVE_PATH"

echo ""
echo "==> Setup complete."
