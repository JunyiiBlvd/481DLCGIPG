#!/bin/bash
# ============================================================
# Download DLCGIPG dataset from Kaggle (private)
# ============================================================
# Prereqs:
#   - kaggle CLI installed: pip install kaggle
#   - API token at ~/.kaggle/kaggle.json
#   - You must be the dataset owner or an invited collaborator
# ============================================================

set -euo pipefail

# === EDIT THIS to match your Kaggle username ===
KAGGLE_DATASET="junyiiblvc/ja-diamond-images-4c"

DEST="$(cd "$(dirname "$0")/.." && pwd)/ja_scraper/output"

echo "Downloading dataset: ${KAGGLE_DATASET}"
echo "Destination: ${DEST}"
echo ""

mkdir -p "${DEST}"

kaggle datasets download -d "${KAGGLE_DATASET}" -p "${DEST}" --unzip

echo ""
echo "=== Download complete ==="
echo "Contents:"
ls -la "${DEST}/"
echo ""
echo "Image count:"
find "${DEST}" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l
