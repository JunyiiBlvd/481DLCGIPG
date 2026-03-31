#!/bin/bash
# ============================================================
# Download DLCGIPG datasets from Kaggle (private)
# ============================================================
# Prereqs:
#   - kaggle CLI installed: pip install kaggle
#   - API token at ~/.kaggle/kaggle.json
#   - You must be the dataset owner or an invited collaborator
# ============================================================

set -euo pipefail

JA_DATASET="junyiiblvc/ja-diamond-images-4c"
BE_DATASET="junyiiblvc/be-diamond-images-4c"

DEST="$(cd "$(dirname "$0")/.." && pwd)"

echo "Downloading JA dataset: ${JA_DATASET}"
kaggle datasets download -d "${JA_DATASET}" -p "${DEST}/ja_scraper/output" --unzip

echo "Downloading BE dataset: ${BE_DATASET}"
kaggle datasets download -d "${BE_DATASET}" -p "${DEST}/be_scraper/output" --unzip

echo ""
echo "=== Download complete ==="
echo "JA image count:"
find "${DEST}/ja_scraper/output" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l
echo "BE image count:"
find "${DEST}/be_scraper/output" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l
