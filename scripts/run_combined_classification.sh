#!/usr/bin/env bash
# run_combined_classification.sh — Train ResNet50, EfficientNetV2, ViT on all 3 combined subsets
# 9 runs total: 3 archs × 3 subsets

set -euo pipefail

PYTHON=/mnt/storage/projects/DLCGIPG/ja_scraper/venv/bin/python
SCRIPT=/mnt/storage/projects/DLCGIPG/src/train_combined.py

cd /mnt/storage/projects/DLCGIPG

run() {
    local arch="$1" subset="$2"
    echo ""
    echo "========================================================"
    echo "  ${arch} / ${subset}"
    echo "========================================================"
    $PYTHON $SCRIPT --arch "$arch" --subset "$subset"
}

for subset in combined_natural combined_lab combined_all; do
    for arch in resnet50 efficientnetv2 vit; do
        run "$arch" "$subset"
    done
done

echo ""
echo "========================================================"
echo "  ALL COMBINED CLASSIFICATION RUNS COMPLETE"
echo "========================================================"
