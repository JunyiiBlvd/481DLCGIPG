#!/usr/bin/env bash
# run_combined_regression.sh — Train all 3 archs regression on combined subsets (9 runs)

set -euo pipefail

PYTHON=/mnt/storage/projects/DLCGIPG/ja_scraper/venv/bin/python
SCRIPT=/mnt/storage/projects/DLCGIPG/src/train_regression_combined.py

cd /mnt/storage/projects/DLCGIPG

for subset in combined_natural combined_lab combined_all; do
    for arch in resnet50 efficientnetv2 vit; do
        echo ""
        echo "========================================================"
        echo "  regression / ${arch} / ${subset}"
        echo "========================================================"
        $PYTHON $SCRIPT --arch "$arch" --subset "$subset"
    done
done

echo ""
echo "========================================================"
echo "  ALL COMBINED REGRESSION RUNS COMPLETE"
echo "========================================================"
