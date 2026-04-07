#!/usr/bin/env bash
# run_regression.sh — Launch EfficientNetV2-S regression runs for ja_natural and be_natural
#
# Usage: bash scripts/run_regression.sh [--epochs N] [--batch_size N]
#
# Runs sequentially: ja_natural first, then be_natural.
# Logs to results/training/regression/efficientnetv2/{subset}/

set -euo pipefail

PYTHON=/mnt/storage/projects/DLCGIPG/ja_scraper/venv/bin/python3
SCRIPT=/mnt/storage/projects/DLCGIPG/src/train_regression.py

DATA_DIR=/mnt/storage/projects/DLCGIPG/data
IMG_JA=/mnt/storage/projects/DLCGIPG/ja_scraper/output/images
IMG_BE=/mnt/storage/projects/DLCGIPG/be_scraper/output/images
RESULTS=/mnt/storage/projects/DLCGIPG/results

EPOCHS=30
BATCH=64

# Parse optional overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)    EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

run() {
    local subset="$1"
    echo ""
    echo "========================================================"
    echo "  STARTING: efficientnetv2 / ${subset} / regression"
    echo "========================================================"
    $PYTHON "$SCRIPT" \
        --subset        "$subset" \
        --data_dir      "$DATA_DIR" \
        --image_dir_ja  "$IMG_JA" \
        --image_dir_be  "$IMG_BE" \
        --results_dir   "$RESULTS" \
        --epochs        "$EPOCHS" \
        --batch_size    "$BATCH" \
        --base_lr       3e-4 \
        --backbone_lr_scale 0.1 \
        --weight_decay  1e-4 \
        --dropout       0.3 \
        --huber_delta   0.5 \
        --patience      5 \
        --num_workers   8
    echo ""
    echo "  DONE: ${subset}"
}

run ja_natural
run be_natural

echo ""
echo "========================================================"
echo "  ALL REGRESSION RUNS COMPLETE"
echo "  Results: ${RESULTS}/training/regression/efficientnetv2/"
echo "========================================================"
