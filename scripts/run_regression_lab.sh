#!/usr/bin/env bash
# run_regression_lab.sh — Train all 3 archs regression within-site for ja_lab and be_lab
# Also fills ResNet50 + ViT for ja_natural and be_natural (previously EfficientNetV2 only)

set -euo pipefail

PYTHON=/mnt/storage/projects/DLCGIPG/ja_scraper/venv/bin/python3
SCRIPT=/mnt/storage/projects/DLCGIPG/src/train_regression.py

DATA_DIR=/mnt/storage/projects/DLCGIPG/data
IMG_JA=/mnt/storage/projects/DLCGIPG/ja_scraper/output/images
IMG_BE=/mnt/storage/projects/DLCGIPG/be_scraper/output/images
RESULTS=/mnt/storage/projects/DLCGIPG/results

run() {
    local arch="$1" subset="$2"
    echo ""
    echo "========================================================"
    echo "  ${arch} / ${subset} / regression"
    echo "========================================================"
    $PYTHON "$SCRIPT" \
        --arch              "$arch" \
        --subset            "$subset" \
        --data_dir          "$DATA_DIR" \
        --image_dir_ja      "$IMG_JA" \
        --image_dir_be      "$IMG_BE" \
        --results_dir       "$RESULTS" \
        --epochs            30 \
        --batch_size        64 \
        --base_lr           3e-4 \
        --backbone_lr_scale 0.1 \
        --weight_decay      1e-4 \
        --dropout           0.3 \
        --huber_delta       0.5 \
        --patience          5 \
        --num_workers       8
    echo "  DONE: ${arch} / ${subset}"
}

# Fill missing archs for ja_natural and be_natural
run resnet50 ja_natural
run vit       ja_natural
run resnet50 be_natural
run vit       be_natural

# All 3 archs for ja_lab and be_lab
run resnet50      ja_lab
run efficientnetv2 ja_lab
run vit            ja_lab
run resnet50      be_lab
run efficientnetv2 be_lab
run vit            be_lab

echo ""
echo "========================================================"
echo "  ALL WITHIN-SITE REGRESSION RUNS COMPLETE"
echo "  Results: ${RESULTS}/training/regression/"
echo "========================================================"
