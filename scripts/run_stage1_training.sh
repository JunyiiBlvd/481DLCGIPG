#!/usr/bin/env bash
# run_stage1_training.sh — Train EfficientNetV2-S and ViT-B/16 on Stage 1 dataset

set -euo pipefail

PYTHON="/mnt/storage/projects/DLCGIPG/ja_scraper/venv/bin/python"
SCRIPT="src/train_stage1.py"

cd /mnt/storage/projects/DLCGIPG

echo "============================================================"
echo " Stage 1 Training — EfficientNetV2-S"
echo "============================================================"
$PYTHON $SCRIPT --arch efficientnetv2 2>&1 | tee results/training/stage1/efficientnetv2_train.log

echo ""
echo "============================================================"
echo " Stage 1 Training — ViT-B/16"
echo "============================================================"
$PYTHON $SCRIPT --arch vit 2>&1 | tee results/training/stage1/vit_train.log

echo ""
echo "Both Stage 1 models trained. Checkpoints in results/training/stage1/"
