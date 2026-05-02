#!/bin/bash
set -e

VENV=/mnt/storage/projects/DLCGIPG/ja_scraper/venv/bin/python3
PROJ=/mnt/storage/projects/DLCGIPG

echo "=== Starting combined runs ==="

# Regression: efficientnetv2 only (clear winner from single-site runs)
# combined_natural: DONE (MAE=0.2917, F1=0.7115)
# combined_lab:     DONE (MAE=0.3746, F1=0.6322)
$VENV $PROJ/src/train_regression_combined.py --arch efficientnetv2 --subset combined_all

# Classification: all architectures on combined_all to confirm efficientnetv2 wins
$VENV $PROJ/src/train_combined.py --arch efficientnetv2 --subset combined_natural
$VENV $PROJ/src/train_combined.py --arch efficientnetv2 --subset combined_lab
$VENV $PROJ/src/train_combined.py --arch efficientnetv2 --subset combined_all
$VENV $PROJ/src/train_combined.py --arch resnet50 --subset combined_all
$VENV $PROJ/src/train_combined.py --arch vit --subset combined_all

echo "=== All combined runs complete ==="
