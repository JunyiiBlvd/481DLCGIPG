#!/usr/bin/env bash
# run_all_training.sh — Master training regiment for full pipeline evaluation
#
# Run order (each step depends on the previous completing):
#
#   STEP 1 — Stage 1 models (may already be running)
#   STEP 2 — Stage 2 regression for ja_lab + be_lab
#   STEP 3 — Stage 2 classification on combined subsets (9 runs)
#   STEP 4 — Stage 2 regression on combined subsets (3 runs)
#
# Already complete (do not re-run):
#   - Stage 2 classification within-site (12 models)
#   - Stage 2 regression within-site ja_natural + be_natural
#   - Stage 2 regression cross-domain (4 runs)
#
# Final model inventory after all steps:
#   Stage 1         :  3 models  (resnet50, efficientnetv2, vit)
#   Stage 2 clf     : 21 models  (3 archs × 4 within-site + 3 archs × 3 combined)
#   Stage 2 reg     : 21 models  (3 archs × 4 within-site + 3 archs × 3 combined)
#   Total Stage 2   : 42 configs
#   Pipeline combos : 3 × 42 = 126
#
# Usage:
#   bash scripts/run_all_training.sh
#   or run each step individually:
#   bash scripts/run_stage1_training.sh
#   bash scripts/run_regression_lab.sh
#   bash scripts/run_combined_classification.sh
#   bash scripts/run_combined_regression.sh

set -euo pipefail
cd /mnt/storage/projects/DLCGIPG

echo "========================================================"
echo "  DLCGIPG — Full Training Regiment"
echo "========================================================"

echo ""
echo "--- STEP 1: Stage 1 models (EfficientNetV2 + ViT) ---"
bash scripts/run_stage1_training.sh

echo ""
echo "--- STEP 2: Stage 2 regression — ja_lab + be_lab ---"
bash scripts/run_regression_lab.sh

echo ""
echo "--- STEP 3: Stage 2 classification — combined subsets ---"
bash scripts/run_combined_classification.sh

echo ""
echo "--- STEP 4: Stage 2 regression — combined subsets ---"
bash scripts/run_combined_regression.sh

echo ""
echo "========================================================"
echo "  ALL TRAINING COMPLETE"
echo "  Run scripts/eval_pipeline.py to evaluate."
echo "========================================================"
