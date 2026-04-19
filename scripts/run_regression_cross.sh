#!/usr/bin/env bash
# =============================================================================
# scripts/run_regression_cross.sh
# =============================================================================
# Cross-domain regression runs — 4 experiments:
#   1. ja_natural → be_natural  (JA→BE natural)
#   2. ja_lab     → be_lab      (JA→BE lab)
#   3. be_natural → ja_natural  (BE→JA natural)
#   4. be_lab     → ja_lab      (BE→JA lab)
#
# Protocol mirrors classification cross-domain:
#   Train:  {source}_cross_train.csv
#   Val:    {source}_cross_val.csv
#   Test:   {target}_test.csv  (opposite site, held-out)
#   Metric: HuberLoss + log-MAE; also reports tier macro-F1 via price bucketing
#
# Hyperparams match run_cross_domain.sh:
#   float32 (no AMP), epochs=50, patience=10, LR=3e-4, batch_size=64
#
# Usage:
#   cd /mnt/storage/projects/DLCGIPG
#   bash scripts/run_regression_cross.sh [--dry_run]
#
# After all 4 runs, prints + saves comparison table to:
#   results/aggregated/regression_cross_comparison.csv
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/mnt/storage/projects/DLCGIPG/ja_scraper/venv/bin/python3"
SRC="${PROJECT_ROOT}/src"
DATA_DIR="${PROJECT_ROOT}/data"
IMAGE_DIR_JA="${PROJECT_ROOT}/ja_scraper/output/images"
IMAGE_DIR_BE="${PROJECT_ROOT}/be_scraper/output/images"
RESULTS_DIR="${PROJECT_ROOT}/results"
LOG_DIR="${RESULTS_DIR}/training/logs"

# ── Hyperparameters (matching classification cross-domain config) ─────────────
EPOCHS=50
BATCH_SIZE=64
BASE_LR=3e-4
BACKBONE_LR_SCALE=0.1
WEIGHT_DECAY=1e-4
DROPOUT=0.3
HUBER_DELTA=0.5
PATIENCE=10
NUM_WORKERS=8
SEED=42

DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry_run) DRY_RUN=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "${LOG_DIR}"

run_cross() {
    local src="$1"
    local tgt="$2"
    local run_id="efficientnetv2__${src}__${tgt}"
    local logfile="${LOG_DIR}/${run_id}.log"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  RUN: ${run_id}"
    echo "  Log: ${logfile}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local cmd=(
        "${PYTHON}" "${SRC}/train_regression_cross.py"
        --source_subset      "${src}"
        --target_subset      "${tgt}"
        --data_dir           "${DATA_DIR}"
        --image_dir_ja       "${IMAGE_DIR_JA}"
        --image_dir_be       "${IMAGE_DIR_BE}"
        --results_dir        "${RESULTS_DIR}"
        --epochs             "${EPOCHS}"
        --batch_size         "${BATCH_SIZE}"
        --base_lr            "${BASE_LR}"
        --backbone_lr_scale  "${BACKBONE_LR_SCALE}"
        --weight_decay       "${WEIGHT_DECAY}"
        --dropout            "${DROPOUT}"
        --huber_delta        "${HUBER_DELTA}"
        --patience           "${PATIENCE}"
        --num_workers        "${NUM_WORKERS}"
        --seed               "${SEED}"
    )

    if ${DRY_RUN}; then
        echo "[DRY RUN] ${cmd[*]}"
    else
        "${cmd[@]}" 2>&1 | tee "${logfile}"
        echo "  ✓ Completed: ${run_id}"
    fi
}

echo "============================================================"
echo " DLCGIPG — Cross-domain REGRESSION training"
echo " $(date)"
echo " Dry run: ${DRY_RUN}"
echo "============================================================"

run_cross "ja_natural" "be_natural"
run_cross "ja_lab"     "be_lab"
run_cross "be_natural" "ja_natural"
run_cross "be_lab"     "ja_lab"

echo ""
echo "============================================================"
echo " All 4 cross-domain regression runs complete."
echo " Building comparison table …"
echo "============================================================"

"${PYTHON}" "${PROJECT_ROOT}/scripts/make_regression_cross_table.py"

echo ""
echo "Done. Results in: ${RESULTS_DIR}/training/regression_cross/"
echo "Table:            ${RESULTS_DIR}/aggregated/regression_cross_comparison.csv"
