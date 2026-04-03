#!/usr/bin/env bash
# =============================================================================
# scripts/run_cross_domain.sh
# =============================================================================
# Launches all 12 cross-domain training runs:
#   2 directions (JA→BE, BE→JA) × 2 origins (natural, lab) × 3 archs = 12 runs
#
# Cross-domain mode: --cross_domain flag in train.py
#   Train on: {subset}_cross_train.csv  (90% of within-site data, excl. test)
#   Val on:   {subset}_cross_val.csv
#   Test on:  {opposite}_test.csv       (the OTHER site's held-out test)
#
# Run AFTER run_within_site.sh is complete. This script is independent
# (does not require within-site checkpoints — trains from scratch each time).
#
# Usage:
#   cd /mnt/storage/projects/DLCGIPG
#   source ja_scraper/venv/bin/activate
#   bash scripts/run_cross_domain.sh [--dry_run] [--arch ARG] [--subset ARG]
# =============================================================================

set -euo pipefail

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${PROJECT_ROOT}/src"
DATA_DIR="${PROJECT_ROOT}/data"
IMAGE_DIR_JA="${PROJECT_ROOT}/ja_scraper/output/images"
IMAGE_DIR_BE="${PROJECT_ROOT}/be_scraper/output/images"
RESULTS_DIR="${PROJECT_ROOT}/results"
LOG_DIR="${RESULTS_DIR}/training/logs"

# ── hyperparameters ───────────────────────────────────────────────────────────
EPOCHS=30
BATCH_SIZE=64
BASE_LR=3e-4
BACKBONE_LR_SCALE=0.1
WEIGHT_DECAY=1e-4
DROPOUT=0.3
PATIENCE=5
NUM_WORKERS=8

# ── parse args ────────────────────────────────────────────────────────────────
DRY_RUN=false
FILTER_ARCH=""
FILTER_SUBSET=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry_run)  DRY_RUN=true;     shift ;;
        --arch)     FILTER_ARCH="$2"; shift 2 ;;
        --subset)   FILTER_SUBSET="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

ARCHS=("resnet50" "efficientnetv2" "vit")
# training site subsets; test site is derived in train.py (ja→be, be→ja)
SUBSETS=("ja_natural" "ja_lab" "be_natural" "be_lab")

mkdir -p "${LOG_DIR}"

run_cross() {
    local arch="$1"
    local subset="$2"
    local run_id="${arch}__${subset}__cross"
    local logfile="${LOG_DIR}/${run_id}.log"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  RUN: ${run_id}"
    echo "  Log: ${logfile}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local cmd=(
        python "${SRC}/train.py"
        --arch             "${arch}"
        --subset           "${subset}"
        --cross_domain
        --data_dir         "${DATA_DIR}"
        --image_dir_ja     "${IMAGE_DIR_JA}"
        --image_dir_be     "${IMAGE_DIR_BE}"
        --results_dir      "${RESULTS_DIR}"
        --epochs           "${EPOCHS}"
        --batch_size       "${BATCH_SIZE}"
        --base_lr          "${BASE_LR}"
        --backbone_lr_scale "${BACKBONE_LR_SCALE}"
        --weight_decay     "${WEIGHT_DECAY}"
        --dropout          "${DROPOUT}"
        --patience         "${PATIENCE}"
        --num_workers      "${NUM_WORKERS}"
    )

    if ${DRY_RUN}; then
        echo "[DRY RUN] ${cmd[*]}"
    else
        "${cmd[@]}" 2>&1 | tee "${logfile}"
        echo "  ✓ Completed: ${run_id}"
    fi
}

# ── main loop ─────────────────────────────────────────────────────────────────
echo "============================================================"
echo " DLCGIPG — Cross-domain training"
echo " $(date)"
echo " Dry run: ${DRY_RUN}"
echo "============================================================"

TOTAL=0
for arch in "${ARCHS[@]}"; do
    [[ -n "${FILTER_ARCH}"   && "${arch}"   != "${FILTER_ARCH}"   ]] && continue
    for subset in "${SUBSETS[@]}"; do
        [[ -n "${FILTER_SUBSET}" && "${subset}" != "${FILTER_SUBSET}" ]] && continue
        run_cross "${arch}" "${subset}"
        (( TOTAL++ )) || true
    done
done

echo ""
echo "============================================================"
echo " All ${TOTAL} cross-domain runs complete."
echo " Results: ${RESULTS_DIR}/training/"
echo "============================================================"
