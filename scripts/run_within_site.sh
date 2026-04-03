#!/usr/bin/env bash
# =============================================================================
# scripts/run_within_site.sh
# =============================================================================
# Launches all 12 within-site training runs:
#   4 subsets × 3 architectures = 12 runs
#
# Runs sequentially (one GPU, one job at a time).
# Each run saves its own checkpoint + logs under results/training/.
#
# Usage:
#   cd /mnt/storage/projects/DLCGIPG
#   source ja_scraper/venv/bin/activate
#   bash scripts/run_within_site.sh [--dry_run]
#
# Optional flags:
#   --dry_run     Print commands without executing them
#   --arch ARG    Run only a specific architecture (resnet50|efficientnetv2|vit)
#   --subset ARG  Run only a specific subset (ja_natural|ja_lab|be_natural|be_lab)
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

# ── hyperparameters (edit here to override for all runs) ──────────────────────
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

# ── helpers ───────────────────────────────────────────────────────────────────
ARCHS=("resnet50" "efficientnetv2" "vit")
SUBSETS=("ja_natural" "ja_lab" "be_natural" "be_lab")

mkdir -p "${LOG_DIR}"

run_train() {
    local arch="$1"
    local subset="$2"
    local run_id="${arch}__${subset}__within"
    local logfile="${LOG_DIR}/${run_id}.log"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  RUN: ${run_id}"
    echo "  Log: ${logfile}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local cmd=(
        /mnt/storage/projects/DLCGIPG/ja_scraper/venv/bin/python3 "${SRC}/train.py"
        --arch             "${arch}"
        --subset           "${subset}"
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
echo " DLCGIPG — Within-site training"
echo " $(date)"
echo " Dry run: ${DRY_RUN}"
echo "============================================================"

TOTAL=0
for arch in "${ARCHS[@]}"; do
    [[ -n "${FILTER_ARCH}"   && "${arch}"   != "${FILTER_ARCH}"   ]] && continue
    for subset in "${SUBSETS[@]}"; do
        [[ -n "${FILTER_SUBSET}" && "${subset}" != "${FILTER_SUBSET}" ]] && continue
        run_train "${arch}" "${subset}"
        (( TOTAL++ )) || true
    done
done

echo ""
echo "============================================================"
echo " All ${TOTAL} within-site runs complete."
echo " Results: ${RESULTS_DIR}/training/"
echo "============================================================"
