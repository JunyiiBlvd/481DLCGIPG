"""
eval_regression.py — Standalone test-set evaluation for regression checkpoints.

Loads best_model.pth for ja_natural and be_natural, runs test split,
reports all metrics (existing + R² and Spearman), and updates final_metrics.json
in-place (adds fields, preserves existing).

No retraining. GPU used if available.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

# ── project src on path ────────────────────────────────────────────────────────
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
SRC_DIR    = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from train_regression import (
    build_dataloader,
    build_regression_model,
    evaluate_test,
    load_checkpoint,
)
import torch.nn as nn

PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_ROOT / "data" / "splits"
CKPT_BASE    = PROJECT_ROOT / "results" / "training" / "regression" / "efficientnetv2"

IMAGE_DIRS = {
    "ja_natural": PROJECT_ROOT / "ja_scraper"  / "output" / "images",
    "be_natural": PROJECT_ROOT / "be_scraper"  / "output" / "images",
}

HUBER_DELTA = 0.5
DROPOUT     = 0.3
BATCH_SIZE  = 128
NUM_WORKERS = 8


def eval_subset(subset: str, device: torch.device) -> dict:
    out_dir   = CKPT_BASE / subset
    ckpt_path = out_dir / "best_model.pth"
    metrics_path = out_dir / "final_metrics.json"

    print(f"\n{'─'*55}")
    print(f"  {subset}")
    print(f"{'─'*55}")

    test_loader = build_dataloader(
        csv_path   = DATA_DIR / f"{subset}_test.csv",
        image_dir  = IMAGE_DIRS[subset],
        split      = "test",
        batch_size = BATCH_SIZE,
        num_workers= NUM_WORKERS,
    )
    print(f"  Test set: {len(test_loader.dataset):,} images")

    model, _ = build_regression_model(
        base_lr           = 3e-4,
        backbone_lr_scale = 0.1,
        dropout           = DROPOUT,
    )
    model = load_checkpoint(model, ckpt_path, device).to(device)

    criterion = torch.nn.HuberLoss(delta=HUBER_DELTA)
    test_m    = evaluate_test(model, test_loader, criterion, device)

    print(f"  Log-MAE     : {test_m['test_log_mae']:.4f}")
    print(f"  Log-RMSE    : {test_m['test_log_rmse']:.4f}")
    print(f"  USD MAE     : ${test_m['test_usd_mae']:>10,.0f}")
    print(f"  USD RMSE    : ${test_m['test_usd_rmse']:>10,.0f}")
    print(f"  Median APE  : {test_m['test_usd_med_ape']:.2f}%")
    print(f"  R²  (log)   : {test_m['test_r2']:.4f}")
    print(f"  Spearman(log): {test_m['test_spearman']:.4f}")

    # update final_metrics.json in-place
    with open(metrics_path) as f:
        existing = json.load(f)
    existing.update(test_m)   # adds/overwrites test_ fields
    with open(metrics_path, "w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")
    print(f"  Updated: {metrics_path}")

    return {subset: test_m}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_results = {}
    for subset in ["ja_natural", "be_natural"]:
        all_results.update(eval_subset(subset, device))

    # summary table
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    metrics = [
        ("test_log_mae",     "log-MAE"),
        ("test_log_rmse",    "log-RMSE"),
        ("test_usd_mae",     "USD-MAE"),
        ("test_usd_rmse",    "USD-RMSE"),
        ("test_usd_med_ape", "MedAPE%"),
        ("test_r2",          "R² (log)"),
        ("test_spearman",    "Spearman"),
    ]
    print(f"{'Metric':<14} {'ja_natural':>14} {'be_natural':>14}")
    print(f"{'-'*14} {'-'*14} {'-'*14}")
    for key, label in metrics:
        ja = all_results["ja_natural"][key]
        be = all_results["be_natural"][key]
        print(f"{label:<14} {ja:>14.4f} {be:>14.4f}")


if __name__ == "__main__":
    main()
