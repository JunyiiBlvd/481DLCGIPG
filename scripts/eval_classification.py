"""
eval_classification.py — Standalone test-set evaluation for classification checkpoints
that completed training but segfaulted before writing final_metrics.json.

Usage:
    python3 scripts/eval_classification.py \
        --run_dir results/training/efficientnetv2__ja_natural__within__seed1 \
        --arch efficientnetv2 \
        --subset ja_natural

Writes final_metrics.json and classification_report.txt to run_dir,
matching the exact format produced by train.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT    = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from diamond_dataset import get_dataloader
from models import get_model, TIER_LABELS


def build_test_loader(subset: str, batch_size: int = 64, num_workers: int = 8):
    splits_dir = ROOT / "data" / "splits"
    image_dir  = (
        ROOT / "ja_scraper" / "output" / "images"
        if subset.startswith("ja")
        else ROOT / "be_scraper" / "output" / "images"
    )
    return get_dataloader(
        csv_path    = str(splits_dir / f"{subset}_test.csv"),
        image_dir   = str(image_dir),
        split       = "test",
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = True,
    )


def run_eval(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)
            logits = model(images)
            loss   = criterion(logits, labels)
            total_loss  += loss.item() * images.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    n            = len(all_labels)
    acc          = float(accuracy_score(all_labels, all_preds))
    macro_f1     = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    per_class    = f1_score(all_labels, all_preds, average=None, zero_division=0).tolist()
    cm           = confusion_matrix(all_labels, all_preds).tolist()
    report       = classification_report(all_labels, all_preds,
                                         target_names=TIER_LABELS, zero_division=0)
    return {
        "test_loss":        total_loss / n,
        "test_accuracy":    acc,
        "test_macro_f1":    macro_f1,
        "per_class_f1":     {TIER_LABELS[i]: per_class[i] for i in range(len(TIER_LABELS))},
        "confusion_matrix": cm,
        "label_order":      TIER_LABELS,
        "classification_report": report,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="Path to the run directory")
    p.add_argument("--arch",    required=True, choices=["resnet50", "efficientnetv2", "vit"])
    p.add_argument("--subset",  required=True)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    run_dir   = Path(args.run_dir)
    ckpt_path = run_dir / "best_model.pth"
    log_path  = run_dir / "train_log.json"

    assert ckpt_path.exists(), f"No checkpoint at {ckpt_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    if device.type == "cuda":
        free  = torch.cuda.mem_get_info()[0] / 1024**3
        total = torch.cuda.mem_get_info()[1] / 1024**3
        print(f"GPU mem : {free:.1f} GiB free / {total:.1f} GiB total")
    print(f"Run dir : {run_dir}")
    print(f"Arch    : {args.arch}  Subset: {args.subset}")

    # ── derive training metadata from log ─────────────────────────────────────
    with open(log_path) as f:
        train_log = json.load(f)
    best_val_f1 = max(e["val"]["macro_f1"] for e in train_log)
    best_epoch  = max(train_log, key=lambda e: e["val"]["macro_f1"])["epoch"]
    epochs_run  = len(train_log)
    print(f"Log     : {epochs_run} epochs recorded  best_epoch={best_epoch}  best_val_F1={best_val_f1:.4f}")
    print("Note    : epochs_run reflects entries in train_log.json (may be truncated by segfault)\n")

    # ── model ──────────────────────────────────────────────────────────────────
    model, _ = get_model(args.arch, dropout=0.3)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)

    # ── data ───────────────────────────────────────────────────────────────────
    test_loader = build_test_loader(args.subset, args.batch_size, args.num_workers)
    print(f"Test set: {len(test_loader.dataset):,} images")

    # ── criterion (same as train.py — CrossEntropy without class weights) ──────
    criterion = nn.CrossEntropyLoss()

    # ── evaluate ───────────────────────────────────────────────────────────────
    print("Running inference...")
    results = run_eval(model, test_loader, criterion, device)

    print(f"\n{'─'*55}")
    print(f"TEST RESULTS  [{run_dir.name}]")
    print(f"  Accuracy : {results['test_accuracy']:.4f}")
    print(f"  Macro F1 : {results['test_macro_f1']:.4f}")
    print("  Per-class F1:")
    for cls, f1 in results["per_class_f1"].items():
        print(f"    {cls:<20s}: {f1:.4f}")
    print(f"{'─'*55}")
    print(results["classification_report"])

    # ── write outputs ──────────────────────────────────────────────────────────
    report_text = results.pop("classification_report")

    final_meta = {
        "arch":         args.arch,
        "subset":       args.subset,
        "cross_domain": False,
        "best_val_f1":  best_val_f1,
        "hyperparams": {
            "epochs_run":        epochs_run,
            "epochs_max":        30,
            "batch_size":        args.batch_size,
            "base_lr":           0.0003,
            "backbone_lr_scale": 0.1,
            "weight_decay":      0.0001,
            "dropout":           0.3,
            "patience":          5,
        },
        **results,
    }

    final_path  = run_dir / "final_metrics.json"
    report_path = run_dir / "classification_report.txt"
    with open(final_path, "w") as f:
        json.dump(final_meta, f, indent=2)
        f.write("\n")
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"Saved: {final_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
