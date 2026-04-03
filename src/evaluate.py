"""
evaluate.py — Standalone test-set evaluator for DLCGIPG Stage 2
================================================================

Loads a saved checkpoint and runs test-set evaluation independently of training.
Useful for re-evaluating a checkpoint with a different test split, or for
running cross-domain evaluation of an already-trained model.

Usage:
    python src/evaluate.py \\
        --arch resnet50 \\
        --checkpoint results/training/resnet50__ja_natural__within/best_model.pth \\
        --test_csv   data/splits/be_natural_test.csv \\
        --image_dir  /mnt/storage/projects/DLCGIPG/be_scraper/output/images \\
        --label      be_natural \\
        --out_dir    results/eval/resnet50__ja_to_be_natural

Outputs (all under --out_dir):
  - final_metrics.json
  - classification_report.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from models import get_model, TIER_LABELS
from diamond_dataset import get_dataloader


LABEL_NAMES = sorted(TIER_LABELS)  # alphabetical


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DLCGIPG Stage 2 — standalone evaluator")
    p.add_argument("--arch",       required=True, choices=["resnet50", "efficientnetv2", "vit"])
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    p.add_argument("--test_csv",   required=True, help="Path to test split CSV")
    p.add_argument("--image_dir",  required=True, help="Image root for this test split")
    p.add_argument("--label",      required=True, help="Human-readable label for this run, e.g. 'ja_to_be_natural'")
    p.add_argument("--out_dir",    required=True, help="Directory to save evaluation outputs")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers",type=int, default=8)
    p.add_argument("--dropout",    type=float, default=0.3,
                   help="Must match the dropout used during training")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── model ─────────────────────────────────────────────────────────────────
    model, _ = get_model(arch=args.arch, dropout=args.dropout)
    state     = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model     = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── data ──────────────────────────────────────────────────────────────────
    loader = get_dataloader(
        csv_path     = args.test_csv,
        image_dir    = args.image_dir,
        batch_size   = args.batch_size,
        split        = "test",
        class_weights= None,
        num_workers  = args.num_workers,
        pin_memory   = True,
    )
    print(f"Test set: {len(loader.dataset):,} samples")

    # ── inference ─────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.no_grad():
        for batch in loader:
            images, labels = batch[0].to(device, non_blocking=True), \
                             batch[1].to(device, non_blocking=True)
            with autocast("cuda"):
                logits = model(images)
                loss   = criterion(logits, labels)
            total_loss  += loss.item() * images.size(0)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    n         = len(all_labels)
    acc       = accuracy_score(all_labels, all_preds)
    macro_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class = f1_score(all_labels, all_preds, average=None,    zero_division=0).tolist()
    cm        = confusion_matrix(all_labels, all_preds).tolist()
    report    = classification_report(all_labels, all_preds, target_names=LABEL_NAMES, zero_division=0)

    print(f"\nAccuracy : {acc:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    for i, cls in enumerate(LABEL_NAMES):
        print(f"  {cls:<18s}: {per_class[i]:.4f}")
    print(f"\n{report}")

    # ── save ──────────────────────────────────────────────────────────────────
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics = {
        "label":           args.label,
        "arch":            args.arch,
        "checkpoint":      args.checkpoint,
        "test_csv":        args.test_csv,
        "test_accuracy":   float(acc),
        "test_macro_f1":   float(macro_f1),
        "test_loss":       total_loss / n,
        "per_class_f1":    {LABEL_NAMES[i]: per_class[i] for i in range(len(LABEL_NAMES))},
        "confusion_matrix": cm,
        "label_order":     LABEL_NAMES,
    }
    with open(out / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out / "classification_report.txt", "w") as f:
        f.write(report)

    print(f"Results saved to {out}/")


if __name__ == "__main__":
    main()
