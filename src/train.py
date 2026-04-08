"""
train.py — Stage 2 training loop for DLCGIPG
=============================================

Trains ResNet50 / EfficientNetV2-S / ViT-B/16 on a single subset split.
Designed to be called once per (arch, subset) pair from a shell launcher.

Usage (single run):
    python src/train.py \\
        --arch resnet50 \\
        --subset ja_natural \\
        --data_dir /mnt/storage/projects/DLCGIPG/data \\
        --image_dir_ja /mnt/storage/projects/DLCGIPG/ja_scraper/output/images \\
        --image_dir_be /mnt/storage/projects/DLCGIPG/be_scraper/output/images \\
        --results_dir /mnt/storage/projects/DLCGIPG/results \\
        [--epochs 30] [--batch_size 64] [--base_lr 3e-4] [--patience 5]

Cross-domain (train on one site, evaluate on another):
    python src/train.py \\
        --arch resnet50 \\
        --subset ja_natural \\
        --cross_domain \\
        --data_dir ... \\
        ...

Outputs (all under results/training/{arch}/{subset}/):
  - best_model.pth          ← state dict of epoch with best val F1
  - train_log.json          ← per-epoch metrics (loss, acc, f1)
  - final_metrics.json      ← test-set evaluation + confusion matrix
  - classification_report.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ── project imports ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from models import get_model, count_params, TIER_LABELS
from diamond_dataset import get_dataloader  # existing src/diamond_dataset.py


# ── constants ────────────────────────────────────────────────────────────────
LABEL_TO_IDX = {label: i for i, label in enumerate(sorted(TIER_LABELS))}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
# alphabetical: budget=0, investment_grade=1, mid_range=2, premium=3

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── image dir resolver ───────────────────────────────────────────────────────

def resolve_image_dir(subset: str, image_dir_ja: str, image_dir_be: str) -> str:
    """Return the correct image root for a given subset name."""
    if subset.startswith("ja"):
        return image_dir_ja
    elif subset.startswith("be"):
        return image_dir_be
    else:
        raise ValueError(f"Cannot resolve image dir for subset '{subset}'")


# ── dataloader builder ───────────────────────────────────────────────────────

def build_loaders(args: argparse.Namespace) -> tuple:
    """
    Build train / val / test DataLoaders for a given subset.

    For within-site: uses {subset}_{train|val|test}.csv
    For cross-domain: uses {subset}_cross_{train|val}.csv for train/val,
                      and the *opposite* site's test split for final evaluation.

    Returns (train_loader, val_loader, test_loader, class_weights_tensor)
    """
    data_dir   = Path(args.data_dir)
    splits_dir = data_dir / "splits"

    # ── load class weights ────────────────────────────────────────────────────
    weights_path = splits_dir / "class_weights.json"
    with open(weights_path) as f:
        all_weights = json.load(f)

    # canonical key: e.g. "ja_natural_cross_train" or "ja_natural_train"
    if args.cross_domain:
        weight_key = f"{args.subset}_cross_train"
    else:
        weight_key = f"{args.subset}_train"

    weights_list = [all_weights[weight_key][str(i)] for i in range(4)]          # list of 4 floats
    class_weights = torch.tensor(weights_list, dtype=torch.float32)

    # ── resolve image directories ─────────────────────────────────────────────
    image_dir = resolve_image_dir(args.subset, args.image_dir_ja, args.image_dir_be)

    if args.cross_domain:
        # Cross-domain: train/val from current site, test from the opposite site
        train_csv = splits_dir / f"{args.subset}_cross_train.csv"
        val_csv   = splits_dir / f"{args.subset}_cross_val.csv"
        # Opposite site test set
        opposite  = args.subset.replace("ja_", "be_") if args.subset.startswith("ja") \
                    else args.subset.replace("be_", "ja_")
        test_csv  = splits_dir / f"{opposite}_test.csv"
        test_image_dir = resolve_image_dir(opposite, args.image_dir_ja, args.image_dir_be)
    else:
        train_csv      = splits_dir / f"{args.subset}_train.csv"
        val_csv        = splits_dir / f"{args.subset}_val.csv"
        test_csv       = splits_dir / f"{args.subset}_test.csv"
        test_image_dir = image_dir

    train_loader = get_dataloader(
        csv_path           = str(train_csv),
        image_dir          = image_dir,
        split              = "train",
        batch_size         = args.batch_size,
        class_weights_path = str(weights_path),
        num_workers        = args.num_workers,
        pin_memory         = True,
    )
    val_loader = get_dataloader(
        csv_path     = str(val_csv),
        image_dir    = image_dir,
        split        = "val",
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
        pin_memory   = True,
    )
    test_loader = get_dataloader(
        csv_path     = str(test_csv),
        image_dir    = test_image_dir,
        split        = "test",
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
        pin_memory   = True,
    )
    return train_loader, val_loader, test_loader, class_weights


# ── train / eval loops ────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler | None,
    device: torch.device,
    is_train: bool,
) -> dict:
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            images, labels = batch[0].to(device, non_blocking=True), \
                             batch[1].to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss   = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                logits = model(images)
                loss   = criterion(logits, labels)

            total_loss  += loss.item() * images.size(0)
            preds        = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    n       = len(all_labels)
    acc     = accuracy_score(all_labels, all_preds)
    macro_f1= f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {
        "loss":     total_loss / n,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
    }


def evaluate_test(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    label_names: list[str],
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            images, labels = batch[0].to(device, non_blocking=True), \
                             batch[1].to(device, non_blocking=True)
            with autocast("cuda"):
                logits = model(images)
                loss   = criterion(logits, labels)
            total_loss  += loss.item() * images.size(0)
            preds        = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    n        = len(all_labels)
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0).tolist()
    cm       = confusion_matrix(all_labels, all_preds).tolist()
    report   = classification_report(
        all_labels, all_preds, target_names=label_names, zero_division=0
    )
    return {
        "test_loss":      total_loss / n,
        "test_accuracy":  float(acc),
        "test_macro_f1":  float(macro_f1),
        "per_class_f1":   {label_names[i]: per_class_f1[i] for i in range(len(label_names))},
        "confusion_matrix": cm,
        "label_order":    label_names,
        "classification_report": report,
    }


# ── checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module, path: Path) -> None:
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> nn.Module:
    model.load_state_dict(torch.load(path, map_location=device))
    return model


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DLCGIPG Stage 2 training")
    # required
    p.add_argument("--arch",         required=True,
                   choices=["resnet50", "efficientnetv2", "vit"],
                   help="Model architecture")
    p.add_argument("--subset",       required=True,
                   choices=["ja_natural", "ja_lab", "be_natural", "be_lab"],
                   help="Dataset subset")
    p.add_argument("--data_dir",     required=True,
                   help="Root data directory (contains splits/)")
    p.add_argument("--image_dir_ja", required=True,
                   help="JA images root (contains budget/mid_range/premium/investment_grade/)")
    p.add_argument("--image_dir_be", required=True,
                   help="BE images root (same tier subfolder structure)")
    p.add_argument("--results_dir",  required=True,
                   help="Root results directory")
    # optional
    p.add_argument("--cross_domain", action="store_true",
                   help="Cross-domain mode: train on cross_train split, test on opposite site")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--base_lr",      type=float, default=3e-4,
                   help="Head learning rate; backbone gets base_lr * backbone_lr_scale")
    p.add_argument("--backbone_lr_scale", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--patience",     type=int,   default=5,
                   help="Early-stopping patience (epochs with no val F1 improvement)")
    p.add_argument("--num_workers",  type=int,   default=8)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--run_id_suffix",  type=str,   default="",
                   help="Optional suffix appended to run_id (e.g. '__seed1') for multi-seed runs")
    p.add_argument("--resume",       action="store_true",
                   help="Resume from best_model.pth if it exists in output dir")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("⚠  WARNING: No CUDA GPU detected — running on CPU. This will be very slow.")
    else:
        print(f"✓  GPU: {torch.cuda.get_device_name(0)}")

    # ── output directory ──────────────────────────────────────────────────────
    mode   = "cross" if args.cross_domain else "within"
    run_id = f"{args.arch}__{args.subset}__{mode}{args.run_id_suffix}"
    out_dir = Path(args.results_dir) / "training" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_model.pth"
    log_path  = out_dir / "train_log.json"
    print(f"→  Output dir: {out_dir}")

    # ── data ──────────────────────────────────────────────────────────────────
    print("Loading data splits …")
    train_loader, val_loader, test_loader, class_weights = build_loaders(args)
    class_weights = class_weights.to(device)
    label_names   = sorted(TIER_LABELS)   # alphabetical: budget, investment_grade, mid_range, premium
    print(f"   train={len(train_loader.dataset):,}  "
          f"val={len(val_loader.dataset):,}  "
          f"test={len(test_loader.dataset):,}")

    # ── model ─────────────────────────────────────────────────────────────────
    print(f"Building model: {args.arch} …")
    model, param_groups = get_model(
        arch             = args.arch,
        base_lr          = args.base_lr,
        backbone_lr_scale= args.backbone_lr_scale,
        dropout          = args.dropout,
    )
    model = model.to(device)
    stats = count_params(model)
    print(f"   Params: total={stats['total']:,}  trainable={stats['trainable']:,}")

    if args.resume and ckpt_path.exists():
        print(f"   Resuming from {ckpt_path}")
        model = load_checkpoint(model, ckpt_path, device)

    # ── loss / optimizer / scheduler ──────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = GradScaler("cuda")

    # ── training loop ─────────────────────────────────────────────────────────
    best_val_f1   = -1.0
    patience_ctr  = 0
    epoch_log     = []

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Train F1':>8}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'Val F1':>6}  {'Time':>6}")
    print("─" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = run_epoch(model, train_loader, criterion, optimizer, scaler, device, is_train=True)
        val_metrics   = run_epoch(model, val_loader,   criterion, None,      None,   device, is_train=False)
        scheduler.step()

        elapsed = time.time() - t0
        row = {
            "epoch":     epoch,
            "train":     train_metrics,
            "val":       val_metrics,
            "elapsed_s": round(elapsed, 1),
        }
        epoch_log.append(row)

        print(f"{epoch:>5}  {train_metrics['loss']:>10.4f}  "
              f"{train_metrics['accuracy']:>9.4f}  {train_metrics['macro_f1']:>8.4f}  "
              f"{val_metrics['loss']:>8.4f}  {val_metrics['accuracy']:>7.4f}  "
              f"{val_metrics['macro_f1']:>6.4f}  {elapsed:>5.0f}s")

        # ── checkpoint ─────────────────────────────────────────────────────────
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1  = val_metrics["macro_f1"]
            patience_ctr = 0
            save_checkpoint(model, ckpt_path)
            print(f"         ✓ New best val F1={best_val_f1:.4f} — checkpoint saved")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\nEarly stopping after {epoch} epochs (no val F1 improvement for {args.patience} epochs).")
                break

        # ── flush log every epoch ──────────────────────────────────────────────
        with open(log_path, "w") as f:
            json.dump(epoch_log, f, indent=2)

    # ── test evaluation ───────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint for test evaluation …")
    model = load_checkpoint(model, ckpt_path, device)
    test_results = evaluate_test(model, test_loader, criterion, device, label_names)

    print(f"\n{'─'*60}")
    print(f"TEST RESULTS  [{run_id}]")
    print(f"  Accuracy : {test_results['test_accuracy']:.4f}")
    print(f"  Macro F1 : {test_results['test_macro_f1']:.4f}")
    print(f"  Per-class F1:")
    for cls, f1 in test_results["per_class_f1"].items():
        print(f"    {cls:<18s}: {f1:.4f}")
    print(f"{'─'*60}\n")
    print(test_results["classification_report"])

    # ── save final results ────────────────────────────────────────────────────
    final_path  = out_dir / "final_metrics.json"
    report_path = out_dir / "classification_report.txt"

    report_text = test_results.pop("classification_report")
    final_meta  = {
        "arch":        args.arch,
        "subset":      args.subset,
        "cross_domain": args.cross_domain,
        "best_val_f1": best_val_f1,
        "hyperparams": {
            "epochs_run":         len(epoch_log),
            "epochs_max":         args.epochs,
            "batch_size":         args.batch_size,
            "base_lr":            args.base_lr,
            "backbone_lr_scale":  args.backbone_lr_scale,
            "weight_decay":       args.weight_decay,
            "dropout":            args.dropout,
            "patience":           args.patience,
        },
        **test_results,
    }
    with open(final_path, "w") as f:
        json.dump(final_meta, f, indent=2)
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"Saved:\n  {final_path}\n  {report_path}\n  {log_path}\n  {ckpt_path}")


if __name__ == "__main__":
    main()
