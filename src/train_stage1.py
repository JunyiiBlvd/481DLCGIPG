"""
train_stage1.py — Stage 1 trainer: 68-class gemstone species classifier

Trains EfficientNetV2-S or ViT-B/16 on the Kaggle Combined-P1-Dataset.
Dataset is in ImageFolder format (one subfolder per class).

Usage:
    python src/train_stage1.py --arch efficientnetv2
    python src/train_stage1.py --arch vit

Outputs (under results/training/stage1/{arch}/):
  - best_model.pth
  - train_log.json
  - final_metrics.json
  - classification_report.txt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    EfficientNet_V2_S_Weights, ResNet50_Weights, ViT_B_16_Weights,
    efficientnet_v2_s, vit_b_16,
)
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data" / "Combined-P1-Dataset"
RESULTS   = ROOT / "results" / "training" / "stage1"

NUM_CLASSES = 68
DROPOUT     = 0.3

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


# ── Model builders ────────────────────────────────────────────────────────────

def build_model(arch: str) -> tuple[nn.Module, list]:
    if arch == "efficientnetv2":
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features, NUM_CLASSES),
        )
        head_params    = list(model.classifier.parameters())
        backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]

    elif arch == "vit":
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features, NUM_CLASSES),
        )
        head_params     = list(model.heads.parameters())
        backbone_params = [p for n, p in model.named_parameters() if "heads" not in n]

    else:
        raise ValueError(f"Unsupported arch: {arch}")

    param_groups = [
        {"params": backbone_params, "lr": 3e-4 * 0.1},
        {"params": head_params,     "lr": 3e-4},
    ]
    return model, param_groups


# ── Data ──────────────────────────────────────────────────────────────────────

def build_loaders(batch_size: int, num_workers: int):
    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_ds   = datasets.ImageFolder(DATA_DIR / "valid", transform=val_test_transform)
    test_ds  = datasets.ImageFolder(DATA_DIR / "test",  transform=val_test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, train_ds.classes


# ── Train / eval loops ────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler, device, is_train):
    model.train() if is_train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad()
                with autocast("cuda"):
                    logits = model(images)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with autocast("cuda"):
                    logits = model(images)
                    loss   = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    n        = len(all_labels)
    acc      = float(accuracy_score(all_labels, all_preds))
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    return {"loss": total_loss / n, "accuracy": acc, "macro_f1": macro_f1}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch",        required=True, choices=["efficientnetv2", "vit"])
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--patience",    type=int, default=10)
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    out_dir = RESULTS / args.arch
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Arch   : {args.arch}")
    print(f"Device : {device}")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"GPU    : {torch.cuda.get_device_name(0)}  {free/1024**3:.1f}/{total/1024**3:.1f} GiB free")

    print("\nBuilding dataloaders...")
    train_loader, val_loader, test_loader, classes = build_loaders(args.batch_size, args.num_workers)
    print(f"  train={len(train_loader.dataset):,}  val={len(val_loader.dataset):,}  test={len(test_loader.dataset):,}  classes={len(classes)}")

    print("Building model...")
    model, param_groups = build_model(args.arch)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(param_groups, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = GradScaler()

    best_val_f1  = -1.0
    patience_ctr = 0
    train_log    = []
    best_path    = out_dir / "best_model.pth"

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...\n")
    for epoch in range(1, args.epochs + 1):
        t0     = time.time()
        train_m = run_epoch(model, train_loader, criterion, optimizer, scaler, device, is_train=True)
        val_m   = run_epoch(model, val_loader,   criterion, None,      None,   device, is_train=False)
        scheduler.step()

        train_log.append({"epoch": epoch, "train": train_m, "val": val_m})

        print(f"Epoch {epoch:3d} | "
              f"train loss={train_m['loss']:.4f} acc={train_m['accuracy']:.4f} f1={train_m['macro_f1']:.4f} | "
              f"val loss={val_m['loss']:.4f} acc={val_m['accuracy']:.4f} f1={val_m['macro_f1']:.4f} | "
              f"{time.time()-t0:.0f}s")

        if val_m["macro_f1"] > best_val_f1:
            best_val_f1  = val_m["macro_f1"]
            patience_ctr = 0
            torch.save(model.state_dict(), best_path)
            print(f"          ↑ new best val F1={best_val_f1:.4f}  saved.")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

    # ── Save train log ────────────────────────────────────────────────────────
    with open(out_dir / "train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_m = run_epoch(model, test_loader, criterion, None, None, device, is_train=False)

    # Full classification report
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    report = classification_report(all_labels, all_preds, target_names=classes, zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0).tolist()

    print(f"\nTest accuracy : {test_m['accuracy']:.4f}")
    print(f"Test macro F1 : {test_m['macro_f1']:.4f}")

    final_metrics = {
        "arch":          args.arch,
        "best_val_f1":   best_val_f1,
        "test_loss":     test_m["loss"],
        "test_accuracy": test_m["accuracy"],
        "test_macro_f1": test_m["macro_f1"],
        "per_class_f1":  {classes[i]: per_class_f1[i] for i in range(len(classes))},
        "classes":       classes,
        "hyperparams": {
            "epochs_run":   len(train_log),
            "epochs_max":   args.epochs,
            "batch_size":   args.batch_size,
            "base_lr":      3e-4,
            "backbone_lr_scale": 0.1,
            "weight_decay": 1e-4,
            "dropout":      DROPOUT,
            "patience":     args.patience,
        },
    }

    with open(out_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)

    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
