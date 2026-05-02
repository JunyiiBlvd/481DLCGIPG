"""
train_combined.py — Classification training on combined multi-site Stage 2 subsets.

Handles combined_natural, combined_lab, combined_all CSVs where each row
carries a source_subset column to route images to the correct site directory.

Usage:
    python src/train_combined.py \
        --subset combined_natural \
        --arch efficientnetv2

Outputs (results/training/{arch}__{subset}__within/):
  best_model.pth, train_log.json, final_metrics.json, classification_report.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT    = Path(__file__).resolve().parent.parent
SPLITS  = ROOT / "data" / "splits"
JA_IMGS = ROOT / "ja_scraper" / "output" / "images"
BE_IMGS = ROOT / "be_scraper" / "output" / "images"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import get_model, TIER_LABELS

TIER_TO_IDX = {t: i for i, t in enumerate(sorted(TIER_LABELS))}
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


class CombinedDiamondDataset(Dataset):
    def __init__(self, csv_path: Path, transform=None):
        self.transform = transform or val_test_transform
        df = pd.read_csv(csv_path, usecols=["diamond_id", "value_tier", "tier_label", "source_subset"],
                         low_memory=False)
        df["_img_dir"] = df["source_subset"].apply(
            lambda s: JA_IMGS if s.startswith("ja") else BE_IMGS
        )
        df["_img_path"] = df.apply(
            lambda r: r["_img_dir"] / str(r["value_tier"]) / f"{int(r['diamond_id'])}.jpg",
            axis=1,
        )
        missing = ~df["_img_path"].apply(lambda p: p.exists())
        if missing.sum():
            df = df[~missing].reset_index(drop=True)
        self.records = df[["tier_label", "_img_path"]].reset_index(drop=True)

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        try:
            img = Image.open(row["_img_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        return self.transform(img), int(row["tier_label"])


def build_loaders(subset: str, batch_size: int, num_workers: int):
    train_ds = CombinedDiamondDataset(SPLITS / f"{subset}_train.csv", train_transform)
    val_ds   = CombinedDiamondDataset(SPLITS / f"{subset}_val.csv",   val_test_transform)
    test_ds  = CombinedDiamondDataset(SPLITS / f"{subset}_test.csv",  val_test_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


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
    n = len(all_labels)
    return {
        "loss":     total_loss / n,
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset",      required=True,
                   choices=["combined_natural", "combined_lab", "combined_all"])
    p.add_argument("--arch",        required=True, choices=["resnet50", "efficientnetv2", "vit"])
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--base_lr",     type=float, default=3e-4)
    p.add_argument("--patience",    type=int,   default=10)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--num_workers", type=int,   default=8)
    args = p.parse_args()

    out_dir = ROOT / "results" / "training" / f"{args.arch}__{args.subset}__within"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Arch   : {args.arch}  Subset: {args.subset}")
    print(f"Device : {device}")

    train_loader, val_loader, test_loader = build_loaders(
        args.subset, args.batch_size, args.num_workers)
    print(f"Train={len(train_loader.dataset):,}  Val={len(val_loader.dataset):,}  "
          f"Test={len(test_loader.dataset):,}")

    model, param_groups = get_model(args.arch, dropout=args.dropout)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(param_groups, lr=args.base_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = GradScaler()

    best_val_f1, patience_ctr, train_log = -1.0, 0, []
    best_path = out_dir / "best_model.pth"

    print(f"\nTraining up to {args.epochs} epochs (patience={args.patience})...\n")
    for epoch in range(1, args.epochs + 1):
        t0      = time.time()
        train_m = run_epoch(model, train_loader, criterion, optimizer, scaler, device, True)
        val_m   = run_epoch(model, val_loader,   criterion, None,      None,   device, False)
        scheduler.step()
        train_log.append({"epoch": epoch, "train": train_m, "val": val_m})

        print(f"Epoch {epoch:3d} | "
              f"train f1={train_m['macro_f1']:.4f} acc={train_m['accuracy']:.4f} | "
              f"val f1={val_m['macro_f1']:.4f} acc={val_m['accuracy']:.4f} | "
              f"{time.time()-t0:.0f}s")

        if val_m["macro_f1"] > best_val_f1:
            best_val_f1, patience_ctr = val_m["macro_f1"], 0
            torch.save(model.state_dict(), best_path)
            print(f"          ↑ best val F1={best_val_f1:.4f} saved.")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    with open(out_dir / "train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)

    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_m = run_epoch(model, test_loader, criterion, None, None, device, False)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device, non_blocking=True))
            all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    report   = classification_report(all_labels, all_preds, target_names=sorted(TIER_LABELS), zero_division=0)
    per_class = f1_score(all_labels, all_preds, average=None, zero_division=0).tolist()

    print(f"Test accuracy={test_m['accuracy']:.4f}  macro_f1={test_m['macro_f1']:.4f}")

    final_metrics = {
        "arch": args.arch, "subset": args.subset,
        "best_val_f1": best_val_f1,
        "test_accuracy": test_m["accuracy"],
        "test_macro_f1": test_m["macro_f1"],
        "per_class_f1": {sorted(TIER_LABELS)[i]: per_class[i] for i in range(len(TIER_LABELS))},
        "hyperparams": {"epochs_run": len(train_log), "epochs_max": args.epochs,
                        "batch_size": args.batch_size, "base_lr": args.base_lr,
                        "dropout": args.dropout, "patience": args.patience},
    }
    with open(out_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
