"""
train_regression_combined.py — Regression training on combined multi-site Stage 2 subsets.

Uses the normalized_log_price column (z-scored per source subset) as the
regression target, so prices are comparable across sites and diamond types.

Usage:
    python src/train_regression_combined.py --subset combined_natural
    python src/train_regression_combined.py --subset combined_lab
    python src/train_regression_combined.py --subset combined_all

Outputs (results/training/regression/efficientnetv2/{subset}/):
  best_model.pth, train_log.json, final_metrics.json
"""

from __future__ import annotations

import argparse
import json
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
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
from sklearn.metrics import f1_score

ROOT    = Path(__file__).resolve().parent.parent
SPLITS  = ROOT / "data" / "splits"
JA_IMGS = ROOT / "ja_scraper" / "output" / "images"
BE_IMGS = ROOT / "be_scraper" / "output" / "images"

TIER_LABELS = sorted(["budget", "investment_grade", "mid_range", "premium"])

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


class CombinedRegressionDataset(Dataset):
    def __init__(self, csv_path: Path, transform=None):
        self.transform = transform or val_test_transform
        df = pd.read_csv(csv_path,
                         usecols=["diamond_id", "value_tier", "tier_label",
                                  "source_subset", "normalized_log_price"],
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
        self.records = df[["normalized_log_price", "tier_label", "_img_path"]].reset_index(drop=True)

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        try:
            img = Image.open(row["_img_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        return self.transform(img), torch.tensor(row["normalized_log_price"], dtype=torch.float32)


def compute_tier_thresholds(csv_path: Path) -> list[float]:
    """Compute tier boundary thresholds from normalized_log_price in train split."""
    df = pd.read_csv(csv_path, usecols=["value_tier", "normalized_log_price"], low_memory=False)
    tier_means = df.groupby("value_tier")["normalized_log_price"].mean().sort_values()
    means = tier_means.values.tolist()
    return [(means[i] + means[i+1]) / 2 for i in range(len(means) - 1)]


def predictions_to_tiers(preds: np.ndarray, thresholds: list[float]) -> np.ndarray:
    tier_idx = np.zeros(len(preds), dtype=np.int64)
    for t in thresholds:
        tier_idx += (preds > t).astype(np.int64)
    return tier_idx


def build_model(arch: str, dropout: float) -> nn.Module:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from models import get_model
    model, _ = get_model(arch, dropout=dropout)
    if arch == "resnet50":
        model.fc[3] = nn.Linear(model.fc[3].in_features, 1)
    elif arch == "efficientnetv2":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    elif arch == "vit":
        model.heads.head[1] = nn.Linear(model.heads.head[1].in_features, 1)
    return model


def run_epoch(model, loader, criterion, optimizer, scaler, device, is_train):
    model.train() if is_train else model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, targets in loader:
            images  = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad()
                preds = model(images).squeeze(1)
                loss  = criterion(preds, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(images).squeeze(1)
                loss  = criterion(preds, targets)
            total_loss += loss.item() * images.size(0)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
    n = len(all_targets)
    mae = float(np.mean(np.abs(np.array(all_preds) - np.array(all_targets))))
    return {"loss": total_loss / n, "mae": mae}, all_preds, all_targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch",        required=True, choices=["resnet50", "efficientnetv2", "vit"])
    p.add_argument("--subset",      required=True,
                   choices=["combined_natural", "combined_lab", "combined_all"])
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--patience",    type=int,   default=5)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--num_workers", type=int,   default=8)
    args = p.parse_args()

    out_dir = ROOT / "results" / "training" / "regression" / args.arch / args.subset
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Subset : {args.subset}")
    print(f"Device : {device}")

    train_csv = SPLITS / f"{args.subset}_train.csv"
    val_csv   = SPLITS / f"{args.subset}_val.csv"
    test_csv  = SPLITS / f"{args.subset}_test.csv"

    thresholds = compute_tier_thresholds(train_csv)
    print(f"Tier thresholds (normalized log-price): {[f'{t:.4f}' for t in thresholds]}")

    def make_loader(csv, tfm, shuffle):
        ds = CombinedRegressionDataset(csv, tfm)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=shuffle)

    train_loader = make_loader(train_csv, train_transform,    True)
    val_loader   = make_loader(val_csv,   val_test_transform, False)
    test_loader  = make_loader(test_csv,  val_test_transform, False)
    print(f"Train={len(train_loader.dataset):,}  Val={len(val_loader.dataset):,}  "
          f"Test={len(test_loader.dataset):,}")

    model = build_model(args.arch, args.dropout).to(device)
    criterion = nn.HuberLoss(delta=0.5)
    head_key        = {"resnet50": "fc", "efficientnetv2": "classifier", "vit": "heads"}[args.arch]
    backbone_params = [p for n, p in model.named_parameters() if head_key not in n]
    head_params     = [p for n, p in model.named_parameters() if head_key in n]
    optimizer = AdamW([{"params": backbone_params, "lr": 3e-5},
                       {"params": head_params,     "lr": 3e-4}], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = GradScaler()

    best_val_mae, patience_ctr, train_log = float("inf"), 0, []
    best_path = out_dir / "best_model.pth"

    print(f"\nTraining up to {args.epochs} epochs (patience={args.patience})...\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m, _, _ = run_epoch(model, train_loader, criterion, optimizer, scaler, device, True)
        val_m,   _, _ = run_epoch(model, val_loader,   criterion, None,      None,   device, False)
        scheduler.step()
        train_log.append({"epoch": epoch, "train": train_m, "val": val_m})
        print(f"Epoch {epoch:3d} | train mae={train_m['mae']:.4f} | "
              f"val mae={val_m['mae']:.4f} | {time.time()-t0:.0f}s")

        if val_m["mae"] < best_val_mae:
            best_val_mae, patience_ctr = val_m["mae"], 0
            torch.save(model.state_dict(), best_path)
            print(f"          ↑ best val MAE={best_val_mae:.4f} saved.")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    with open(out_dir / "train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)

    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_m, test_preds, test_targets = run_epoch(
        model, test_loader, criterion, None, None, device, False)

    pred_arr   = np.array(test_preds)
    target_arr = np.array(test_targets)
    pred_tiers = predictions_to_tiers(pred_arr, thresholds)

    test_labels = pd.read_csv(test_csv, usecols=["tier_label"], low_memory=False)["tier_label"].values
    macro_f1    = float(f1_score(test_labels, pred_tiers, average="macro", zero_division=0))

    print(f"Test MAE={test_m['mae']:.4f}  Tier macro_f1={macro_f1:.4f}")

    final_metrics = {
        "arch": args.arch, "subset": args.subset, "task": "regression",
        "target": "normalized_log_price",
        "best_val_mae":  best_val_mae,
        "test_mae":      test_m["mae"],
        "test_macro_f1": macro_f1,
        "thresholds":    thresholds,
        "hyperparams": {"epochs_run": len(train_log), "epochs_max": args.epochs,
                        "batch_size": args.batch_size, "dropout": args.dropout,
                        "patience": args.patience},
    }
    with open(out_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
