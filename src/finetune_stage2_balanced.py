"""
finetune_stage2_balanced.py — Tier 3.2 Stage 2 retrain with class-balanced sampling

Trains a fresh EfficientNetV2-S regression model on combined_all_train.csv with a
WeightedRandomSampler over tier_label so the backbone sees premium and
investment_grade samples disproportionately during training. Goal: learn features
that better separate the harder tiers, lifting macro F1 above the 0.667 ceiling
of the regression+thresholds baseline.

Key differences from train_regression_combined.py:
  - WeightedRandomSampler with sqrt inverse-frequency tier weights
  - Early-stops on val tier macro F1 (not val MAE), since that's the metric we're
    trying to lift; tracks both metrics each epoch for diagnostic purposes
  - Outputs to a new directory (efficientnetv2_v2) so the original stays as fallback
  - Per-epoch checkpoint snapshots to enable resume / compare

Usage:
    python src/finetune_stage2_balanced.py
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
from sklearn.metrics import f1_score, classification_report
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

ROOT          = Path(__file__).resolve().parent.parent
SPLITS        = ROOT / "data" / "splits"
JA_IMGS       = ROOT / "ja_scraper" / "output" / "images"
BE_IMGS       = ROOT / "be_scraper" / "output" / "images"
OUT_DIR       = ROOT / "results" / "training" / "regression" / "efficientnetv2_v2" / "combined_all"

DROPOUT     = 0.3
TIER_LABELS = ["budget", "mid_range", "premium", "investment_grade"]
NUM_TIERS   = 4

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

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


# ── Dataset ──────────────────────────────────────────────────────────────────

class CombinedRegressionDataset(Dataset):
    """Returns (image, normalized_log_price, tier_label) — tier_label used for
    weighted sampling and val metrics, not for the regression objective."""

    def __init__(self, csv_path: Path, transform):
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
        df = df[df["_img_path"].apply(lambda p: p.exists())].reset_index(drop=True)
        self.records = df[["normalized_log_price", "tier_label", "_img_path"]].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    @property
    def tier_labels(self) -> np.ndarray:
        return self.records["tier_label"].values.astype(np.int64)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        try:
            img = Image.open(row["_img_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        return (self.transform(img),
                torch.tensor(row["normalized_log_price"], dtype=torch.float32),
                int(row["tier_label"]))


# ── Sampler ──────────────────────────────────────────────────────────────────

def build_weighted_sampler(tier_labels: np.ndarray) -> WeightedRandomSampler:
    """Sqrt inverse-frequency weighted sampler (soft balancing)."""
    counts  = np.bincount(tier_labels, minlength=NUM_TIERS).astype(np.float64)
    inv     = 1.0 / np.sqrt(counts)
    sample_w = inv[tier_labels]
    print(f"  Tier counts (train): {dict(zip(TIER_LABELS, counts.astype(int).tolist()))}")
    expected_frac = inv * counts / (inv * counts).sum()
    print(f"  Expected sample frac after weighting: " +
          "  ".join(f"{t}={float(f):.3f}" for t, f in zip(TIER_LABELS, expected_frac)))
    return WeightedRandomSampler(weights=torch.tensor(sample_w, dtype=torch.double),
                                  num_samples=len(tier_labels), replacement=True)


# ── Tier prediction from regression output ───────────────────────────────────

def compute_tier_thresholds(csv_path: Path) -> list[float]:
    df = pd.read_csv(csv_path, usecols=["value_tier", "normalized_log_price"], low_memory=False)
    tier_means = df.groupby("value_tier")["normalized_log_price"].mean().sort_values()
    means = tier_means.values.tolist()
    return [(means[i] + means[i+1]) / 2 for i in range(len(means) - 1)]


def predictions_to_tiers(preds: np.ndarray, thresholds: list[float]) -> np.ndarray:
    out = np.zeros(len(preds), dtype=np.int64)
    for t in thresholds:
        out += (preds > t).astype(np.int64)
    return out


# ── Model ────────────────────────────────────────────────────────────────────

def build_model() -> nn.Module:
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(in_features, 1),
    )
    return model


# ── Train / eval ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler, device, is_train):
    model.train() if is_train else model.eval()
    total_loss, n_seen = 0.0, 0
    all_preds, all_targets, all_tiers = [], [], []
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, targets, tier_lbls in loader:
            imgs    = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad()
                with autocast("cuda"):
                    preds = model(imgs).squeeze(1)
                    loss  = criterion(preds, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with autocast("cuda"):
                    preds = model(imgs).squeeze(1)
                    loss  = criterion(preds, targets)
            total_loss += loss.item() * imgs.size(0)
            n_seen     += imgs.size(0)
            all_preds.extend(preds.detach().float().cpu().numpy().tolist())
            all_targets.extend(targets.detach().cpu().numpy().tolist())
            all_tiers.extend(tier_lbls.numpy().tolist())
    return {
        "loss":    total_loss / n_seen,
        "mae":     float(np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))),
        "preds":   np.array(all_preds),
        "targets": np.array(all_targets),
        "tiers":   np.array(all_tiers),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--patience",     type=int,   default=5)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--num_workers",  type=int,   default=8)
    p.add_argument("--lr_backbone",  type=float, default=3e-5)
    p.add_argument("--lr_head",      type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed",         type=int,   default=42)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"GPU    : {torch.cuda.get_device_name(0)}  {free/1024**3:.1f}/{total/1024**3:.1f} GiB free")

    train_csv = SPLITS / "combined_all_train.csv"
    val_csv   = SPLITS / "combined_all_val.csv"

    print("\nBuilding datasets...")
    train_ds = CombinedRegressionDataset(train_csv, train_transform)
    val_ds   = CombinedRegressionDataset(val_csv,   val_transform)
    print(f"  train: {len(train_ds):,}   val: {len(val_ds):,}")

    print("\nBuilding weighted sampler (sqrt inverse-frequency)...")
    sampler = build_weighted_sampler(train_ds.tier_labels)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print("\nThresholds (train-set tier midpoints, used for tier predictions):")
    thresholds = compute_tier_thresholds(train_csv)
    print(f"  {[f'{t:.4f}' for t in thresholds]}")

    print("\nBuilding model (EfficientNetV2-S, ImageNet init)...")
    model = build_model().to(device)
    in_features = model.classifier[1].in_features
    head_params     = list(model.classifier.parameters())
    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    optimizer = AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params,     "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.HuberLoss(delta=0.5)
    scaler    = GradScaler()

    best_f1, best_e, patience_ctr = -1.0, 0, 0
    best_path  = OUT_DIR / "best_model.pth"
    last_path  = OUT_DIR / "last_model.pth"
    log = []

    print(f"\nTraining up to {args.epochs} epochs (patience={args.patience}, "
          f"early-stop on val tier macro F1)...\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, criterion, optimizer, scaler, device, True)
        val_m   = run_epoch(model, val_loader,   criterion, None,      None,   device, False)
        scheduler.step()

        val_pred_tiers = predictions_to_tiers(val_m["preds"], thresholds)
        val_macro_f1   = float(f1_score(val_m["tiers"], val_pred_tiers, average="macro", zero_division=0))
        val_acc        = float((val_pred_tiers == val_m["tiers"]).mean())
        per_class_f1   = f1_score(val_m["tiers"], val_pred_tiers, average=None,
                                   labels=list(range(NUM_TIERS)), zero_division=0).tolist()

        elapsed = time.time() - t0
        per_class_str = "  ".join(f"{t[:3]}={f:.3f}" for t, f in zip(TIER_LABELS, per_class_f1))
        print(f"Epoch {epoch:3d} | train mae={train_m['mae']:.4f} | "
              f"val mae={val_m['mae']:.4f} acc={val_acc:.4f} f1={val_macro_f1:.4f} | "
              f"{per_class_str} | {elapsed:.0f}s")

        log.append({
            "epoch": epoch,
            "train_loss": train_m["loss"], "train_mae": train_m["mae"],
            "val_loss":   val_m["loss"],   "val_mae":   val_m["mae"],
            "val_acc":    val_acc,         "val_macro_f1": val_macro_f1,
            "val_per_class_f1": dict(zip(TIER_LABELS, per_class_f1)),
            "elapsed_s": elapsed,
        })
        with open(OUT_DIR / "train_log.json", "w") as f:
            json.dump(log, f, indent=2)
        torch.save(model.state_dict(), last_path)

        if val_macro_f1 > best_f1:
            best_f1, best_e, patience_ctr = val_macro_f1, epoch, 0
            torch.save(model.state_dict(), best_path)
            print(f"          [BEST] val macro F1={val_macro_f1:.4f}  saved {best_path.name}")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\nEarly stop at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

    # Final report
    print(f"\nLoading best checkpoint (epoch {best_e}) for final val report...")
    model.load_state_dict(torch.load(best_path, map_location=device))
    val_m = run_epoch(model, val_loader, criterion, None, None, device, False)
    val_pred_tiers = predictions_to_tiers(val_m["preds"], thresholds)
    print("\nValidation classification report (best by macro F1):")
    print(classification_report(val_m["tiers"], val_pred_tiers,
                                 target_names=TIER_LABELS, zero_division=0))

    final = {
        "out_dir":        str(OUT_DIR),
        "best_epoch":     best_e,
        "best_val_macro_f1": best_f1,
        "best_val_mae":   val_m["mae"],
        "thresholds":     thresholds,
        "hyperparams":    vars(args),
        "sampler":        "WeightedRandomSampler sqrt inverse-frequency",
    }
    with open(OUT_DIR / "final_metrics.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nDone. Best val macro F1={best_f1:.4f} at epoch {best_e}")
    print(f"Best weights: {best_path}")


if __name__ == "__main__":
    main()
