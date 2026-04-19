"""
train_regression_cross.py — Cross-domain regression (EfficientNetV2-S) for DLCGIPG
====================================================================================

Trains on the SOURCE domain's cross_train split, validates on SOURCE cross_val,
and tests on TARGET domain's test split.  This mirrors the classification
cross-domain protocol from train.py / run_cross_domain.sh.

Usage:
    python src/train_regression_cross.py \
        --source_subset ja_natural \
        --target_subset be_natural \
        --data_dir /mnt/storage/projects/DLCGIPG/data \
        --image_dir_ja /mnt/storage/projects/DLCGIPG/ja_scraper/output/images \
        --image_dir_be /mnt/storage/projects/DLCGIPG/be_scraper/output/images \
        --results_dir /mnt/storage/projects/DLCGIPG/results \
        --epochs 50 --patience 10 --batch_size 64 --base_lr 3e-4

Outputs (under results/training/regression_cross/efficientnetv2__{source}__{target}/):
  - best_model.pth       checkpoint (best val log-MAE)
  - train_log.json       per-epoch metrics
  - final_metrics.json   test-set metrics incl. tier_macro_f1 (regression→tier bucketing)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import f1_score, r2_score
from scipy.stats import spearmanr

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from models import get_model, count_params

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TIER_ORDER = ["budget", "mid_range", "premium", "investment_grade"]
TIER_INT   = {t: i for i, t in enumerate(TIER_ORDER)}

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


# ── Dataset ───────────────────────────────────────────────────────────────────

class DiamondRegressionDataset(Dataset):
    """Loads diamond images, returns (image_tensor, log_price).  Also stores
    integer tier labels so we can compute classification-style F1 after inference."""

    def __init__(self, csv_path: str | Path, image_dir: str | Path, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform if transform is not None else val_test_transform

        df = pd.read_csv(
            csv_path,
            usecols=["diamond_id", "value_tier", "price_usd"],
            low_memory=False,
        )
        df["_log_price"] = np.log(df["price_usd"].astype(float))
        df["_tier_int"]  = df["value_tier"].map(TIER_INT).fillna(-1).astype(int)
        df["_img_path"]  = df.apply(
            lambda r: self.image_dir / str(r["value_tier"]) / f"{r['diamond_id']}.jpg",
            axis=1,
        )

        missing = ~df["_img_path"].apply(lambda p: p.exists())
        n_missing = int(missing.sum())
        if n_missing:
            log.warning("%d image(s) missing from disk — skipping.", n_missing)
            df = df[~missing].reset_index(drop=True)

        self.records = df[["diamond_id", "_log_price", "_tier_int", "_img_path"]].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.records.iloc[idx]
        target = torch.tensor(float(row["_log_price"]), dtype=torch.float32)
        try:
            image = Image.open(row["_img_path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (128, 128, 128))
        if self.transform:
            image = self.transform(image)
        return image, target

    @property
    def tier_labels(self) -> list[int]:
        return self.records["_tier_int"].tolist()


def build_dataloader(
    csv_path: str | Path,
    image_dir: str | Path,
    split: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
) -> DataLoader:
    tfm = train_transform if split == "train" else val_test_transform
    dataset = DiamondRegressionDataset(csv_path, image_dir, transform=tfm)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
    )


def resolve_image_dir(subset: str, image_dir_ja: str, image_dir_be: str) -> str:
    if subset.startswith("ja"):
        return image_dir_ja
    elif subset.startswith("be"):
        return image_dir_be
    raise ValueError(f"Cannot resolve image dir for subset '{subset}'")


# ── Tier threshold helpers ────────────────────────────────────────────────────

def compute_tier_thresholds(csv_path: str | Path) -> list[float]:
    """Return 3 log-price thresholds computed from tier mean log-prices in csv_path.

    Thresholds are midpoints between adjacent tier means so predictions can be
    bucketed into (budget, mid_range, premium, investment_grade) = (0,1,2,3).
    """
    df = pd.read_csv(csv_path, usecols=["value_tier", "price_usd"], low_memory=False)
    df["log_price"] = np.log(df["price_usd"].astype(float))
    means = []
    for tier in TIER_ORDER:
        sub = df[df["value_tier"] == tier]["log_price"]
        if len(sub) == 0:
            raise ValueError(f"Tier '{tier}' not found in {csv_path}")
        means.append(float(sub.mean()))
    return [(means[i] + means[i + 1]) / 2.0 for i in range(len(means) - 1)]


def log_price_to_tier(log_price: float, thresholds: list[float]) -> int:
    for i, t in enumerate(thresholds):
        if log_price < t:
            return i
    return len(thresholds)


# ── Model factory ─────────────────────────────────────────────────────────────

def build_regression_model(
    base_lr: float,
    backbone_lr_scale: float,
    dropout: float,
) -> tuple[nn.Module, list[dict]]:
    model, _ = get_model("efficientnetv2", dropout=dropout)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, 1),
    )
    head_params     = [p for n, p in model.named_parameters() if "classifier" in n]
    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    param_groups = [
        {"params": backbone_params, "lr": base_lr * backbone_lr_scale},
        {"params": head_params,     "lr": base_lr},
    ]
    return model, param_groups


# ── Train / eval loops ────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    is_train: bool,
) -> dict:
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_preds: list[float] = []
    all_targets: list[float] = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, targets in loader:
            images  = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            preds = model(images).squeeze(1)
            loss  = criterion(preds, targets)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    preds_arr   = np.array(all_preds,   dtype=np.float64)
    targets_arr = np.array(all_targets, dtype=np.float64)
    n = len(targets_arr)

    return {
        "loss":     total_loss / n,
        "log_mae":  float(np.abs(preds_arr - targets_arr).mean()),
        "log_rmse": float(np.sqrt(((preds_arr - targets_arr) ** 2).mean())),
    }


def evaluate_test(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tier_thresholds: list[float],
) -> dict:
    """Run test inference.  Computes regression metrics and a classification-style
    macro F1 by bucketing log-price predictions using TARGET domain thresholds."""
    model.eval()
    total_loss = 0.0
    all_preds:   list[float] = []
    all_targets: list[float] = []

    with torch.no_grad():
        for images, targets in loader:
            images  = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            preds   = model(images).squeeze(1)
            loss    = criterion(preds, targets)
            total_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    preds_arr   = np.array(all_preds,   dtype=np.float64)
    targets_arr = np.array(all_targets, dtype=np.float64)
    n = len(targets_arr)

    log_mae  = float(np.abs(preds_arr - targets_arr).mean())
    log_rmse = float(np.sqrt(((preds_arr - targets_arr) ** 2).mean()))

    usd_preds   = np.exp(preds_arr)
    usd_true    = np.exp(targets_arr)
    usd_mae     = float(np.abs(usd_preds - usd_true).mean())
    usd_rmse    = float(np.sqrt(((usd_preds - usd_true) ** 2).mean()))
    usd_med_ape = float(np.median(np.abs(usd_preds - usd_true) / usd_true) * 100)

    r2       = float(r2_score(targets_arr, preds_arr))
    spearman = float(spearmanr(targets_arr, preds_arr).statistic)

    # ── classification-style F1 via tier bucketing ────────────────────────────
    true_tiers = loader.dataset.tier_labels           # list[int] from CSV tier_label
    pred_tiers = [log_price_to_tier(p, tier_thresholds) for p in all_preds]
    tier_macro_f1 = float(f1_score(true_tiers, pred_tiers, average="macro", zero_division=0))
    per_class_f1  = f1_score(true_tiers, pred_tiers, average=None,
                             labels=[0, 1, 2, 3], zero_division=0).tolist()

    return {
        "test_loss":        total_loss / n,
        "test_log_mae":     log_mae,
        "test_log_rmse":    log_rmse,
        "test_usd_mae":     usd_mae,
        "test_usd_rmse":    usd_rmse,
        "test_usd_med_ape": usd_med_ape,
        "test_r2":          r2,
        "test_spearman":    spearman,
        "test_tier_macro_f1":  tier_macro_f1,
        "test_tier_per_class_f1": {
            TIER_ORDER[i]: per_class_f1[i] for i in range(4)
        },
        "n_test":           n,
    }


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module, path: Path) -> None:
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> nn.Module:
    model.load_state_dict(torch.load(path, map_location=device))
    return model


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DLCGIPG — EfficientNetV2-S regression, cross-domain evaluation"
    )
    p.add_argument("--source_subset", required=True,
                   choices=["ja_natural", "ja_lab", "be_natural", "be_lab"],
                   help="Domain to train on")
    p.add_argument("--target_subset", required=True,
                   choices=["ja_natural", "ja_lab", "be_natural", "be_lab"],
                   help="Domain to evaluate on (test split + tier thresholds)")
    p.add_argument("--data_dir",     required=True)
    p.add_argument("--image_dir_ja", required=True)
    p.add_argument("--image_dir_be", required=True)
    p.add_argument("--results_dir",  required=True)
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--base_lr",      type=float, default=3e-4)
    p.add_argument("--backbone_lr_scale", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--huber_delta",  type=float, default=0.5)
    p.add_argument("--patience",     type=int,   default=10)
    p.add_argument("--num_workers",  type=int,   default=8)
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: No CUDA GPU detected — running on CPU.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── output directory ──────────────────────────────────────────────────────
    run_id  = f"efficientnetv2__{args.source_subset}__{args.target_subset}"
    out_dir = Path(args.results_dir) / "training" / "regression_cross" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_model.pth"
    log_path  = out_dir / "train_log.json"
    print(f"Run:        {run_id}")
    print(f"Output dir: {out_dir}")

    # ── data ──────────────────────────────────────────────────────────────────
    splits_dir     = Path(args.data_dir) / "splits"
    src_image_dir  = resolve_image_dir(args.source_subset, args.image_dir_ja, args.image_dir_be)
    tgt_image_dir  = resolve_image_dir(args.target_subset, args.image_dir_ja, args.image_dir_be)

    train_csv = splits_dir / f"{args.source_subset}_cross_train.csv"
    val_csv   = splits_dir / f"{args.source_subset}_cross_val.csv"
    test_csv  = splits_dir / f"{args.target_subset}_test.csv"
    # Use target domain's cross_train to compute tier thresholds
    tgt_ref_csv = splits_dir / f"{args.target_subset}_cross_train.csv"

    print(f"Train CSV:  {train_csv}")
    print(f"Val CSV:    {val_csv}")
    print(f"Test CSV:   {test_csv}")
    print(f"Tier thresholds from: {tgt_ref_csv}")

    print("Computing target tier thresholds …")
    tier_thresholds = compute_tier_thresholds(tgt_ref_csv)
    print(f"   thresholds (log-price): {[f'{t:.3f}' for t in tier_thresholds]}")

    print("Loading data splits …")
    train_loader = build_dataloader(train_csv, src_image_dir, "train",
                                    args.batch_size, args.num_workers)
    val_loader   = build_dataloader(val_csv,   src_image_dir, "val",
                                    args.batch_size, args.num_workers)
    test_loader  = build_dataloader(test_csv,  tgt_image_dir, "test",
                                    args.batch_size, args.num_workers)
    print(f"   train={len(train_loader.dataset):,}  "
          f"val={len(val_loader.dataset):,}  "
          f"test={len(test_loader.dataset):,}")

    # ── model ─────────────────────────────────────────────────────────────────
    print("Building EfficientNetV2-S regression model …")
    model, param_groups = build_regression_model(
        base_lr=args.base_lr,
        backbone_lr_scale=args.backbone_lr_scale,
        dropout=args.dropout,
    )
    model = model.to(device)
    stats = count_params(model)
    print(f"   Params: total={stats['total']:,}  trainable={stats['trainable']:,}")

    # ── loss / optimizer / scheduler ──────────────────────────────────────────
    criterion = nn.HuberLoss(delta=args.huber_delta)
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── training loop ─────────────────────────────────────────────────────────
    best_val_mae = float("inf")
    patience_ctr = 0
    epoch_log    = []

    header = (f"{'Epoch':>5}  {'Train Loss':>10}  {'Train logMAE':>12}  "
              f"{'Val Loss':>8}  {'Val logMAE':>10}  {'Time':>6}")
    print(f"\n{header}")
    print("─" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_m   = run_epoch(model, val_loader,   criterion, None,      device, is_train=False)
        scheduler.step()

        elapsed = time.time() - t0
        row = {"epoch": epoch, "train": train_m, "val": val_m, "elapsed_s": round(elapsed, 1)}
        epoch_log.append(row)

        print(f"{epoch:>5}  {train_m['loss']:>10.4f}  {train_m['log_mae']:>12.4f}  "
              f"{val_m['loss']:>8.4f}  {val_m['log_mae']:>10.4f}  {elapsed:>5.0f}s")

        if val_m["log_mae"] < best_val_mae:
            best_val_mae = val_m["log_mae"]
            patience_ctr = 0
            save_checkpoint(model, ckpt_path)
            print(f"         ✓ New best val log-MAE={best_val_mae:.4f} — checkpoint saved")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\nEarly stopping after {epoch} epochs "
                      f"(no improvement for {args.patience} epochs).")
                break

        with open(log_path, "w") as f:
            json.dump(epoch_log, f, indent=2)

    # ── test evaluation ───────────────────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation …")
    model = load_checkpoint(model, ckpt_path, device)
    test_results = evaluate_test(model, test_loader, criterion, device, tier_thresholds)

    sep = "─" * 65
    print(f"\n{sep}")
    print(f"TEST RESULTS  [efficientnetv2  {args.source_subset} → {args.target_subset}]")
    print(f"  Log-MAE          : {test_results['test_log_mae']:.4f}")
    print(f"  Log-RMSE         : {test_results['test_log_rmse']:.4f}")
    print(f"  R²               : {test_results['test_r2']:.4f}")
    print(f"  Spearman ρ       : {test_results['test_spearman']:.4f}")
    print(f"  USD MAE          : ${test_results['test_usd_mae']:>10,.0f}")
    print(f"  USD RMSE         : ${test_results['test_usd_rmse']:>10,.0f}")
    print(f"  USD Median APE   : {test_results['test_usd_med_ape']:.1f}%")
    print(f"  Tier Macro F1    : {test_results['test_tier_macro_f1']:.4f}")
    print(f"  n_test           : {test_results['n_test']:,}")
    print(sep)

    # ── save ──────────────────────────────────────────────────────────────────
    final_meta = {
        "arch":           "efficientnetv2",
        "source_subset":  args.source_subset,
        "target_subset":  args.target_subset,
        "task":           "regression_cross",
        "direction":      f"{args.source_subset}→{args.target_subset}",
        "target":         "log(price_usd)",
        "best_val_log_mae": best_val_mae,
        "tier_thresholds": tier_thresholds,
        "hyperparams": {
            "epochs_run":        len(epoch_log),
            "epochs_max":        args.epochs,
            "batch_size":        args.batch_size,
            "base_lr":           args.base_lr,
            "backbone_lr_scale": args.backbone_lr_scale,
            "weight_decay":      args.weight_decay,
            "dropout":           args.dropout,
            "huber_delta":       args.huber_delta,
            "patience":          args.patience,
        },
        **test_results,
    }
    final_path = out_dir / "final_metrics.json"
    with open(final_path, "w") as f:
        json.dump(final_meta, f, indent=2)

    print(f"\nSaved:\n  {final_path}\n  {log_path}\n  {ckpt_path}")


if __name__ == "__main__":
    main()
