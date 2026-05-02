"""
train_regression.py — EfficientNetV2-S regression variant for DLCGIPG
======================================================================

Replaces the 4-class classification head with a single linear output
predicting log(price_usd).  Loss: HuberLoss.  Metrics: MAE and RMSE in
log-space; USD MAE / RMSE / Median-APE on test set.

Usage:
    /mnt/storage/projects/DLCGIPG/ja_scraper/venv/bin/python3 src/train_regression.py \
        --subset ja_natural \
        --data_dir /mnt/storage/projects/DLCGIPG/data \
        --image_dir_ja /mnt/storage/projects/DLCGIPG/ja_scraper/output/images \
        --image_dir_be /mnt/storage/projects/DLCGIPG/be_scraper/output/images \
        --results_dir /mnt/storage/projects/DLCGIPG/results

Outputs (all under results/training/regression/efficientnetv2/{subset}/):
  - best_model.pth     ← state dict of epoch with best val log-MAE
  - train_log.json     ← per-epoch metrics (loss, log_mae, log_rmse)
  - final_metrics.json ← test-set evaluation (log + USD metrics)
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
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

import pandas as pd

# ── project imports ───────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from models import get_model, count_params

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── ImageNet transforms (same as classification) ──────────────────────────────
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
    """
    Loads diamond images and returns (image_tensor, log_price_float32).

    Images expected at: image_dir / value_tier / {diamond_id}.jpg
    """

    def __init__(self, csv_path: str | Path, image_dir: str | Path, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform if transform is not None else val_test_transform

        df = pd.read_csv(
            csv_path,
            usecols=["diamond_id", "value_tier", "price_usd"],
            low_memory=False,
        )
        df["_log_price"] = np.log(df["price_usd"].astype(float))
        df["_img_path"] = df.apply(
            lambda r: self.image_dir / str(r["value_tier"]) / f"{r['diamond_id']}.jpg",
            axis=1,
        )

        missing = ~df["_img_path"].apply(lambda p: p.exists())
        n_missing = int(missing.sum())
        if n_missing:
            log.warning("%d image(s) missing from disk — skipping.", n_missing)
            df = df[~missing].reset_index(drop=True)

        self.records = df[["diamond_id", "_log_price", "_img_path"]].reset_index(drop=True)

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


# ── Model factory ─────────────────────────────────────────────────────────────

def build_regression_model(
    arch: str,
    base_lr: float,
    backbone_lr_scale: float,
    dropout: float,
) -> tuple[nn.Module, list[dict]]:
    """Any supported arch with its 4-class head swapped for a single linear output."""
    model, _ = get_model(arch, dropout=dropout)

    if arch == "resnet50":
        in_features = model.fc[3].in_features
        model.fc[3] = nn.Linear(in_features, 1)
        head_key = "fc"
    elif arch == "efficientnetv2":
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)
        head_key = "classifier"
    elif arch == "vit":
        in_features = model.heads.head[1].in_features
        model.heads.head[1] = nn.Linear(in_features, 1)
        head_key = "heads"
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    head_params     = [p for n, p in model.named_parameters() if head_key in n]
    backbone_params = [p for n, p in model.named_parameters() if head_key not in n]
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

            preds = model(images).squeeze(1)   # [B, 1] → [B]
            loss  = criterion(preds, targets)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss  += loss.item() * images.size(0)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    preds_arr   = np.array(all_preds,   dtype=np.float64)
    targets_arr = np.array(all_targets, dtype=np.float64)
    n = len(targets_arr)

    log_mae  = float(np.abs(preds_arr - targets_arr).mean())
    log_rmse = float(np.sqrt(((preds_arr - targets_arr) ** 2).mean()))

    return {
        "loss":     total_loss / n,
        "log_mae":  log_mae,
        "log_rmse": log_rmse,
    }


def evaluate_test(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
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
            total_loss  += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    preds_arr   = np.array(all_preds,   dtype=np.float64)
    targets_arr = np.array(all_targets, dtype=np.float64)
    n = len(targets_arr)

    log_mae  = float(np.abs(preds_arr - targets_arr).mean())
    log_rmse = float(np.sqrt(((preds_arr - targets_arr) ** 2).mean()))

    # Back-transform to USD for interpretability
    usd_preds  = np.exp(preds_arr)
    usd_true   = np.exp(targets_arr)
    usd_mae    = float(np.abs(usd_preds - usd_true).mean())
    usd_rmse   = float(np.sqrt(((usd_preds - usd_true) ** 2).mean()))
    usd_med_ape = float(np.median(np.abs(usd_preds - usd_true) / usd_true) * 100)  # %

    r2       = float(r2_score(targets_arr, preds_arr))
    spearman = float(spearmanr(targets_arr, preds_arr).statistic)

    return {
        "test_loss":        total_loss / n,
        "test_log_mae":     log_mae,
        "test_log_rmse":    log_rmse,
        "test_usd_mae":     usd_mae,
        "test_usd_rmse":    usd_rmse,
        "test_usd_med_ape": usd_med_ape,
        "test_r2":          r2,
        "test_spearman":    spearman,
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
    p = argparse.ArgumentParser(description="DLCGIPG Stage 2 — regression trainer")
    p.add_argument("--arch",         required=True,
                   choices=["resnet50", "efficientnetv2", "vit"])
    p.add_argument("--subset",       required=True,
                   choices=["ja_natural", "be_natural", "ja_lab", "be_lab"],
                   help="Within-site subset to train on")
    p.add_argument("--data_dir",     required=True)
    p.add_argument("--image_dir_ja", required=True)
    p.add_argument("--image_dir_be", required=True)
    p.add_argument("--results_dir",  required=True)
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--base_lr",      type=float, default=3e-4)
    p.add_argument("--backbone_lr_scale", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--huber_delta",  type=float, default=0.5,
                   help="Huber loss delta (in log-price units)")
    p.add_argument("--patience",     type=int,   default=5,
                   help="Early-stopping patience on val log-MAE")
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
    out_dir   = Path(args.results_dir) / "training" / "regression" / args.arch / args.subset
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_model.pth"
    log_path  = out_dir / "train_log.json"
    print(f"Output dir: {out_dir}")

    # ── data ──────────────────────────────────────────────────────────────────
    splits_dir = Path(args.data_dir) / "splits"
    image_dir  = args.image_dir_ja if args.subset.startswith("ja") else args.image_dir_be

    print("Loading data splits …")
    train_loader = build_dataloader(
        splits_dir / f"{args.subset}_train.csv", image_dir,
        split="train", batch_size=args.batch_size, num_workers=args.num_workers,
    )
    val_loader = build_dataloader(
        splits_dir / f"{args.subset}_val.csv", image_dir,
        split="val",   batch_size=args.batch_size, num_workers=args.num_workers,
    )
    test_loader = build_dataloader(
        splits_dir / f"{args.subset}_test.csv", image_dir,
        split="test",  batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print(f"   train={len(train_loader.dataset):,}  "
          f"val={len(val_loader.dataset):,}  "
          f"test={len(test_loader.dataset):,}")

    # ── model ─────────────────────────────────────────────────────────────────
    print(f"Building {args.arch} regression model …")
    model, param_groups = build_regression_model(
        arch             = args.arch,
        base_lr          = args.base_lr,
        backbone_lr_scale= args.backbone_lr_scale,
        dropout          = args.dropout,
    )
    model = model.to(device)
    stats = count_params(model)
    print(f"   Params: total={stats['total']:,}  trainable={stats['trainable']:,}")
    if args.arch == "resnet50":       head = model.fc
    elif args.arch == "efficientnetv2": head = model.classifier
    else:                               head = model.heads.head
    print(f"   Head  : {head}")

    # ── loss / optimizer / scheduler ──────────────────────────────────────────
    criterion = nn.HuberLoss(delta=args.huber_delta)
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── training loop ─────────────────────────────────────────────────────────
    best_val_mae  = float("inf")
    patience_ctr  = 0
    epoch_log     = []

    header = (f"{'Epoch':>5}  {'Train Loss':>10}  {'Train logMAE':>12}  "
              f"{'Train logRMSE':>13}  {'Val Loss':>8}  "
              f"{'Val logMAE':>10}  {'Val logRMSE':>11}  {'Time':>6}")
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
              f"{train_m['log_rmse']:>13.4f}  {val_m['loss']:>8.4f}  "
              f"{val_m['log_mae']:>10.4f}  {val_m['log_rmse']:>11.4f}  {elapsed:>5.0f}s")

        # ── checkpoint on best val log-MAE ────────────────────────────────────
        if val_m["log_mae"] < best_val_mae:
            best_val_mae = val_m["log_mae"]
            patience_ctr = 0
            save_checkpoint(model, ckpt_path)
            print(f"         ✓ New best val log-MAE={best_val_mae:.4f} — checkpoint saved")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\nEarly stopping after {epoch} epochs "
                      f"(no val log-MAE improvement for {args.patience} epochs).")
                break

        with open(log_path, "w") as f:
            json.dump(epoch_log, f, indent=2)

    # ── test evaluation ───────────────────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation …")
    model = load_checkpoint(model, ckpt_path, device)
    test_results = evaluate_test(model, test_loader, criterion, device)

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"TEST RESULTS  [{args.arch} / {args.subset}]")
    print(f"  Log-MAE        : {test_results['test_log_mae']:.4f}")
    print(f"  Log-RMSE       : {test_results['test_log_rmse']:.4f}")
    print(f"  USD MAE        : ${test_results['test_usd_mae']:>10,.0f}")
    print(f"  USD RMSE       : ${test_results['test_usd_rmse']:>10,.0f}")
    print(f"  USD Median APE : {test_results['test_usd_med_ape']:.1f}%")
    print(f"  n_test         : {test_results['n_test']:,}")
    print(sep)

    # ── save ──────────────────────────────────────────────────────────────────
    final_meta = {
        "arch":           args.arch,
        "subset":         args.subset,
        "task":           "regression",
        "target":         "log(price_usd)",
        "best_val_log_mae": best_val_mae,
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
