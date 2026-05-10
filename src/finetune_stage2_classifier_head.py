"""
finetune_stage2_classifier_head.py — Tier 2.2 Stage 2 auxiliary classifier head

Adds a 4-way classification head to the existing Stage 2 EfficientNetV2-S regression
model. Backbone and regression head are frozen; only the new classifier head is
trained, with class-weighted cross-entropy to combat tier imbalance.

Strategy:
  1. Load Stage 2 weights, replace classifier with Identity to expose 1280-d features.
  2. Extract features once for train + val (no augmentation; head is essentially
     logistic regression on a frozen feature space).
  3. Train classifier head with weighted CE on cached features. Epochs are sub-second.
  4. Early-stop on val macro F1.

Class weights: sqrt inverse-frequency, mean-normalized.
  budget=0.87  mid_range=0.63  premium=1.15  investment_grade=1.35

Outputs (results/training/regression/efficientnetv2/combined_all/):
  - classifier_head.pth
  - cls_head_train_log.json
  - cls_head_final.json
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
from sklearn.metrics import classification_report, f1_score
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_v2_s

ROOT          = Path(__file__).resolve().parent.parent
SPLITS        = ROOT / "data" / "splits"
JA_IMGS       = ROOT / "ja_scraper" / "output" / "images"
BE_IMGS       = ROOT / "be_scraper" / "output" / "images"
STAGE2_DIR    = ROOT / "results" / "training" / "regression" / "efficientnetv2" / "combined_all"
SRC_WEIGHTS   = STAGE2_DIR / "best_model.pth"
HEAD_OUT      = STAGE2_DIR / "classifier_head.pth"

DROPOUT     = 0.3
NUM_TIERS   = 4
TIER_LABELS = ["budget", "mid_range", "premium", "investment_grade"]

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


# ── Dataset ──────────────────────────────────────────────────────────────────

class Stage2TierDataset(Dataset):
    """Yields (image, tier_label int) from a combined_all_*.csv split."""

    def __init__(self, csv_path: Path, transform):
        df = pd.read_csv(csv_path,
                         usecols=["diamond_id", "value_tier", "tier_label", "source_subset"],
                         low_memory=False)
        df["_img_dir"] = df["source_subset"].apply(
            lambda s: JA_IMGS if s.startswith("ja") else BE_IMGS
        )
        df["_img_path"] = df.apply(
            lambda r: r["_img_dir"] / str(r["value_tier"]) / f"{int(r['diamond_id'])}.jpg",
            axis=1,
        )
        df = df[df["_img_path"].apply(lambda p: p.exists())].reset_index(drop=True)
        self.records = df[["tier_label", "_img_path"]].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        try:
            img = Image.open(row["_img_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        return self.transform(img), int(row["tier_label"])


# ── Backbone (frozen) ────────────────────────────────────────────────────────

def build_frozen_backbone(weights_path: Path, device: torch.device) -> tuple[nn.Module, int]:
    """Load Stage 2 weights, strip the regression head, return frozen backbone + feature dim."""
    model = efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(in_features, 1),
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    # Strip classifier — model(x) now returns 1280-d pooled features
    model.classifier = nn.Identity()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device).eval(), in_features


# ── Feature extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(loader: DataLoader, backbone: nn.Module, device: torch.device,
                     tag: str) -> tuple[torch.Tensor, torch.Tensor]:
    print(f"\nExtracting features [{tag}] over {len(loader.dataset):,} images...")
    feats, labels = [], []
    t0 = time.time()
    n_seen = 0
    n_total = len(loader.dataset)
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast("cuda"):
            f = backbone(imgs)
        feats.append(f.float().cpu())
        labels.append(lbls)
        n_seen += imgs.size(0)
        if n_seen % (loader.batch_size * 200) == 0:
            print(f"  {n_seen:,}/{n_total:,}  ({(time.time()-t0):.0f}s)")
    feats  = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    print(f"  done: feats={tuple(feats.shape)} labels={tuple(labels.shape)}  "
          f"({time.time()-t0:.0f}s)")
    return feats, labels


# ── Training (head only, on cached features) ─────────────────────────────────

def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """sqrt inverse-frequency, mean-normalized."""
    counts = torch.bincount(labels, minlength=NUM_TIERS).float()
    n      = counts.sum()
    raw    = torch.sqrt(n / (NUM_TIERS * counts))
    return raw / raw.mean()


def train_head(train_feats, train_labels, val_feats, val_labels,
               in_features, weights, args, device):
    cls_head = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(in_features, NUM_TIERS),
    ).to(device)

    optimizer = torch.optim.AdamW(cls_head.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    train_feats  = train_feats.to(device)
    train_labels = train_labels.to(device)
    val_feats    = val_feats.to(device)
    val_labels   = val_labels.to(device)

    n      = len(train_feats)
    best   = -1.0
    best_e = 0
    patience_ctr = 0
    log = []

    print(f"\nTraining classifier head: {n:,} train feats, {len(val_feats):,} val")
    print(f"  in_features={in_features}  classes={NUM_TIERS}  lr={args.lr}  "
          f"weight_decay={args.weight_decay}")
    print(f"  class weights: " +
          "  ".join(f"{t}={float(w):.3f}" for t, w in zip(TIER_LABELS, weights)))
    print(f"  epochs={args.epochs}  patience={args.patience}  batch_size={args.batch_size}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        cls_head.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        n_correct  = 0
        for i in range(0, n, args.batch_size):
            idx    = perm[i:i + args.batch_size]
            batch_x = train_feats[idx]
            batch_y = train_labels[idx]
            optimizer.zero_grad()
            logits = cls_head(batch_x)
            loss   = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            n_correct  += (logits.argmax(dim=1) == batch_y).sum().item()
        train_loss = total_loss / n
        train_acc  = n_correct / n

        # Validate
        cls_head.eval()
        with torch.no_grad():
            val_logits = cls_head(val_feats)
        val_preds  = val_logits.argmax(dim=1).cpu().numpy()
        val_lbls   = val_labels.cpu().numpy()
        val_acc    = float((val_preds == val_lbls).mean())
        val_f1     = float(f1_score(val_lbls, val_preds, average="macro", zero_division=0))
        per_class_f1 = f1_score(val_lbls, val_preds, average=None,
                                 labels=list(range(NUM_TIERS)), zero_division=0).tolist()

        elapsed = time.time() - t0
        per_class_str = "  ".join(f"{t[:3]}={f:.3f}" for t, f in zip(TIER_LABELS, per_class_f1))
        print(f"Epoch {epoch:3d} | train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val acc={val_acc:.4f} macro_f1={val_f1:.4f} | {per_class_str} | {elapsed:.1f}s")

        log.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_acc":    val_acc,    "val_macro_f1": val_f1,
            "val_per_class_f1": dict(zip(TIER_LABELS, per_class_f1)),
            "elapsed_s":  elapsed,
        })

        if val_f1 > best:
            best   = val_f1
            best_e = epoch
            torch.save(cls_head.state_dict(), HEAD_OUT)
            print(f"          [BEST] val_macro_f1={val_f1:.4f}  saved {HEAD_OUT.name}")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\nEarly stop at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

    return best, best_e, log


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--patience",     type=int,   default=5)
    p.add_argument("--batch_size",   type=int,   default=512)
    p.add_argument("--num_workers",  type=int,   default=8)
    p.add_argument("--lr",           type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed",         type=int,   default=42)
    args = p.parse_args()

    STAGE2_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"GPU   : {torch.cuda.get_device_name(0)}  {free/1024**3:.1f}/{total/1024**3:.1f} GiB free")

    print("\nBuilding frozen Stage 2 backbone...")
    backbone, in_features = build_frozen_backbone(SRC_WEIGHTS, device)
    print(f"  loaded: {SRC_WEIGHTS}")
    print(f"  in_features: {in_features}")

    print("\nBuilding datasets...")
    train_ds = Stage2TierDataset(SPLITS / "combined_all_train.csv", val_transform)
    val_ds   = Stage2TierDataset(SPLITS / "combined_all_val.csv",   val_transform)
    print(f"  train: {len(train_ds):,}   val: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)

    train_feats, train_labels = extract_features(train_loader, backbone, device, "train")
    val_feats,   val_labels   = extract_features(val_loader,   backbone, device, "val")

    weights = compute_class_weights(train_labels)
    best, best_e, log = train_head(train_feats, train_labels, val_feats, val_labels,
                                    in_features, weights, args, device)

    # Final report on val using best head
    cls_head = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(in_features, NUM_TIERS),
    ).to(device)
    cls_head.load_state_dict(torch.load(HEAD_OUT, map_location=device))
    cls_head.eval()
    with torch.no_grad():
        val_logits = cls_head(val_feats.to(device))
    val_preds = val_logits.argmax(dim=1).cpu().numpy()
    val_lbls  = val_labels.cpu().numpy()
    print("\nValidation classification report (best head):")
    print(classification_report(val_lbls, val_preds, target_names=TIER_LABELS, zero_division=0))

    # Save
    final = {
        "src_weights":   str(SRC_WEIGHTS),
        "head_weights":  str(HEAD_OUT),
        "best_macro_f1": best,
        "best_epoch":    best_e,
        "class_weights": dict(zip(TIER_LABELS, [float(w) for w in weights])),
        "hyperparams":   vars(args),
    }
    with open(STAGE2_DIR / "cls_head_final.json", "w") as f:
        json.dump(final, f, indent=2)
    with open(STAGE2_DIR / "cls_head_train_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nDone. Best val macro F1 = {best:.4f} at epoch {best_e}")
    print(f"Head weights: {HEAD_OUT}")


if __name__ == "__main__":
    main()
