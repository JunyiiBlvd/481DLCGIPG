"""
finetune_stage1_pyrite.py — Tier 2.1 Stage 1 fine-tune

Warm-starts from results/training/stage1/efficientnetv2/best_model.pth and fine-tunes
to reduce ja_natural Diamond -> Pyrite confusion (1,540 / 1,595 of Stage 1 misses).

Training data:
  - Full Combined-P1-Dataset/train/ ImageFolder (68 classes, preserves all other species)
  - Augmentation: N ja_natural diamonds from combined_all_train.csv injected as Diamond
    class (label 24). Test set is untouched.

Validation (two signals tracked each epoch):
  - Multi-species macro F1 on Combined-P1-Dataset/valid/ (catches collateral damage)
  - ja_natural Diamond recall on a sample of combined_all_val.csv ja_natural rows

Early-stop on composite: 0.5 * multi_f1 + 0.5 * ja_nat_recall.

Outputs: results/training/stage1/efficientnetv2_v2/{best_model.pth, train_log.json,
final_metrics.json}
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
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_s

ROOT          = Path(__file__).resolve().parent.parent
DATA_ROOT     = ROOT / "data" / "Combined-P1-Dataset"
SPLITS        = ROOT / "data" / "splits"
JA_IMGS       = ROOT / "ja_scraper" / "output" / "images"
SRC_WEIGHTS   = ROOT / "results" / "training" / "stage1" / "efficientnetv2" / "best_model.pth"
OUT_DIR       = ROOT / "results" / "training" / "stage1" / "efficientnetv2_v2"

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

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


# ── ja_natural Diamond augmentation dataset ──────────────────────────────────

class JaNaturalDiamondAug(Dataset):
    """Yields (image, diamond_label) pairs from combined_all_*.csv ja_natural rows."""

    def __init__(self, csv_path: Path, diamond_label: int, transform, sample_n: int | None = None,
                 seed: int = 42):
        df = pd.read_csv(csv_path, usecols=["diamond_id", "source_subset", "value_tier"],
                         low_memory=False)
        df = df[df["source_subset"] == "ja_natural"].copy()
        df["_path"] = df.apply(
            lambda r: JA_IMGS / str(r["value_tier"]) / f"{int(r['diamond_id'])}.jpg",
            axis=1,
        )
        df = df[df["_path"].apply(lambda p: p.exists())].reset_index(drop=True)
        if sample_n is not None and sample_n < len(df):
            df = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)
        self.paths = df["_path"].tolist()
        self.label = diamond_label
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        return self.transform(img), self.label


# ── Model ────────────────────────────────────────────────────────────────────

def build_model(device: torch.device) -> tuple[nn.Module, list, list]:
    model = efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(in_features, NUM_CLASSES),
    )
    model.load_state_dict(torch.load(SRC_WEIGHTS, map_location=device))
    head_params     = list(model.classifier.parameters())
    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    return model.to(device), backbone_params, head_params


# ── Eval helpers ─────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_multi_species(model, loader, device) -> tuple[float, float]:
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast("cuda"):
            logits = model(imgs)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return float(acc), float(f1)


@torch.no_grad()
def eval_ja_natural_recall(model, loader, diamond_idx, device) -> tuple[float, int, int]:
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast("cuda"):
            preds = model(imgs).argmax(dim=1).cpu().numpy()
        correct += int((preds == diamond_idx).sum())
        total   += len(preds)
    return (correct / total) if total else 0.0, correct, total


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",       type=int, default=5)
    p.add_argument("--patience",     type=int, default=2)
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--num_workers",  type=int, default=8)
    p.add_argument("--ja_aug_n",     type=int, default=3000,
                   help="ja_natural diamonds to inject as Diamond class")
    p.add_argument("--ja_val_n",     type=int, default=3000,
                   help="ja_natural val sample for retailer recall signal")
    p.add_argument("--lr_backbone",  type=float, default=1e-5)
    p.add_argument("--lr_head",      type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed",         type=int, default=42)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"GPU    : {torch.cuda.get_device_name(0)}  {free/1024**3:.1f}/{total/1024**3:.1f} GiB free")

    print("\nBuilding train dataset...")
    base_train  = datasets.ImageFolder(DATA_ROOT / "train", transform=train_transform)
    diamond_idx = base_train.class_to_idx["Diamond"]
    pyrite_idx  = base_train.class_to_idx["Pyrite"]
    assert diamond_idx == 24 and pyrite_idx == 41, "class index drift"
    print(f"  ImageFolder train: {len(base_train):,} imgs across {len(base_train.classes)} classes")
    print(f"  Diamond idx={diamond_idx}  Pyrite idx={pyrite_idx}")

    ja_aug = JaNaturalDiamondAug(SPLITS / "combined_all_train.csv",
                                 diamond_label=diamond_idx,
                                 transform=train_transform,
                                 sample_n=args.ja_aug_n,
                                 seed=args.seed)
    print(f"  ja_natural aug   : {len(ja_aug):,} imgs (Diamond label={diamond_idx})")

    train_ds = ConcatDataset([base_train, ja_aug])
    print(f"  Total train      : {len(train_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)

    print("Building val datasets...")
    multi_val_ds = datasets.ImageFolder(DATA_ROOT / "valid", transform=val_transform)
    multi_loader = DataLoader(multi_val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    print(f"  multi-species val: {len(multi_val_ds):,} imgs")

    ja_val = JaNaturalDiamondAug(SPLITS / "combined_all_val.csv",
                                 diamond_label=diamond_idx,
                                 transform=val_transform,
                                 sample_n=args.ja_val_n,
                                 seed=args.seed)
    ja_val_loader = DataLoader(ja_val, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)
    print(f"  ja_natural val   : {len(ja_val):,} imgs")

    print("\nLoading warm-start checkpoint and building model...")
    model, backbone_params, head_params = build_model(device)
    print(f"  loaded: {SRC_WEIGHTS}")

    optimizer = AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params,     "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler()

    # ── Pre-fine-tune baseline (sanity) ──────────────────────────────────
    print("\nBaseline (warm-start, before any updates)...")
    base_acc, base_f1 = eval_multi_species(model, multi_loader, device)
    base_recall, base_c, base_t = eval_ja_natural_recall(model, ja_val_loader, diamond_idx, device)
    base_score = 0.5 * base_f1 + 0.5 * base_recall
    print(f"  multi-species: acc={base_acc:.4f}  macro_f1={base_f1:.4f}")
    print(f"  ja_natural   : recall={base_recall:.4f} ({base_c}/{base_t})")
    print(f"  composite    : {base_score:.4f}")

    best_score   = base_score
    patience_ctr = 0
    best_path    = OUT_DIR / "best_model.pth"
    train_log    = [{
        "epoch": 0, "phase": "baseline",
        "multi_acc": base_acc, "multi_f1": base_f1,
        "ja_nat_recall": base_recall, "ja_nat_correct": base_c, "ja_nat_total": base_t,
        "composite": base_score,
    }]
    torch.save(model.state_dict(), best_path)
    print(f"  saved baseline to {best_path}")

    # ── Train ────────────────────────────────────────────────────────────
    print(f"\nFine-tuning up to {args.epochs} epochs, patience={args.patience}...")
    print(f"  LR: backbone={args.lr_backbone}  head={args.lr_head}  weight_decay={args.weight_decay}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, n_seen, n_correct = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast("cuda"):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * imgs.size(0)
            n_seen     += imgs.size(0)
            n_correct  += (logits.argmax(dim=1) == labels).sum().item()
        train_loss = total_loss / n_seen
        train_acc  = n_correct / n_seen

        m_acc, m_f1 = eval_multi_species(model, multi_loader, device)
        ja_recall, ja_c, ja_t = eval_ja_natural_recall(model, ja_val_loader, diamond_idx, device)
        composite = 0.5 * m_f1 + 0.5 * ja_recall

        elapsed = time.time() - t0
        print(f"Epoch {epoch:2d} | train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"multi acc={m_acc:.4f} f1={m_f1:.4f} | ja_nat recall={ja_recall:.4f} "
              f"({ja_c}/{ja_t}) | composite={composite:.4f} | {elapsed:.0f}s")

        train_log.append({
            "epoch": epoch, "phase": "train",
            "train_loss": train_loss, "train_acc": train_acc,
            "multi_acc": m_acc, "multi_f1": m_f1,
            "ja_nat_recall": ja_recall, "ja_nat_correct": ja_c, "ja_nat_total": ja_t,
            "composite": composite, "elapsed_s": elapsed,
        })

        if composite > best_score:
            best_score   = composite
            patience_ctr = 0
            torch.save(model.state_dict(), best_path)
            print(f"          [BEST] composite={composite:.4f}  saved to {best_path.name}")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\nEarly stop at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

        # Save train log incrementally
        with open(OUT_DIR / "train_log.json", "w") as f:
            json.dump(train_log, f, indent=2)

    # ── Final ────────────────────────────────────────────────────────────
    final = {
        "src_weights":     str(SRC_WEIGHTS),
        "out_weights":     str(best_path),
        "baseline":        train_log[0],
        "best_composite":  best_score,
        "best_epoch":      max(range(len(train_log)),
                                key=lambda i: train_log[i]["composite"]),
        "hyperparams": vars(args),
    }
    with open(OUT_DIR / "final_metrics.json", "w") as f:
        json.dump(final, f, indent=2)
    with open(OUT_DIR / "train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)

    print(f"\nDone. Best composite={best_score:.4f}")
    print(f"  baseline composite = {train_log[0]['composite']:.4f}")
    print(f"  best weights at    : {best_path}")


if __name__ == "__main__":
    main()
