"""
regression_to_tier.py — Bridge EfficientNetV2 price regression → value tier

For each subset (ja_natural, be_natural):
  1. Loads best regression checkpoint, runs inference on test split
  2. Maps predicted log(price) → price → tier using original percentile boundaries
  3. Computes macro F1 and per-class F1 vs ground-truth value_tier labels
  4. Prints comparison table vs direct classification (EfficientNetV2-S within-site)
  5. Saves to results/training/regression/regression_to_tier.json

Tier boundaries:
  JA natural: p25=$810, p75=$4,890, p90=$13,330
    (computed from all splits; 95.5% ground-truth reconstruction — original
     boundaries used a larger pre-dedup raw CSV, unavailable here)
  BE natural: p25=$990, p75=$4,510, p90=$9,300
    (from be_scraper/output/be_tier_stats.json; 99.1% reconstruction)
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── project imports ────────────────────────────────────────────────────────────
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
SRC_DIR    = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))
from models import get_model

# ── sklearn / scipy ────────────────────────────────────────────────────────────
from sklearn.metrics import f1_score, classification_report

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_ROOT / "data" / "splits"
CKPT_BASE    = PROJECT_ROOT / "results" / "training" / "regression" / "efficientnetv2"
OUT_DIR      = PROJECT_ROOT / "results" / "training" / "regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_DIRS = {
    "ja_natural": PROJECT_ROOT / "ja_scraper" / "output" / "images",
    "be_natural": PROJECT_ROOT / "be_scraper" / "output" / "images",
}

# ── tier boundaries ────────────────────────────────────────────────────────────
# Sources and reconstruction accuracy documented in script header.
TIER_BOUNDS = {
    "ja_natural": {"p25": 810.0,  "p75": 4890.0,  "p90": 13330.0},
    "be_natural": {"p25": 990.0,  "p75": 4510.0,  "p90":  9300.0},
}

# ── direct classification baselines (EfficientNetV2-S, within-site) ────────────
DIRECT_CLASSIFICATION_F1 = {
    "ja_natural": 0.6724,
    "be_natural": 0.6093,
}

# ── ImageNet transforms ─────────────────────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])

TIER_ORDER = ["budget", "mid_range", "premium", "investment_grade"]


# ── dataset ────────────────────────────────────────────────────────────────────

class RegressionInferenceDataset(Dataset):
    """Load test images; return (tensor, true_log_price, row_index)."""

    def __init__(self, csv_path: pathlib.Path, image_dir: pathlib.Path):
        df = pd.read_csv(csv_path, usecols=["diamond_id", "value_tier", "price_usd"])
        df["_log_price"] = np.log(df["price_usd"].astype(float))
        df["_img_path"]  = df.apply(
            lambda r: image_dir / str(r["value_tier"]) / f"{r['diamond_id']}.jpg",
            axis=1,
        )
        missing = ~df["_img_path"].apply(lambda p: p.exists())
        n_missing = int(missing.sum())
        if n_missing:
            print(f"  WARNING: {n_missing} images missing — skipping")
            df = df[~missing].reset_index(drop=True)
        self.records = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records.iloc[idx]
        target = torch.tensor(float(row["_log_price"]), dtype=torch.float32)
        try:
            img = Image.open(row["_img_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        return val_transform(img), target, idx


# ── model ──────────────────────────────────────────────────────────────────────

def build_model(dropout: float = 0.3) -> nn.Module:
    model, _ = get_model("efficientnetv2", dropout=dropout)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, 1),
    )
    return model


# ── tier assignment ────────────────────────────────────────────────────────────

def assign_tier(price: float, p25: float, p75: float, p90: float) -> str:
    if price <= p25:
        return "budget"
    elif price <= p75:
        return "mid_range"
    elif price <= p90:
        return "premium"
    else:
        return "investment_grade"


# ── inference ──────────────────────────────────────────────────────────────────

def run_inference(subset: str, device: torch.device) -> dict:
    print(f"\n{'='*60}")
    print(f"  {subset}")
    print(f"{'='*60}")

    bounds     = TIER_BOUNDS[subset]
    ckpt_path  = CKPT_BASE / subset / "best_model.pth"
    image_dir  = IMAGE_DIRS[subset]
    test_csv   = DATA_DIR / f"{subset}_test.csv"

    print(f"  Checkpoint : {ckpt_path.name}")
    print(f"  Tier bounds: p25=${bounds['p25']:,.0f}  p75=${bounds['p75']:,.0f}  p90=${bounds['p90']:,.0f}")

    # dataset
    dataset = RegressionInferenceDataset(test_csv, image_dir)
    loader  = DataLoader(dataset, batch_size=128, shuffle=False,
                         num_workers=8, pin_memory=True)
    print(f"  Test images: {len(dataset):,}")

    # model
    model = build_model(dropout=0.3).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_pred_log = np.zeros(len(dataset), dtype=np.float64)

    with torch.no_grad():
        for images, targets, indices in loader:
            images = images.to(device, non_blocking=True)
            preds  = model(images).squeeze(1).cpu().numpy()
            for i, pred in zip(indices.numpy(), preds):
                all_pred_log[i] = float(pred)

    # true labels (from CSV, aligned to dataset.records)
    true_tiers = dataset.records["value_tier"].tolist()

    # predicted tiers via boundary mapping
    pred_prices = np.exp(all_pred_log)
    pred_tiers  = [assign_tier(p, **bounds) for p in pred_prices]

    # metrics
    macro_f1 = f1_score(true_tiers, pred_tiers, average="macro", labels=TIER_ORDER, zero_division=0)
    per_class = f1_score(true_tiers, pred_tiers, average=None,    labels=TIER_ORDER, zero_division=0)

    print(f"\n  Macro F1 (regression→tier):  {macro_f1:.4f}")
    print(f"  Direct classification F1:    {DIRECT_CLASSIFICATION_F1[subset]:.4f}")
    print()
    print(classification_report(true_tiers, pred_tiers, labels=TIER_ORDER, zero_division=0))

    return {
        "subset":         subset,
        "n_test":         len(dataset),
        "tier_bounds":    bounds,
        "macro_f1":       macro_f1,
        "per_class_f1":   {t: float(f) for t, f in zip(TIER_ORDER, per_class)},
        "direct_clf_f1":  DIRECT_CLASSIFICATION_F1[subset],
    }


# ── comparison table ───────────────────────────────────────────────────────────

def print_comparison(results: dict) -> None:
    print("\n" + "=" * 68)
    print("SUMMARY: Regression-derived tiers vs Direct classification")
    print("=" * 68)
    print(f"{'Subset':<16} {'Reg→Tier F1':>14} {'Direct Clf F1':>14}  {'Delta':>10}")
    print(f"{'-'*16} {'-'*14} {'-'*14}  {'-'*10}")
    for subset, r in results.items():
        reg_f1  = r["macro_f1"]
        clf_f1  = r["direct_clf_f1"]
        delta   = reg_f1 - clf_f1
        sign    = "+" if delta >= 0 else ""
        print(f"{subset:<16} {reg_f1:>14.4f} {clf_f1:>14.4f}  {sign}{delta:>9.4f}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = {}
    for subset in ["ja_natural", "be_natural"]:
        results[subset] = run_inference(subset, device)

    print_comparison(results)

    out_path = OUT_DIR / "regression_to_tier.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
