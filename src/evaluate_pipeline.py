"""
evaluate_pipeline.py — End-to-end pipeline evaluation

Stage 1: EfficientNetV2 gem species classifier (68 classes)
         Input: any gem image
         Output: predicted species

Stage 2: EfficientNetV2 regression model (combined_all)
         Input: images Stage 1 identified as Diamond
         Output: predicted normalized log-price → tier + USD price estimate

Test set composition:
  - Stage 2 combined_all test split: ~89,557 diamond images (natural + lab, ja + be)
  - Stage 1 test split non-diamond images: 1,767 images across 67 other species

Reports:
  - Stage 1 diamond recall     : % of true diamonds correctly identified as Diamond
  - Stage 1 false positive rate: % of non-diamonds incorrectly called Diamond
  - Stage 2 tier accuracy and macro F1 on Stage-1-passed images
  - Stage 2 USD price error: mean absolute error and median absolute % error per subset
  - End-to-end accuracy (Stage 1 miss = wrong answer)
  - Per-subset breakdown

Usage:
    python src/evaluate_pipeline.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_v2_s

ROOT           = Path(__file__).resolve().parent.parent
SPLITS         = ROOT / "data" / "splits"
JA_IMGS        = ROOT / "ja_scraper" / "output" / "images"
BE_IMGS        = ROOT / "be_scraper" / "output" / "images"
STAGE1_DATA    = ROOT / "data" / "Combined-P1-Dataset" / "test"
STAGE1_WEIGHTS = ROOT / "results" / "training" / "stage1" / "efficientnetv2" / "best_model.pth"
STAGE2_WEIGHTS = ROOT / "results" / "training" / "regression" / "efficientnetv2" / "combined_all" / "best_model.pth"
OUT_DIR        = ROOT / "results" / "pipeline_eval"

DROPOUT     = 0.3
NUM_STAGE1  = 68
BATCH_SIZE  = 64
NUM_WORKERS = 8

TIER_LABELS = ["budget", "investment_grade", "mid_range", "premium"]

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


# ── Normalization stats (computed from train split, used to recover USD price) ─

def compute_norm_stats() -> dict[str, dict[str, float]]:
    """Per-subset log-price mean and std from training data for denormalization."""
    df = pd.read_csv(SPLITS / "combined_all_train.csv",
                     usecols=["source_subset", "price_usd"], low_memory=False)
    stats = {}
    for subset, grp in df.groupby("source_subset"):
        log_prices = np.log(grp["price_usd"].values)
        stats[subset] = {"mean": float(log_prices.mean()), "std": float(log_prices.std())}
    return stats


def denormalize_to_usd(norm_pred: float, subset: str, norm_stats: dict) -> float:
    """Convert normalized log-price prediction back to USD."""
    s = norm_stats[subset]
    log_price = norm_pred * s["std"] + s["mean"]
    return float(np.exp(log_price))


# ── Datasets ──────────────────────────────────────────────────────────────────

class DiamondRetailerDataset(Dataset):
    """Stage 2 test images — all diamonds (natural + lab, ja + be) with price labels."""
    def __init__(self, csv_path: Path):
        df = pd.read_csv(csv_path, low_memory=False)
        df["_img_dir"] = df["source_subset"].apply(
            lambda s: JA_IMGS if s.startswith("ja") else BE_IMGS
        )
        df["_img_path"] = df.apply(
            lambda r: Path(r["_img_dir"]) / str(r["value_tier"]) / f"{int(r['diamond_id'])}.jpg",
            axis=1,
        )
        missing = ~df["_img_path"].apply(lambda p: p.exists())
        if missing.sum() > 0:
            print(f"  Warning: {missing.sum()} images not found, skipping.")
            df = df[~missing].reset_index(drop=True)
        self.records = df[["diamond_id", "source_subset", "tier_label",
                            "price_usd", "normalized_log_price", "_img_path"]].copy()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        img = Image.open(row["_img_path"]).convert("RGB")
        return (transform(img), row["tier_label"], row["source_subset"],
                "diamond", float(row["price_usd"]))


class NonDiamondDataset(Dataset):
    """Stage 1 test images — non-diamond species only."""
    def __init__(self, test_dir: Path):
        self.samples = []
        for species_dir in sorted(test_dir.iterdir()):
            if not species_dir.is_dir() or species_dir.name == "Diamond":
                continue
            for img_path in species_dir.glob("*.jpg"):
                self.samples.append((img_path, species_dir.name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, species = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return transform(img), "non_diamond", species, "non_diamond", -1.0


def collate_fn(batch):
    imgs, tiers, subsets, types, prices = zip(*batch)
    return torch.stack(imgs), list(tiers), list(subsets), list(types), list(prices)


# ── Model builders ────────────────────────────────────────────────────────────

def build_stage1(weights_path: Path, device: torch.device) -> tuple[nn.Module, list[str]]:
    model = efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(in_features, NUM_STAGE1),
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    classes = sorted([d.name for d in (ROOT / "data" / "Combined-P1-Dataset" / "train").iterdir() if d.is_dir()])
    return model.to(device).eval(), classes


def build_stage2(weights_path: Path, device: torch.device) -> nn.Module:
    model = efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(in_features, 1),
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.to(device).eval()


def compute_tier_thresholds() -> list[float]:
    df = pd.read_csv(SPLITS / "combined_all_train.csv",
                     usecols=["value_tier", "normalized_log_price"], low_memory=False)
    tier_means = df.groupby("value_tier")["normalized_log_price"].mean().sort_values()
    means = tier_means.values.tolist()
    return [(means[i] + means[i+1]) / 2 for i in range(len(means) - 1)]


def pred_to_tier(val: float, thresholds: list[float]) -> int:
    return sum(val > t for t in thresholds)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(loader, stage1, stage2, classes, diamond_idx,
                  thresholds, norm_stats, device):
    results = []
    with torch.no_grad():
        for batch_idx, (imgs, true_tiers, subsets, sample_types, true_prices) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)

            s1_logits    = stage1(imgs)
            s1_preds     = s1_logits.argmax(dim=1).cpu().numpy()
            diamond_mask = s1_preds == diamond_idx
            diamond_imgs = imgs[diamond_mask]

            s2_norm_preds = [None] * len(imgs)
            if diamond_mask.sum() > 0:
                s2_raw = stage2(diamond_imgs).squeeze(1).cpu().numpy()
                di = 0
                for i, is_diamond in enumerate(diamond_mask):
                    if is_diamond:
                        s2_norm_preds[i] = float(s2_raw[di])
                        di += 1

            for i in range(len(imgs)):
                norm_pred = s2_norm_preds[i]
                subset    = subsets[i]
                pred_usd  = None
                pred_tier = None
                if norm_pred is not None:
                    pred_tier = TIER_LABELS[pred_to_tier(norm_pred, thresholds)]
                    if subset in norm_stats:
                        pred_usd = denormalize_to_usd(norm_pred, subset, norm_stats)

                results.append({
                    "sample_type":   sample_types[i],
                    "subset":        subset,
                    "true_tier":     true_tiers[i],
                    "true_price_usd": true_prices[i],
                    "s1_pred_class": classes[s1_preds[i]],
                    "s1_is_diamond": bool(diamond_mask[i]),
                    "s2_pred_tier":  pred_tier,
                    "s2_pred_usd":   pred_usd,
                })

            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(loader)}")

    return pd.DataFrame(results)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Stage 1 model...")
    stage1, classes = build_stage1(STAGE1_WEIGHTS, device)
    diamond_idx = classes.index("Diamond")
    print(f"  {len(classes)} classes  |  Diamond index: {diamond_idx}")

    print("Loading Stage 2 model...")
    stage2     = build_stage2(STAGE2_WEIGHTS, device)
    thresholds = compute_tier_thresholds()
    norm_stats = compute_norm_stats()
    print(f"  Tier thresholds: {[f'{t:.4f}' for t in thresholds]}")
    print(f"  Norm stats loaded for subsets: {list(norm_stats.keys())}")

    print("Building datasets...")
    diamond_ds     = DiamondRetailerDataset(SPLITS / "combined_all_test.csv")
    non_diamond_ds = NonDiamondDataset(STAGE1_DATA)
    n_species      = len(set(s for _, s in non_diamond_ds.samples))
    print(f"  Diamond images    : {len(diamond_ds):,}")
    print(f"  Non-diamond images: {len(non_diamond_ds):,}  ({n_species} species)")

    loader = DataLoader(ConcatDataset([diamond_ds, non_diamond_ds]),
                        batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    print(f"  Total: {len(diamond_ds) + len(non_diamond_ds):,} images")

    print("\nRunning pipeline inference...")
    df = run_inference(loader, stage1, stage2, classes, diamond_idx,
                       thresholds, norm_stats, device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    diamonds     = df[df["sample_type"] == "diamond"].copy()
    non_diamonds = df[df["sample_type"] == "non_diamond"].copy()

    s1_recall = diamonds["s1_is_diamond"].mean()
    s1_fpr    = non_diamonds["s1_is_diamond"].mean()

    passed  = diamonds[diamonds["s1_is_diamond"]].copy()
    s2_acc  = (passed["s2_pred_tier"] == passed["true_tier"]).mean() if len(passed) > 0 else 0
    s2_f1   = float(f1_score(passed["true_tier"], passed["s2_pred_tier"],
                              average="macro", zero_division=0)) if len(passed) > 0 else 0
    e2e_acc = (diamonds["s2_pred_tier"] == diamonds["true_tier"]).mean()

    # USD price metrics on passed images that have a valid prediction
    passed_priced = passed[passed["s2_pred_usd"].notna() & (passed["true_price_usd"] > 0)].copy()
    usd_mae  = float(np.mean(np.abs(passed_priced["s2_pred_usd"] - passed_priced["true_price_usd"])))
    usd_mape = float(np.median(np.abs(
        (passed_priced["s2_pred_usd"] - passed_priced["true_price_usd"]) / passed_priced["true_price_usd"]
    ) * 100))

    print("\n" + "="*60)
    print("PIPELINE EVALUATION RESULTS")
    print("="*60)

    print(f"\n--- Stage 1: Diamond Detection ---")
    print(f"  True diamonds      : {len(diamonds):,}")
    print(f"  Non-diamonds       : {len(non_diamonds):,}")
    print(f"  Diamond recall     : {s1_recall:.1%}  ({int(diamonds['s1_is_diamond'].sum()):,} / {len(diamonds):,} caught)")
    print(f"  False positive rate: {s1_fpr:.1%}  ({int(non_diamonds['s1_is_diamond'].sum()):,} / {len(non_diamonds):,} non-diamonds called Diamond)")

    print(f"\n--- Stage 2: Price Tier (Stage-1-passed images, n={len(passed):,}) ---")
    print(f"  Accuracy : {s2_acc:.4f}")
    print(f"  Macro F1 : {s2_f1:.4f}")
    print()
    if len(passed) > 0:
        print(classification_report(passed["true_tier"], passed["s2_pred_tier"],
                                     labels=TIER_LABELS, zero_division=0))

    print(f"--- Stage 2: USD Price Estimation (n={len(passed_priced):,}) ---")
    print(f"  Mean absolute error    : ${usd_mae:,.0f}")
    print(f"  Median absolute % error: {usd_mape:.1f}%")

    print(f"\n--- Per-subset USD price error ---")
    for subset in sorted(passed_priced["subset"].unique()):
        sub = passed_priced[passed_priced["subset"] == subset]
        mae  = float(np.mean(np.abs(sub["s2_pred_usd"] - sub["true_price_usd"])))
        mape = float(np.median(np.abs(
            (sub["s2_pred_usd"] - sub["true_price_usd"]) / sub["true_price_usd"]
        ) * 100))
        print(f"  {subset:20s}  n={len(sub):6,}  MAE=${mae:8,.0f}  MdAPE={mape:.1f}%")

    print(f"\n--- End-to-end (Stage 1 miss = wrong) ---")
    print(f"  Tier accuracy: {e2e_acc:.1%}  ({int((diamonds['s2_pred_tier'] == diamonds['true_tier']).sum()):,} / {len(diamonds):,})")

    print(f"\n--- Per-subset breakdown (diamonds only) ---")
    for subset in sorted(diamonds["subset"].unique()):
        sub        = diamonds[diamonds["subset"] == subset]
        sub_passed = sub[sub["s1_is_diamond"]]
        s1_r  = sub["s1_is_diamond"].mean()
        s2_f  = float(f1_score(sub_passed["true_tier"], sub_passed["s2_pred_tier"],
                                average="macro", zero_division=0)) if len(sub_passed) > 0 else 0
        e2e   = (sub["s2_pred_tier"] == sub["true_tier"]).mean()
        print(f"  {subset:20s}  n={len(sub):6,}  S1_recall={s1_r:.1%}  S2_F1={s2_f:.4f}  E2E={e2e:.1%}")

    # ── Save ──────────────────────────────────────────────────────────────────
    summary = {
        "total_diamond_images":       len(diamonds),
        "total_nondiamond_images":    len(non_diamonds),
        "stage1_diamond_recall":      float(s1_recall),
        "stage1_false_positive_rate": float(s1_fpr),
        "stage2_accuracy":            float(s2_acc),
        "stage2_macro_f1":            float(s2_f1),
        "stage2_usd_mae":             usd_mae,
        "stage2_usd_median_ape":      usd_mape,
        "end_to_end_accuracy":        float(e2e_acc),
        "per_subset": {}
    }
    for subset in sorted(diamonds["subset"].unique()):
        sub        = diamonds[diamonds["subset"] == subset]
        sub_passed = sub[sub["s1_is_diamond"]]
        sub_priced = sub_passed[sub_passed["s2_pred_usd"].notna() & (sub_passed["true_price_usd"] > 0)]
        summary["per_subset"][subset] = {
            "n":                len(sub),
            "stage1_recall":    float(sub["s1_is_diamond"].mean()),
            "stage2_macro_f1":  float(f1_score(sub_passed["true_tier"], sub_passed["s2_pred_tier"],
                                               average="macro", zero_division=0)) if len(sub_passed) > 0 else 0,
            "end_to_end_acc":   float((sub["s2_pred_tier"] == sub["true_tier"]).mean()),
            "usd_mae":          float(np.mean(np.abs(sub_priced["s2_pred_usd"] - sub_priced["true_price_usd"]))) if len(sub_priced) > 0 else None,
            "usd_median_ape":   float(np.median(np.abs((sub_priced["s2_pred_usd"] - sub_priced["true_price_usd"]) / sub_priced["true_price_usd"]) * 100)) if len(sub_priced) > 0 else None,
        }

    with open(OUT_DIR / "pipeline_eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    df.to_csv(OUT_DIR / "pipeline_eval_detail.csv", index=False)
    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
