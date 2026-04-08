"""
brightness_error_correlation.py

Tests whether per-image brightness (r_mean from image_stats.csv) correlates
with classification correctness on the 2000 sampled BE natural images, using
the cross-domain EfficientNetV2 checkpoint (trained on JA, tested on BE).

No predictions CSV exists in the results directory, so inference is run here
on the fly for just the 2000 sampled images.

Outputs:
  results/domain_analysis/brightness_error_corr.json
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from scipy.stats import spearmanr, pointbiserialr
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── project imports ────────────────────────────────────────────────────────────
ROOT    = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
from models import get_model, TIER_LABELS  # alphabetical: budget, investment_grade, mid_range, premium

CKPT    = ROOT / "results" / "training" / "efficientnetv2__ja_natural__cross" / "best_model.pth"
STATS   = ROOT / "results" / "domain_analysis" / "image_stats.csv"
IMG_DIR = ROOT / "be_scraper" / "output" / "images"
OUT_DIR = ROOT / "results" / "domain_analysis"

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


# ── dataset ────────────────────────────────────────────────────────────────────

class BESubsetDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        path = IMG_DIR / r["value_tier"] / f"{r['diamond_id']}.jpg"
        try:
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        return TRANSFORM(img), idx


# ── inference ──────────────────────────────────────────────────────────────────

def run_inference(records: list[dict], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns pred_labels (int indices into TIER_LABELS) and
    correct_class_probs (softmax probability of the true class) for each record.
    """
    model, _ = get_model("efficientnetv2", dropout=0.3)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.to(device).eval()

    dataset = BESubsetDataset(records)
    loader  = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

    pred_labels    = np.zeros(len(records), dtype=np.int32)
    correct_probs  = np.zeros(len(records), dtype=np.float32)

    tier_to_idx = {t: i for i, t in enumerate(TIER_LABELS)}

    with torch.no_grad():
        for images, indices in loader:
            images  = images.to(device, non_blocking=True)
            logits  = model(images)                           # [B, 4]
            probs   = torch.softmax(logits, dim=1).cpu().numpy()
            preds   = logits.argmax(dim=1).cpu().numpy()
            for i, (pred, prob_row, orig_idx) in enumerate(zip(preds, probs, indices.numpy())):
                pred_labels[orig_idx]   = int(pred)
                true_idx = tier_to_idx[records[orig_idx]["value_tier"]]
                correct_probs[orig_idx] = float(prob_row[true_idx])

    return pred_labels, correct_probs


# ── analysis ───────────────────────────────────────────────────────────────────

def analyze(df: pd.DataFrame, pred_labels: np.ndarray, correct_probs: np.ndarray) -> dict:
    tier_to_idx = {t: i for i, t in enumerate(TIER_LABELS)}

    true_idx = df["value_tier"].map(tier_to_idx).values
    correct  = (pred_labels == true_idx).astype(int)   # binary: 1 = correct

    results = {
        "n":           len(df),
        "n_correct":   int(correct.sum()),
        "accuracy":    float(correct.mean()),
        "pred_distribution": {
            t: int((pred_labels == i).sum()) for i, t in enumerate(TIER_LABELS)
        },
        "true_distribution": {
            t: int((true_idx == i).sum()) for i, t in enumerate(TIER_LABELS)
        },
    }

    # ── primary correlations ───────────────────────────────────────────────────
    channel_corrs = {}
    for ch in ["r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std"]:
        vals = df[ch].values
        sr, sp   = spearmanr(vals, correct)
        pb, pbp  = pointbiserialr(correct, vals)
        channel_corrs[ch] = {
            "spearman_rho":  round(float(sr),  4),
            "spearman_p":    round(float(sp),  4),
            "point_biserial_r": round(float(pb),  4),
            "point_biserial_p": round(float(pbp), 4),
        }

    # correlation with correct-class softmax probability (continuous, more sensitive)
    sr_prob, sp_prob = spearmanr(df["r_mean"].values, correct_probs)
    channel_corrs["r_mean_vs_correct_class_prob"] = {
        "spearman_rho": round(float(sr_prob), 4),
        "spearman_p":   round(float(sp_prob), 4),
        "note": "Spearman between r_mean and softmax prob of true class (continuous signal)",
    }

    results["channel_vs_correct"] = channel_corrs

    # ── per-tier breakdown ─────────────────────────────────────────────────────
    # How does r_mean differ between correctly and incorrectly classified images
    # within each tier?
    tier_breakdown = {}
    for tier in TIER_LABELS:
        mask = df["value_tier"].values == tier
        if mask.sum() < 10:
            continue
        tier_correct = correct[mask]
        tier_r       = df.loc[mask, "r_mean"].values
        n_c  = int(tier_correct.sum())
        n_w  = int((~tier_correct.astype(bool)).sum())
        r_correct = tier_r[tier_correct.astype(bool)]
        r_wrong   = tier_r[~tier_correct.astype(bool)]
        entry = {
            "n_total":   int(mask.sum()),
            "n_correct": n_c,
            "n_wrong":   n_w,
            "accuracy":  round(float(tier_correct.mean()), 4),
            "r_mean_correct_images": round(float(r_correct.mean()), 2) if len(r_correct) > 0 else None,
            "r_mean_wrong_images":   round(float(r_wrong.mean()),   2) if len(r_wrong)   > 0 else None,
        }
        if len(r_correct) > 1 and len(r_wrong) > 1:
            sr_t, sp_t = spearmanr(tier_r, tier_correct)
            entry["spearman_rho"] = round(float(sr_t), 4)
            entry["spearman_p"]   = round(float(sp_t), 4)
        tier_breakdown[tier] = entry

    results["per_tier"] = tier_breakdown
    return results


# ── printing ───────────────────────────────────────────────────────────────────

def print_results(results: dict) -> None:
    print("\n" + "=" * 68)
    print("BRIGHTNESS vs CLASSIFICATION ERROR  —  BE Natural (n=2000)")
    print("Cross-domain model: EfficientNetV2 trained on JA, tested on BE")
    print("=" * 68)

    acc = results["accuracy"]
    print(f"\n  Overall accuracy on these 2000 images: {acc:.4f}  ({results['n_correct']}/{results['n']})")

    print("\n  Prediction distribution (model output):")
    for t, n in results["pred_distribution"].items():
        bar = "█" * int(40 * n / results["n"])
        print(f"    {t:<20s}  {n:>5}  ({100*n/results['n']:5.1f}%)  {bar}")

    print("\n  True tier distribution:")
    for t, n in results["true_distribution"].items():
        print(f"    {t:<20s}  {n:>5}  ({100*n/results['n']:5.1f}%)")

    print("\n  ── Spearman & Point-Biserial:  channel stat vs correct(0/1) ──────")
    print(f"  {'Feature':<30s}  {'Spearman ρ':>12}  {'p-value':>10}  {'PB r':>8}  {'p-value':>10}")
    print("  " + "-" * 76)
    for ch, v in results["channel_vs_correct"].items():
        if ch == "r_mean_vs_correct_class_prob":
            continue
        sig = "*" if v["spearman_p"] < 0.05 else " "
        print(f"  {ch:<30s}  {v['spearman_rho']:>12.4f}  {v['spearman_p']:>10.4f}{sig}"
              f"  {v['point_biserial_r']:>8.4f}  {v['point_biserial_p']:>10.4f}{sig}")

    # continuous signal
    v_prob = results["channel_vs_correct"]["r_mean_vs_correct_class_prob"]
    print(f"\n  r_mean vs correct-class softmax prob (continuous):")
    print(f"    Spearman ρ = {v_prob['spearman_rho']:.4f}   p = {v_prob['spearman_p']:.4f}")

    print("\n  ── Per-tier: r_mean of correct vs wrong images ─────────────────")
    print(f"  {'Tier':<20s}  {'Acc':>6}  {'n':>5}  {'r_mean correct':>15}  {'r_mean wrong':>13}  {'ρ':>7}  {'p':>8}")
    print("  " + "-" * 82)
    for tier, v in results["per_tier"].items():
        rc = f"{v['r_mean_correct_images']:.2f}" if v["r_mean_correct_images"] is not None else "  N/A"
        rw = f"{v['r_mean_wrong_images']:.2f}"   if v["r_mean_wrong_images"]   is not None else "  N/A"
        rho = f"{v.get('spearman_rho', float('nan')):.4f}"
        p   = f"{v.get('spearman_p',   float('nan')):.4f}"
        print(f"  {tier:<20s}  {v['accuracy']:>6.4f}  {v['n_total']:>5}  {rc:>15}  {rw:>13}  {rho:>7}  {p:>8}")

    print()
    print("  * p < 0.05")
    print("=" * 68)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {CKPT}")

    # Load BE rows from image_stats
    stats_df = pd.read_csv(STATS)
    be_df    = stats_df[stats_df["domain"] == "be_natural"].reset_index(drop=True)
    print(f"BE images in image_stats.csv: {len(be_df)}")

    records = be_df[["diamond_id", "value_tier"]].to_dict("records")
    print("Running inference on 2000 BE images...")
    pred_labels, correct_probs = run_inference(records, device)

    results = analyze(be_df, pred_labels, correct_probs)
    print_results(results)

    # save
    out_path = OUT_DIR / "brightness_error_corr.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
