"""
domain_image_stats.py — Pixel-level domain gap analysis: JA vs BE natural test splits.

Stratified-samples 2000 images per domain from the test split (never seen during
training), computes per-image channel statistics and resolution, then reports
aggregate distributional comparison including Cohen's d per channel.

Outputs:
  results/domain_analysis/image_stats.csv          (per-image rows)
  results/domain_analysis/image_stats_summary.json (paper-ready summary)
"""

from __future__ import annotations

import json
import math
import pathlib
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT     = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "splits"
OUT_DIR  = ROOT / "results" / "domain_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_DIRS = {
    "ja_natural": ROOT / "ja_scraper" / "output" / "images",
    "be_natural": ROOT / "be_scraper" / "output" / "images",
}

SAMPLE_N    = 2000
RANDOM_SEED = 42
MAX_WORKERS = 8

TIERS = ["budget", "mid_range", "premium", "investment_grade"]


# ── sampling ───────────────────────────────────────────────────────────────────

def stratified_sample(df: pd.DataFrame, n: int, rng: random.Random) -> pd.DataFrame:
    """
    Proportional stratified sample by value_tier.
    Rounds each tier's quota to the nearest integer; corrects total by
    adjusting the largest tier so sum == n exactly.
    """
    tier_counts = df["value_tier"].value_counts()
    total = len(df)
    quotas = {t: max(1, round(n * count / total)) for t, count in tier_counts.items()}

    # Fix rounding drift
    diff = sum(quotas.values()) - n
    if diff != 0:
        largest = max(quotas, key=lambda t: tier_counts[t])
        quotas[largest] -= diff

    sampled = []
    for tier, quota in quotas.items():
        tier_rows = df[df["value_tier"] == tier]
        k = min(quota, len(tier_rows))
        sampled.append(tier_rows.sample(n=k, random_state=rng.randint(0, 2**31)))

    result = pd.concat(sampled).reset_index(drop=True)
    return result, quotas


# ── per-image stats ────────────────────────────────────────────────────────────

def process_image(row: dict, image_dir: pathlib.Path) -> dict | None:
    """
    Load one image, compute per-channel pixel stats and resolution.
    Returns None if the file is missing or unreadable.
    """
    img_path = image_dir / str(row["value_tier"]) / f"{row['diamond_id']}.jpg"
    if not img_path.exists():
        return None
    try:
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)   # honour EXIF rotation
        img = img.convert("RGB")
        w, h = img.size                      # PIL: (width, height)
        arr  = np.asarray(img, dtype=np.float32)  # [H, W, 3]

        # Per-channel mean and std over all pixels
        r_mean, g_mean, b_mean = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
        r_std,  g_std,  b_std  = arr[:, :, 0].std(),  arr[:, :, 1].std(),  arr[:, :, 2].std()

        return {
            "diamond_id":  row["diamond_id"],
            "value_tier":  row["value_tier"],
            "width":       w,
            "height":      h,
            "aspect_ratio": round(w / h, 4),
            "r_mean": round(float(r_mean), 4),
            "g_mean": round(float(g_mean), 4),
            "b_mean": round(float(b_mean), 4),
            "r_std":  round(float(r_std),  4),
            "g_std":  round(float(g_std),  4),
            "b_std":  round(float(b_std),  4),
        }
    except Exception as e:
        print(f"  WARN: could not process {img_path.name}: {e}")
        return None


def compute_image_stats(rows: list[dict], image_dir: pathlib.Path, domain: str) -> list[dict]:
    """Process all rows in parallel; return list of stat dicts."""
    results = []
    failed  = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_image, r, image_dir): r for r in rows}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 200 == 0:
                print(f"  [{domain}] processed {done}/{len(rows)}...", flush=True)
            res = fut.result()
            if res is None:
                failed += 1
            else:
                results.append(res)
    if failed:
        print(f"  [{domain}] {failed} images skipped (missing or unreadable)")
    return results


# ── aggregation ────────────────────────────────────────────────────────────────

def agg_stats(df: pd.DataFrame) -> dict:
    """Compute aggregate statistics for a single domain's per-image DataFrame."""
    stats = {}

    for ch in ["r", "g", "b"]:
        m_col = f"{ch}_mean"
        s_col = f"{ch}_std"
        stats[f"{ch}_mean_of_means"] = float(df[m_col].mean())
        stats[f"{ch}_std_of_means"]  = float(df[m_col].std())
        stats[f"{ch}_mean_of_stds"]  = float(df[s_col].mean())
        stats[f"{ch}_std_of_stds"]   = float(df[s_col].std())

    # Resolution
    resolutions = list(zip(df["height"], df["width"]))
    res_strs    = [f"{h}x{w}" for h, w in resolutions]
    res_counter = Counter(res_strs)
    stats["resolution_min_h"]    = int(df["height"].min())
    stats["resolution_max_h"]    = int(df["height"].max())
    stats["resolution_mean_h"]   = float(df["height"].mean())
    stats["resolution_min_w"]    = int(df["width"].min())
    stats["resolution_max_w"]    = int(df["width"].max())
    stats["resolution_mean_w"]   = float(df["width"].mean())
    stats["resolution_most_common"] = res_counter.most_common(3)

    # Aspect ratio
    stats["aspect_ratio_mean"] = float(df["aspect_ratio"].mean())
    stats["aspect_ratio_std"]  = float(df["aspect_ratio"].std())
    stats["aspect_ratio_min"]  = float(df["aspect_ratio"].min())
    stats["aspect_ratio_max"]  = float(df["aspect_ratio"].max())
    stats["n_images"]          = len(df)

    return stats


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-variance Cohen's d: (mean_a - mean_b) / pooled_std."""
    na, nb  = len(a), len(b)
    var_a, var_b = float(a.var(ddof=1)), float(b.var(ddof=1))
    pooled_std = math.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


# ── printing ───────────────────────────────────────────────────────────────────

def print_summary(agg: dict, cohens: dict) -> None:
    ja, be = agg["ja_natural"], agg["be_natural"]

    print("\n" + "=" * 72)
    print("DOMAIN IMAGE STATISTICS  —  JA Natural vs BE Natural  (n=2000 each)")
    print("=" * 72)

    # Channel stats
    print(f"\n{'':20s} {'JA natural':>22} {'BE natural':>22}  {'Cohen d':>8}")
    print(f"{'':20s} {'mean ± std':>22} {'mean ± std':>22}  {'':>8}")
    print("-" * 76)
    for ch, label in [("r", "R channel mean"), ("g", "G channel mean"), ("b", "B channel mean")]:
        ja_m  = ja[f"{ch}_mean_of_means"];  ja_s  = ja[f"{ch}_std_of_means"]
        be_m  = be[f"{ch}_mean_of_means"];  be_s  = be[f"{ch}_std_of_means"]
        d     = cohens[f"{ch}_mean"]
        print(f"  {label:<18s}  {ja_m:7.2f} ± {ja_s:6.2f}        {be_m:7.2f} ± {be_s:6.2f}  {d:>8.3f}")

    print()
    for ch, label in [("r", "R channel std"), ("g", "G channel std"), ("b", "B channel std")]:
        ja_m  = ja[f"{ch}_mean_of_stds"];  ja_s  = ja[f"{ch}_std_of_stds"]
        be_m  = be[f"{ch}_mean_of_stds"];  be_s  = be[f"{ch}_std_of_stds"]
        print(f"  {label:<18s}  {ja_m:7.2f} ± {ja_s:6.2f}        {be_m:7.2f} ± {be_s:6.2f}")

    # Resolution
    print(f"\n  {'Resolution (H×W)':<18s}")
    print(f"    JA:  mean {ja['resolution_mean_h']:.0f}×{ja['resolution_mean_w']:.0f}"
          f"  range [{ja['resolution_min_h']}×{ja['resolution_min_w']} – "
          f"{ja['resolution_max_h']}×{ja['resolution_max_w']}]")
    print(f"    BE:  mean {be['resolution_mean_h']:.0f}×{be['resolution_mean_w']:.0f}"
          f"  range [{be['resolution_min_h']}×{be['resolution_min_w']} – "
          f"{be['resolution_max_h']}×{be['resolution_max_w']}]")
    print(f"    JA most common: {ja['resolution_most_common']}")
    print(f"    BE most common: {be['resolution_most_common']}")

    # Aspect ratio
    print(f"\n  {'Aspect ratio (W/H)':<18s}")
    print(f"    JA:  {ja['aspect_ratio_mean']:.4f} ± {ja['aspect_ratio_std']:.4f}"
          f"  [{ja['aspect_ratio_min']:.4f} – {ja['aspect_ratio_max']:.4f}]")
    print(f"    BE:  {be['aspect_ratio_mean']:.4f} ± {be['aspect_ratio_std']:.4f}"
          f"  [{be['aspect_ratio_min']:.4f} – {be['aspect_ratio_max']:.4f}]")
    print()
    print("  Cohen's d interpretation: |d| < 0.2 small, 0.2–0.8 medium, > 0.8 large")
    print("=" * 72)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    rng = random.Random(RANDOM_SEED)
    all_per_image: list[pd.DataFrame] = []
    agg: dict[str, dict] = {}

    for domain in ["ja_natural", "be_natural"]:
        print(f"\n── {domain} ──────────────────────────────────────────────────")
        df_full = pd.read_csv(
            DATA_DIR / f"{domain}_test.csv",
            usecols=["diamond_id", "value_tier"],
        )
        sampled_df, quotas = stratified_sample(df_full, SAMPLE_N, rng)
        print(f"  Stratified sample: {len(sampled_df)} rows  |  quotas: {quotas}")

        rows = sampled_df.to_dict("records")
        stats_list = compute_image_stats(rows, IMAGE_DIRS[domain], domain)

        per_image_df = pd.DataFrame(stats_list)
        per_image_df.insert(0, "domain", domain)
        all_per_image.append(per_image_df)

        agg[domain] = agg_stats(per_image_df)
        print(f"  OK: {len(per_image_df)} images processed")

    # Per-image CSV
    combined = pd.concat(all_per_image, ignore_index=True)
    csv_path = OUT_DIR / "image_stats.csv"
    combined.to_csv(csv_path, index=False)
    print(f"\nPer-image CSV: {csv_path}  ({len(combined)} rows)")

    # Cohen's d for each channel mean
    ja_df = combined[combined["domain"] == "ja_natural"]
    be_df = combined[combined["domain"] == "be_natural"]
    cohens = {}
    for ch in ["r", "g", "b"]:
        cohens[f"{ch}_mean"] = cohens_d(
            ja_df[f"{ch}_mean"].values,
            be_df[f"{ch}_mean"].values,
        )

    print_summary(agg, cohens)

    # Summary JSON
    summary = {
        "sample_n_per_domain": SAMPLE_N,
        "random_seed":         RANDOM_SEED,
        "domains": {
            domain: {
                "n_processed":       agg[domain]["n_images"],
                "channel_means": {
                    ch: {
                        "mean_of_means": agg[domain][f"{ch}_mean_of_means"],
                        "std_of_means":  agg[domain][f"{ch}_std_of_means"],
                    }
                    for ch in ["r", "g", "b"]
                },
                "channel_stds": {
                    ch: {
                        "mean_of_stds": agg[domain][f"{ch}_mean_of_stds"],
                        "std_of_stds":  agg[domain][f"{ch}_std_of_stds"],
                    }
                    for ch in ["r", "g", "b"]
                },
                "resolution": {
                    "mean_hw":   [round(agg[domain]["resolution_mean_h"], 1),
                                  round(agg[domain]["resolution_mean_w"], 1)],
                    "min_hw":    [agg[domain]["resolution_min_h"],
                                  agg[domain]["resolution_min_w"]],
                    "max_hw":    [agg[domain]["resolution_max_h"],
                                  agg[domain]["resolution_max_w"]],
                    "most_common_top3": agg[domain]["resolution_most_common"],
                },
                "aspect_ratio": {
                    "mean": agg[domain]["aspect_ratio_mean"],
                    "std":  agg[domain]["aspect_ratio_std"],
                    "min":  agg[domain]["aspect_ratio_min"],
                    "max":  agg[domain]["aspect_ratio_max"],
                },
            }
            for domain in ["ja_natural", "be_natural"]
        },
        "cohens_d_channel_means": {
            ch: round(cohens[f"{ch}_mean"], 4)
            for ch in ["r", "g", "b"]
        },
        "cohens_d_note": (
            "d = (mean_JA - mean_BE) / pooled_std of per-image channel means. "
            "|d| < 0.2 small, 0.2-0.8 medium, >0.8 large."
        ),
    }

    json_path = OUT_DIR / "image_stats_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Summary JSON: {json_path}")


if __name__ == "__main__":
    main()
