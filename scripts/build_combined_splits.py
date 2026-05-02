"""
build_combined_splits.py — Build normalized combined Stage 2 train/val/test CSVs.

Creates three combined subsets:
  combined_natural  — ja_natural + be_natural  (subsample to balance sites)
  combined_lab      — ja_lab + be_lab          (subsample to balance sites)
  combined_all      — all four subsets         (subsample to balance sites)

Price normalization for regression:
  normalized_log_price = (log(price_usd) - subset_mean) / subset_std
  Computed on the train split of each source subset, applied to all splits.

Tier labels (classification) are kept as-is — they already represent
relative value within each subset's price distribution.

Outputs to data/splits/:
  combined_natural_{train|val|test}.csv
  combined_lab_{train|val|test}.csv
  combined_all_{train|val|test}.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
SPLITS    = ROOT / "data" / "splits"
SEED      = 42

SUBSETS = ["ja_natural", "be_natural", "ja_lab", "be_lab"]
SPLITS_TO_BUILD = ["train", "val", "test"]

COMBINATIONS = {
    "combined_natural": ["ja_natural", "be_natural"],
    "combined_lab":     ["ja_lab",     "be_lab"],
    "combined_all":     ["ja_natural", "be_natural", "ja_lab", "be_lab"],
}


def load_split(subset: str, split: str) -> pd.DataFrame:
    path = SPLITS / f"{subset}_{split}.csv"
    df = pd.read_csv(path, low_memory=False)
    df["source_subset"] = subset
    return df


def compute_log_price_stats(train_df: pd.DataFrame) -> tuple[float, float]:
    log_prices = np.log(train_df["price_usd"].clip(lower=1))
    return float(log_prices.mean()), float(log_prices.std())


def normalize_log_price(df: pd.DataFrame, mean: float, std: float) -> pd.DataFrame:
    df = df.copy()
    df["normalized_log_price"] = (np.log(df["price_usd"].clip(lower=1)) - mean) / std
    return df


def subsample_to_smallest(dfs: list[pd.DataFrame], split: str, rng) -> list[pd.DataFrame]:
    """Subsample each df to the size of the smallest one (train/val only)."""
    if split == "test":
        return dfs
    min_size = min(len(df) for df in dfs)
    return [df.sample(n=min_size, random_state=rng).reset_index(drop=True)
            if len(df) > min_size else df.reset_index(drop=True)
            for df in dfs]


def main() -> None:
    rng = np.random.default_rng(SEED)

    # ── Compute per-subset log-price stats from train splits ──────────────────
    print("Computing per-subset log-price normalization stats...")
    stats: dict[str, tuple[float, float]] = {}
    for subset in SUBSETS:
        train_df = load_split(subset, "train")
        mean, std = compute_log_price_stats(train_df)
        stats[subset] = (mean, std)
        print(f"  {subset:<14s}: log_price mean={mean:.4f}  std={std:.4f}")

    # ── Build each combined subset ────────────────────────────────────────────
    for combined_name, source_subsets in COMBINATIONS.items():
        print(f"\nBuilding {combined_name} ({' + '.join(source_subsets)})...")

        for split in SPLITS_TO_BUILD:
            dfs = []
            for subset in source_subsets:
                df = load_split(subset, split)
                mean, std = stats[subset]
                df = normalize_log_price(df, mean, std)
                dfs.append(df)

            # Subsample train/val to balance site contributions
            dfs = subsample_to_smallest(dfs, split, rng)

            combined = pd.concat(dfs, ignore_index=True)
            if split == "train":
                combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

            out_path = SPLITS / f"{combined_name}_{split}.csv"
            combined.to_csv(out_path, index=False)
            print(f"  {split:<6s}: {len(combined):>7,} rows → {out_path.name}")

            if split == "train":
                print(f"          source breakdown:")
                for subset in source_subsets:
                    n = (combined["source_subset"] == subset).sum()
                    print(f"            {subset}: {n:,}")

    print("\nDone. Combined splits saved to data/splits/")


if __name__ == "__main__":
    main()
