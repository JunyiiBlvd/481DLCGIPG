"""
build_combined_dataset.py — Build a unified manifest CSV for full pipeline evaluation.

Combines:
  - Stage 1 Kaggle test images (68 gemstone classes, no tier label)
  - Stage 2 diamond test images (JA/BE splits, with tier labels)

Output: data/combined_pipeline_manifest.csv
Columns:
  image_path   — absolute path to image file
  species      — gemstone class name (e.g. "Diamond", "Ruby")
  value_tier   — tier label or N/A (Stage 1 images have no tier)
  subset       — source subset identifier
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
SPLITS    = DATA_DIR / "splits"
STAGE1    = DATA_DIR / "Combined-P1-Dataset" / "test"
JA_IMGS   = ROOT / "ja_scraper" / "output" / "images"
BE_IMGS   = ROOT / "be_scraper" / "output" / "images"
OUT_CSV   = DATA_DIR / "combined_pipeline_manifest.csv"

STAGE2_SUBSETS = {
    "ja_natural": JA_IMGS,
    "be_natural": BE_IMGS,
    "ja_lab":     JA_IMGS,
    "be_lab":     BE_IMGS,
}


def build_stage1_rows() -> list[dict]:
    rows = []
    for class_dir in sorted(STAGE1.iterdir()):
        if not class_dir.is_dir():
            continue
        species = class_dir.name
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            rows.append({
                "image_path": str(img_path),
                "species":    species,
                "value_tier": "N/A",
                "subset":     "kaggle_stage1",
            })
    return rows


def build_stage2_rows() -> list[dict]:
    rows = []
    for subset, img_root in STAGE2_SUBSETS.items():
        csv_path = SPLITS / f"{subset}_test.csv"
        df = pd.read_csv(csv_path, usecols=["diamond_id", "value_tier"], low_memory=False)
        for _, row in df.iterrows():
            img_path = img_root / str(row["value_tier"]) / f"{int(row['diamond_id'])}.jpg"
            if not img_path.exists():
                continue
            rows.append({
                "image_path": str(img_path),
                "species":    "Diamond",
                "value_tier": row["value_tier"],
                "subset":     subset,
            })
    return rows


def main() -> None:
    print("Building Stage 1 rows (Kaggle test split)...")
    stage1_rows = build_stage1_rows()
    print(f"  {len(stage1_rows):,} images across {len({r['species'] for r in stage1_rows})} classes")

    print("Building Stage 2 rows (JA/BE test splits)...")
    stage2_rows = build_stage2_rows()
    print(f"  {len(stage2_rows):,} images across {len(STAGE2_SUBSETS)} subsets")

    df = pd.DataFrame(stage1_rows + stage2_rows)

    print(f"\nCombined manifest: {len(df):,} total images")
    print("\nBreakdown by subset:")
    print(df.groupby("subset").size().to_string())
    print("\nBreakdown by species (top 10):")
    print(df.groupby("species").size().sort_values(ascending=False).head(10).to_string())
    print("\nDiamond images with tier labels:")
    diamond_with_tier = df[(df["species"] == "Diamond") & (df["value_tier"] != "N/A")]
    print(diamond_with_tier.groupby("value_tier").size().to_string())

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
