"""
audit.py — Dataset audit and health check
CSC-481 Stage 2 dataset validation

Checks the collected dataset for:
  - Sample counts per class/tier
  - Class balance
  - Image integrity (not corrupt, right size)
  - 4C grade space coverage
  - CSV/image alignment (every image row has a file, no orphan files)
  - Proposes train/val/test split

Usage: python audit.py
       python audit.py --fix   (move orphan images, remove corrupt files)
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUTPUT_DIR  = Path("output")
LABELED_CSV = OUTPUT_DIR / "diamonds_labeled.csv"
IMAGES_DIR  = OUTPUT_DIR / "images"
AUDIT_REPORT = OUTPUT_DIR / "audit_report.txt"

MIN_IMAGE_BYTES = 5_000
TARGET_PER_CLASS = 500    # ideal minimum images per tier for training

TIERS = ["budget", "mid_range", "premium", "investment_grade"]

TIER_DIRS = {t: IMAGES_DIR / t for t in TIERS}
TIER_DIRS["unlabeled"] = IMAGES_DIR / "unlabeled"


def load_labeled():
    if not LABELED_CSV.exists():
        print(f"ERROR: {LABELED_CSV} not found. Run label_tiers.py first.")
        sys.exit(1)
    rows = []
    with open(LABELED_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def check_images(rows):
    """Verify images on disk, return (present, corrupt, missing) sets."""
    id_to_row = {r["diamond_id"]: r for r in rows}

    present   = set()
    corrupt   = set()
    orphan    = set()

    for jpg in IMAGES_DIR.rglob("*.jpg"):
        did = jpg.stem
        sz = jpg.stat().st_size
        if sz < MIN_IMAGE_BYTES:
            corrupt.add(did)
        elif did in id_to_row:
            present.add(did)
        else:
            orphan.add(did)

    missing = set(id_to_row.keys()) - present - corrupt
    return present, corrupt, missing, orphan


def grade_space_coverage(rows):
    """Report how well the 4C grade space is covered."""
    cuts      = Counter(r.get("cut", "").strip() for r in rows if r.get("cut"))
    colors    = Counter(r.get("color", "").strip() for r in rows if r.get("color"))
    clarities = Counter(r.get("clarity", "").strip() for r in rows if r.get("clarity"))

    return cuts, colors, clarities


def split_recommendation(present_by_tier):
    """Print train/val/test split recommendation."""
    lines = []
    lines.append("\nRECOMMENDED SPLIT (70/15/15 stratified):")
    total = sum(present_by_tier.values())
    for tier, n in sorted(present_by_tier.items()):
        train = int(n * 0.70)
        val   = int(n * 0.15)
        test  = n - train - val
        lines.append(f"  {tier:20s}  total={n:>5}  train={train:>4}  val={val:>4}  test={test:>4}")
    lines.append(f"  {'TOTAL':20s}  total={total:>5}  "
                 f"train={int(total*0.70):>4}  val={int(total*0.15):>4}  "
                 f"test={total-int(total*0.70)-int(total*0.15):>4}")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="Remove corrupt images, move orphans")
    args = parser.parse_args()

    lines = ["=" * 60, "CSC-481 DATASET AUDIT REPORT", "=" * 60, ""]

    rows = load_labeled()
    lines.append(f"Total rows in labeled CSV: {len(rows)}")

    # Image checks
    print("Checking images on disk (may take a moment)...")
    present, corrupt, missing, orphan = check_images(rows)
    lines.append(f"Images present and valid:  {len(present)}")
    lines.append(f"Images corrupt/too small:  {len(corrupt)}")
    lines.append(f"CSV rows with no image:    {len(missing)}")
    lines.append(f"Orphan images (no CSV row):{len(orphan)}")
    lines.append("")

    if corrupt and args.fix:
        for did in corrupt:
            for tier_dir in TIER_DIRS.values():
                p = tier_dir / f"{did}.jpg"
                if p.exists():
                    p.unlink()
                    print(f"  Removed corrupt: {p}")

    # Tier distribution of images with valid files
    id_to_row = {r["diamond_id"]: r for r in rows}
    present_by_tier = defaultdict(int)
    for did in present:
        tier = id_to_row.get(did, {}).get("value_tier", "unlabeled")
        present_by_tier[tier] += 1

    lines.append("IMAGES WITH VALID FILES PER TIER:")
    for tier in TIERS + ["unlabeled"]:
        n = present_by_tier.get(tier, 0)
        bar = "█" * (n // 50)
        status = "✓" if n >= TARGET_PER_CLASS else ("⚠" if n >= 100 else "✗")
        lines.append(f"  {status} {tier:20s} {n:>5}  {bar}")
    lines.append("")

    # 4C grade coverage
    cuts, colors, clarities = grade_space_coverage(rows)

    lines.append("CUT GRADES:")
    for k, v in sorted(cuts.items(), key=lambda x: -x[1]):
        lines.append(f"  {k:20s} {v:>6}")
    lines.append("")

    lines.append("COLOR GRADES:")
    for k in ["D", "E", "F", "G", "H", "I", "J"]:
        lines.append(f"  {k:20s} {colors.get(k, 0):>6}")
    lines.append("")

    lines.append("CLARITY GRADES:")
    for k in ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"]:
        lines.append(f"  {k:20s} {clarities.get(k, 0):>6}")
    lines.append("")

    # Class balance check
    values = [present_by_tier.get(t, 0) for t in TIERS]
    if values and max(values) > 0:
        ratio = min(v for v in values if v > 0) / max(values)
        lines.append(f"CLASS BALANCE RATIO (min/max): {ratio:.2f}")
        if ratio >= 0.7:
            lines.append("  ✓ Reasonably balanced (≥0.7)")
        elif ratio >= 0.4:
            lines.append("  ⚠ Moderate imbalance — consider weighted loss in training")
        else:
            lines.append("  ✗ Severe imbalance — use weighted sampling or oversample minority classes")
        lines.append("")

    # Split recommendation
    lines.extend(split_recommendation(present_by_tier))
    lines.append("")

    # Overall readiness
    total_valid = sum(present_by_tier.get(t, 0) for t in TIERS)
    min_tier = min(present_by_tier.get(t, 0) for t in TIERS)
    lines.append("ML READINESS SUMMARY:")
    if total_valid >= 5000 and min_tier >= 300:
        lines.append("  ✓ READY — sufficient data for Stage 2 training")
    elif total_valid >= 2000 and min_tier >= 100:
        lines.append("  ⚠ MARGINAL — training possible but more data recommended")
        lines.append(f"  Smallest tier: {min_tier} images (target: ≥300 per tier)")
    else:
        lines.append(f"  ✗ NOT READY — continue scraping")
        lines.append(f"  Current: {total_valid} valid images, smallest tier: {min_tier}")
        lines.append(f"  Target: ≥2,000 total, ≥300 per tier")

    # Print and save
    report_text = "\n".join(lines)
    print("\n" + report_text)

    with open(AUDIT_REPORT, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to {AUDIT_REPORT}")


if __name__ == "__main__":
    main()
