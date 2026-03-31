"""
be_audit.py — Dataset alignment check for Brilliant Earth diamonds
DLCGIPG project — cross-domain generalization experiment

Checks:
  - Total records in be_labeled.csv
  - Images found on disk vs CSV rows
  - Images missing (in CSV but no file)
  - Orphan images (file on disk but no CSV row)
  - Corrupt/too-small images
  - 4C grade-space coverage
  - Class balance across tiers
  - ML readiness summary

Output format matches ja_scraper/audit.py for direct cross-site comparison.
Report saved to output/be_audit_report.txt.

Usage:
    python be_audit.py
    python be_audit.py --fix    # remove corrupt images, relocate orphans
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUTPUT_DIR   = Path("output")
LABELED_CSV  = OUTPUT_DIR / "be_labeled.csv"
IMAGES_DIR   = OUTPUT_DIR / "images"
AUDIT_REPORT = OUTPUT_DIR / "be_audit_report.txt"

MIN_IMAGE_BYTES  = 5_000
TARGET_PER_CLASS = 500

TIERS = ["budget", "mid_range", "premium", "investment_grade"]
TIER_DIRS = {t: IMAGES_DIR / t for t in TIERS}
TIER_DIRS["unlabeled"] = IMAGES_DIR / "unlabeled"


# ─────────────────────────────────────────────────────────────────────
def load_labeled() -> list[dict]:
    if not LABELED_CSV.exists():
        print(f"ERROR: {LABELED_CSV} not found. Run be_labeler.py first.")
        sys.exit(1)
    rows = []
    with open(LABELED_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────
def check_images(rows: list[dict]):
    """Return (present, corrupt, missing, orphan) sets of diamond_ids."""
    id_to_row = {r["diamond_id"]: r for r in rows}

    present = set()
    corrupt = set()
    orphan  = set()

    for jpg in IMAGES_DIR.rglob("*.jpg"):
        did = jpg.stem
        sz  = jpg.stat().st_size
        if sz < MIN_IMAGE_BYTES:
            corrupt.add(did)
        elif did in id_to_row:
            present.add(did)
        else:
            orphan.add(did)

    missing = set(id_to_row.keys()) - present - corrupt
    return present, corrupt, missing, orphan


# ─────────────────────────────────────────────────────────────────────
def grade_space_coverage(rows: list[dict]):
    cuts      = Counter(r.get("cut",     "").strip() for r in rows if r.get("cut"))
    colors    = Counter(r.get("color",   "").strip() for r in rows if r.get("color"))
    clarities = Counter(r.get("clarity", "").strip() for r in rows if r.get("clarity"))
    return cuts, colors, clarities


# ─────────────────────────────────────────────────────────────────────
def split_recommendation(present_by_tier: dict) -> list[str]:
    lines = ["\nRECOMMENDED SPLIT (70/15/15 stratified):"]
    total = sum(present_by_tier.values())
    for tier, n in sorted(present_by_tier.items()):
        train = int(n * 0.70)
        val   = int(n * 0.15)
        test  = n - train - val
        lines.append(
            f"  {tier:20s}  total={n:>5,}  "
            f"train={train:>4,}  val={val:>4,}  test={test:>4,}"
        )
    lines.append(
        f"  {'TOTAL':20s}  total={total:>5,}  "
        f"train={int(total*0.70):>4,}  val={int(total*0.15):>4,}  "
        f"test={total-int(total*0.70)-int(total*0.15):>4,}"
    )
    return lines


# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Brilliant Earth dataset audit — mirrors ja_scraper/audit.py format"
    )
    parser.add_argument("--fix", action="store_true",
                        help="Remove corrupt images; move orphans to unlabeled/")
    args = parser.parse_args()

    lines = ["=" * 60, "BRILLIANT EARTH DATASET AUDIT REPORT", "=" * 60, ""]

    rows = load_labeled()
    lines.append(f"Total rows in be_labeled.csv:  {len(rows):,}")

    # ── Site column check ──
    sites = Counter(r.get("site", "") for r in rows)
    if sites:
        for site_val, n in sorted(sites.items()):
            lines.append(f"  site={site_val!r:25s} {n:,}")
    lines.append("")

    # ── Image checks ──
    print("Checking images on disk (may take a moment) ...")
    present, corrupt, missing, orphan = check_images(rows)

    total_rows  = len(rows)
    found_count = len(present)
    miss_count  = len(missing)
    fail_rate   = 100 * miss_count / total_rows if total_rows else 0.0

    lines.append(f"Images present and valid:      {found_count:,}")
    lines.append(f"Images corrupt / too small:    {len(corrupt):,}")
    lines.append(f"CSV rows with no image:        {miss_count:,}")
    lines.append(f"Orphan images (no CSV row):    {len(orphan):,}")
    lines.append(f"Image coverage:                {100-fail_rate:.1f}%  "
                 f"(failure rate {fail_rate:.2f}%)")
    lines.append("")

    if corrupt and args.fix:
        for did in corrupt:
            for td in TIER_DIRS.values():
                p = td / f"{did}.jpg"
                if p.exists():
                    p.unlink()
                    print(f"  Removed corrupt: {p}")

    if orphan and args.fix:
        unlabeled_dir = TIER_DIRS["unlabeled"]
        unlabeled_dir.mkdir(parents=True, exist_ok=True)
        for did in orphan:
            for td in TIER_DIRS.values():
                p = td / f"{did}.jpg"
                if p.exists():
                    dest = unlabeled_dir / p.name
                    p.rename(dest)
                    print(f"  Moved orphan: {p} → {dest}")

    # ── Per-tier image counts ──
    id_to_row        = {r["diamond_id"]: r for r in rows}
    present_by_tier  = defaultdict(int)
    for did in present:
        tier = id_to_row.get(did, {}).get("value_tier", "unlabeled")
        present_by_tier[tier] += 1

    lines.append("IMAGES WITH VALID FILES PER TIER:")
    for tier in TIERS + ["unlabeled"]:
        n      = present_by_tier.get(tier, 0)
        bar    = "█" * (n // 50)
        status = "✓" if n >= TARGET_PER_CLASS else ("⚠" if n >= 100 else "✗")
        lines.append(f"  {status} {tier:20s} {n:>6,}  {bar}")
    lines.append("")

    # ── 4C grade coverage ──
    cuts, colors, clarities = grade_space_coverage(rows)

    lines.append("CUT GRADES:")
    for k, v in sorted(cuts.items(), key=lambda x: -x[1]):
        lines.append(f"  {k:20s} {v:>7,}")
    lines.append("")

    lines.append("COLOR GRADES:")
    for k in ["D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
              "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]:
        n = colors.get(k, 0)
        if n:
            lines.append(f"  {k:20s} {n:>7,}")
    lines.append("")

    lines.append("CLARITY GRADES:")
    for k in ["FL", "IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1", "I2", "I3"]:
        n = clarities.get(k, 0)
        if n:
            lines.append(f"  {k:20s} {n:>7,}")
    lines.append("")

    # ── Class balance ──
    values = [present_by_tier.get(t, 0) for t in TIERS]
    nonzero = [v for v in values if v > 0]
    if nonzero and max(nonzero) > 0:
        ratio = min(nonzero) / max(nonzero)
        lines.append(f"CLASS BALANCE RATIO (min/max labeled tiers): {ratio:.2f}")
        if ratio >= 0.7:
            lines.append("  ✓ Reasonably balanced (≥0.7)")
        elif ratio >= 0.4:
            lines.append("  ⚠ Moderate imbalance — consider weighted loss in training")
        else:
            lines.append("  ✗ Severe imbalance — use weighted sampling / oversample minority")
        lines.append("")

    # ── Split recommendation ──
    lines.extend(split_recommendation(present_by_tier))
    lines.append("")

    # ── ML readiness ──
    total_valid = sum(present_by_tier.get(t, 0) for t in TIERS)
    min_tier    = min((present_by_tier.get(t, 0) for t in TIERS), default=0)
    lines.append("ML READINESS SUMMARY:")
    if total_valid >= 5_000 and min_tier >= 300:
        lines.append("  ✓ READY — sufficient data for Stage 2 training")
    elif total_valid >= 2_000 and min_tier >= 100:
        lines.append("  ⚠ MARGINAL — training possible but more data recommended")
        lines.append(f"  Smallest tier: {min_tier:,} images  (target: ≥300 per tier)")
    else:
        lines.append(f"  ✗ NOT READY — continue scraping")
        lines.append(f"  Current: {total_valid:,} valid images, smallest tier: {min_tier:,}")
        lines.append(f"  Target: ≥2,000 total, ≥300 per tier")

    report = "\n".join(lines)
    print("\n" + report)

    AUDIT_REPORT.write_text(report)
    print(f"\nReport saved → {AUDIT_REPORT}")


if __name__ == "__main__":
    main()
