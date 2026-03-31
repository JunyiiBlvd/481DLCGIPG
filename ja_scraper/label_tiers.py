"""
label_tiers.py — Compute value tier labels from price distribution
CSC-481 Stage 2 dataset preparation

1. Loads diamonds_raw.csv
2. Computes price percentile boundaries (p25, p75, p90)
3. Assigns each diamond a value_tier label
4. Writes diamonds_labeled.csv
5. Renames already-downloaded images into tier folders
6. Prints tier distribution and data quality stats

Run this AFTER scrape.py completes (or after enough rows exist to
compute a stable price distribution — recommended: ≥1,000 rows).

Usage: python label_tiers.py
"""

import csv
import json
import os
import shutil
import sys
from pathlib import Path
from collections import Counter

# ─────────────────────────────────────────
# Paths
# ─────────────────────────────────────────
OUTPUT_DIR  = Path("output")
NAT_RAW_CSV = OUTPUT_DIR / "diamonds_natural_raw.csv"
LAB_RAW_CSV = OUTPUT_DIR / "diamonds_lab_raw.csv"
LABELED_CSV = OUTPUT_DIR / "diamonds_labeled.csv"
IMAGES_DIR  = OUTPUT_DIR / "images"
STATS_FILE  = OUTPUT_DIR / "tier_stats.json"

TIER_DIRS = {
    "budget":           IMAGES_DIR / "budget",
    "mid_range":        IMAGES_DIR / "mid_range",
    "premium":          IMAGES_DIR / "premium",
    "investment_grade": IMAGES_DIR / "investment_grade",
    "unlabeled":        IMAGES_DIR / "unlabeled",
}
for d in TIER_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────
# Load raw CSVs
# ─────────────────────────────────────────
def load_csv(path):
    if not path.exists():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    print(f"  Loaded {len(rows):,} rows from {path.name}")
    return rows


# ─────────────────────────────────────────
# Parse price safely
# ─────────────────────────────────────────
def parse_price(val):
    if not val:
        return None
    try:
        return float(str(val).replace(",", "").replace("$", "").strip())
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────
# Compute price percentiles
# ─────────────────────────────────────────
def percentile(values, p):
    if not values:
        return 0
    sorted_v = sorted(values)
    idx = (len(sorted_v) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_v) - 1)
    return sorted_v[lo] + (sorted_v[hi] - sorted_v[lo]) * (idx - lo)


def compute_boundaries(rows, label=""):
    prices = [parse_price(r.get("price_usd")) for r in rows]
    prices = [p for p in prices if p is not None and p > 0]

    if len(prices) < 100:
        print(f"WARNING: Only {len(prices)} rows have valid prices in {label}. Boundaries may be unstable.")

    p25 = percentile(prices, 25)
    p75 = percentile(prices, 75)
    p90 = percentile(prices, 90)

    print(f"\nPrice distribution — {label} ({len(prices):,} diamonds with price):")
    print(f"  Min:         ${min(prices):>10,.0f}")
    print(f"  P25 (Budget boundary):     ${p25:>10,.0f}")
    print(f"  Median:      ${percentile(prices, 50):>10,.0f}")
    print(f"  P75 (Premium boundary):    ${p75:>10,.0f}")
    print(f"  P90 (Investment boundary): ${p90:>10,.0f}")
    print(f"  Max:         ${max(prices):>10,.0f}")

    return p25, p75, p90


# ─────────────────────────────────────────
# Assign tier
# ─────────────────────────────────────────
def assign_tier(price, p25, p75, p90):
    if price is None or price <= 0:
        return "unlabeled"
    if price <= p25:
        return "budget"
    if price <= p75:
        return "mid_range"
    if price <= p90:
        return "premium"
    return "investment_grade"


# ─────────────────────────────────────────
# Deduplicate (keep first occurrence per diamond_id)
# ─────────────────────────────────────────
def deduplicate(rows):
    seen = set()
    deduped = []
    for row in rows:
        did = row.get("diamond_id", "")
        if did and did not in seen:
            seen.add(did)
            deduped.append(row)
    n_removed = len(rows) - len(deduped)
    if n_removed:
        print(f"Removed {n_removed} duplicate diamond IDs")
    return deduped


# ─────────────────────────────────────────
# Write labeled CSV
# ─────────────────────────────────────────
def write_labeled(rows):
    fields = list(rows[0].keys())
    for col in ("value_tier", "site"):
        if col not in fields:
            fields.append(col)

    with open(LABELED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows):,} labeled rows to {LABELED_CSV.name}")


# ─────────────────────────────────────────
# Move images into tier folders
# ─────────────────────────────────────────
def organize_images(rows):
    moved = 0
    missing = 0

    # Build id→tier lookup
    tier_map = {r["diamond_id"]: r.get("value_tier", "unlabeled") for r in rows}

    # Scan all jpg files under images/
    for jpg in IMAGES_DIR.rglob("*.jpg"):
        did = jpg.stem
        if did not in tier_map:
            continue
        tier = tier_map[did]
        dest_dir = TIER_DIRS.get(tier, TIER_DIRS["unlabeled"])
        dest = dest_dir / jpg.name
        if dest == jpg:
            continue
        try:
            shutil.move(str(jpg), str(dest))
            moved += 1
        except Exception as e:
            print(f"  Move failed {jpg} → {dest}: {e}")

    # Count missing images
    for row in rows:
        tier = row.get("value_tier", "unlabeled")
        dest = TIER_DIRS.get(tier, TIER_DIRS["unlabeled"]) / f"{row['diamond_id']}.jpg"
        if not dest.exists():
            missing += 1

    print(f"Organized images: {moved} moved")
    if missing:
        print(f"  Images not yet downloaded: {missing} (run download_images.py)")


# ─────────────────────────────────────────
# Print distribution and quality report
# ─────────────────────────────────────────
def print_report(rows, nat_bounds, lab_bounds):
    tier_counts = Counter(r.get("value_tier") for r in rows)
    total = len(rows)
    n_natural = sum(1 for r in rows if str(r.get("is_lab_diamond","")).lower() not in ("true","1","yes"))
    n_lab     = total - n_natural

    print("\n" + "=" * 55)
    print("JAMES ALLEN — VALUE TIER DISTRIBUTION")
    print("=" * 55)
    for tier in ["budget", "mid_range", "premium", "investment_grade", "unlabeled"]:
        n = tier_counts.get(tier, 0)
        pct = 100 * n / total if total else 0
        bar = "█" * int(pct / 2)
        print(f"  {tier:20s} {n:>7,} ({pct:5.1f}%)  {bar}")
    print(f"\n  Natural: {n_natural:,}  |  Lab-grown: {n_lab:,}  |  Total: {total:,}")

    print("\nDATA QUALITY")
    has_image = sum(1 for r in rows if r.get("image_url"))
    has_cut   = sum(1 for r in rows if r.get("cut"))
    has_color = sum(1 for r in rows if r.get("color"))
    has_cert  = sum(1 for r in rows if r.get("cert_lab"))
    print(f"  image_url populated:  {has_image:,}/{total:,} ({100*has_image/total:.1f}%)")
    print(f"  cut populated:        {has_cut:,}/{total:,}  ({100*has_cut/total:.1f}%)")
    print(f"  color populated:      {has_color:,}/{total:,} ({100*has_color/total:.1f}%)")
    print(f"  cert_lab populated:   {has_cert:,}/{total:,}  ({100*has_cert/total:.1f}%)")

    print("\nTIER BOUNDARIES (computed independently per subset):")
    if nat_bounds:
        p25, p75, p90 = nat_bounds
        print(f"  NATURAL  — Budget ≤${p25:,.0f}  Mid ≤${p75:,.0f}  Premium ≤${p90:,.0f}  Investment >${p90:,.0f}")
    if lab_bounds:
        p25, p75, p90 = lab_bounds
        print(f"  LAB      — Budget ≤${p25:,.0f}  Mid ≤${p75:,.0f}  Premium ≤${p90:,.0f}  Investment >${p90:,.0f}")

    # Check image counts per tier (what's actually downloaded)
    print("\nDOWNLOADED IMAGES PER TIER:")
    for tier, d in TIER_DIRS.items():
        n = sum(1 for p in d.glob("*.jpg") if p.stat().st_size >= 5000)
        if n > 0:
            print(f"  {tier:20s} {n:>6} images")

    print("\nML READINESS:")
    has_both = sum(
        1 for r in rows
        if r.get("value_tier") not in (None, "unlabeled")
        and r.get("image_url")
    )
    print(f"  Rows with tier + image_url: {has_both}")
    if has_both >= 2000:
        print(f"  ✓ Sufficient for Stage 2 training (≥2,000)")
    elif has_both >= 500:
        print(f"  ⚠ Marginal for training — continue scraping")
    else:
        print(f"  ✗ Too few — continue scraping before training")

    # Save stats as JSON for the proposal
    stats = {
        "total_diamonds": total,
        "tier_counts": dict(tier_counts),
        "boundaries": {"p25": round(p25, 2), "p75": round(p75, 2), "p90": round(p90, 2)},
        "data_quality": {
            "has_image_url": has_image,
            "has_cut": has_cut,
            "has_color": has_color,
            "has_cert": has_cert,
        },
        "ml_ready_rows": has_both,
    }
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {STATS_FILE}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    print("Loading raw CSVs ...")
    nat_rows = load_csv(NAT_RAW_CSV)
    lab_rows = load_csv(LAB_RAW_CSV)

    if not nat_rows and not lab_rows:
        print("ERROR: No data found. Run scrape_v3.py first.")
        sys.exit(1)

    nat_rows = deduplicate(nat_rows)
    lab_rows = deduplicate(lab_rows)

    # Compute boundaries INDEPENDENTLY per natural/lab subset
    nat_bounds = lab_bounds = None

    if nat_rows:
        nat_bounds = compute_boundaries(nat_rows, "Natural")
        for row in nat_rows:
            row["value_tier"] = assign_tier(parse_price(row.get("price_usd")), *nat_bounds)
            row["site"] = "james_allen"

    if lab_rows:
        lab_bounds = compute_boundaries(lab_rows, "Lab-grown")
        for row in lab_rows:
            row["value_tier"] = assign_tier(parse_price(row.get("price_usd")), *lab_bounds)
            row["site"] = "james_allen"

    all_rows = nat_rows + lab_rows
    write_labeled(all_rows)
    organize_images(all_rows)
    print_report(all_rows, nat_bounds, lab_bounds)

    print("\nDone. Next step: python download_images.py")


if __name__ == "__main__":
    main()
