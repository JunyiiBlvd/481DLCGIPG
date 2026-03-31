"""
be_labeler.py — Compute value-tier labels for Brilliant Earth diamonds
DLCGIPG project — cross-domain generalization experiment

1. Loads output/diamonds_natural_raw.csv  AND  output/diamonds_lab_raw.csv
2. Filters to D–Z color scale only (drops fancy colored diamonds)
3. Computes price percentile boundaries INDEPENDENTLY per subset
   (natural and lab-grown have very different price distributions)
4. Assigns each diamond a value_tier label using BE's own distribution:
     Budget:           price ≤ p25
     Mid-Range:        p25 < price ≤ p75
     Premium:          p75 < price ≤ p90
     Investment-Grade: price > p90
5. Writes output/be_labeled.csv  (combined natural + lab)
6. Moves already-downloaded images into output/images/{tier}/ folders
7. Prints distribution + data-quality report and saves tier_stats.json

Run AFTER scrape.py completes (or after ≥1,000 rows for stable percentiles).

Usage:
    python be_labeler.py
    python be_labeler.py --natural-only     # skip lab CSV (if not scraped yet)
    python be_labeler.py --lab-only         # skip natural CSV
    python be_labeler.py --dry-run          # print report, do not write files
"""

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR       = Path("output")
NAT_RAW_CSV      = OUTPUT_DIR / "diamonds_natural_raw.csv"
LAB_RAW_CSV      = OUTPUT_DIR / "diamonds_lab_raw.csv"
LABELED_CSV      = OUTPUT_DIR / "be_labeled.csv"
IMAGES_DIR       = OUTPUT_DIR / "images"
STATS_FILE       = OUTPUT_DIR / "be_tier_stats.json"

TIER_DIRS = {
    "budget":            IMAGES_DIR / "budget",
    "mid_range":         IMAGES_DIR / "mid_range",
    "premium":           IMAGES_DIR / "premium",
    "investment_grade":  IMAGES_DIR / "investment_grade",
    "unlabeled":         IMAGES_DIR / "unlabeled",
}
for d in TIER_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# D–Z color grades — everything outside this is fancy colored
STANDARD_COLORS = {"D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
                   "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
                   "X", "Y", "Z"}


# ─────────────────────────────────────────────────────────────────────
# Load one raw CSV
# ─────────────────────────────────────────────────────────────────────
def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    print(f"  Loaded {len(rows):,} rows from {path.name}")
    return rows


# ─────────────────────────────────────────────────────────────────────
# Filter: D–Z color only
# ─────────────────────────────────────────────────────────────────────
def filter_standard_color(rows: list[dict]) -> tuple[list[dict], int]:
    kept   = []
    skipped = 0
    for row in rows:
        color = str(row.get("color") or "").strip().upper()
        if not color or color in STANDARD_COLORS:
            kept.append(row)
        else:
            skipped += 1
    return kept, skipped


# ─────────────────────────────────────────────────────────────────────
# Deduplicate by diamond_id (keep first occurrence)
# ─────────────────────────────────────────────────────────────────────
def deduplicate(rows: list[dict]) -> list[dict]:
    seen   = set()
    deduped = []
    for row in rows:
        did = row.get("diamond_id", "")
        if did and did not in seen:
            seen.add(did)
            deduped.append(row)
    removed = len(rows) - len(deduped)
    if removed:
        print(f"  Removed {removed:,} duplicate diamond IDs")
    return deduped


# ─────────────────────────────────────────────────────────────────────
# Price parsing
# ─────────────────────────────────────────────────────────────────────
def parse_price(val) -> float | None:
    if not val:
        return None
    try:
        return float(str(val).replace(",", "").replace("$", "").strip())
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────────────
# Percentile (linear interpolation)
# ─────────────────────────────────────────────────────────────────────
def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sv  = sorted(values)
    idx = (len(sv) - 1) * p / 100
    lo  = int(idx)
    hi  = min(lo + 1, len(sv) - 1)
    return sv[lo] + (sv[hi] - sv[lo]) * (idx - lo)


# ─────────────────────────────────────────────────────────────────────
# Compute price boundaries for one subset
# ─────────────────────────────────────────────────────────────────────
def compute_boundaries(rows: list[dict], label: str) -> tuple[float, float, float]:
    prices = [parse_price(r.get("price_usd")) for r in rows]
    prices = [p for p in prices if p is not None and p > 0]

    if len(prices) < 100:
        print(
            f"  WARNING: Only {len(prices)} rows have valid prices in {label}. "
            f"Tier boundaries may be unstable (recommend ≥1,000)."
        )

    p25 = percentile(prices, 25)
    p75 = percentile(prices, 75)
    p90 = percentile(prices, 90)

    print(f"\nPrice distribution — {label} ({len(prices):,} diamonds with price):")
    for pct_label, pval in [
        ("Min",        min(prices)),
        ("P10",        percentile(prices, 10)),
        ("P25 (Budget boundary)", p25),
        ("Median",     percentile(prices, 50)),
        ("P75 (Premium boundary)", p75),
        ("P90 (Investment boundary)", p90),
        ("Max",        max(prices)),
    ]:
        print(f"  {pct_label:38s} ${pval:>12,.0f}")

    return p25, p75, p90


# ─────────────────────────────────────────────────────────────────────
# Assign tier
# ─────────────────────────────────────────────────────────────────────
def assign_tier(price: float | None, p25: float, p75: float, p90: float) -> str:
    if price is None or price <= 0:
        return "unlabeled"
    if price <= p25:
        return "budget"
    if price <= p75:
        return "mid_range"
    if price <= p90:
        return "premium"
    return "investment_grade"


# ─────────────────────────────────────────────────────────────────────
# Write combined labeled CSV
# ─────────────────────────────────────────────────────────────────────
def write_labeled(rows: list[dict], dry_run: bool):
    if dry_run:
        print(f"\n[dry-run] Would write {len(rows):,} rows to {LABELED_CSV}")
        return

    # Build field list: all keys from first row + value_tier (if not already present)
    fields = list(rows[0].keys()) if rows else []
    for extra in ("value_tier", "site"):
        if extra not in fields:
            fields.append(extra)

    with open(LABELED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows):,} labeled rows → {LABELED_CSV}")


# ─────────────────────────────────────────────────────────────────────
# Organize images into tier subdirectories
# ─────────────────────────────────────────────────────────────────────
def organize_images(rows: list[dict], dry_run: bool):
    tier_map = {r["diamond_id"]: r.get("value_tier", "unlabeled") for r in rows}

    moved   = 0
    missing = 0

    for jpg in IMAGES_DIR.rglob("*.jpg"):
        # Skip files already inside a tier subdirectory
        did  = jpg.stem
        tier = tier_map.get(did)
        if tier is None:
            continue  # orphan — not in our CSV

        dest_dir = TIER_DIRS.get(tier, TIER_DIRS["unlabeled"])
        dest     = dest_dir / jpg.name

        if dest == jpg:
            continue  # already in the right place

        if dry_run:
            moved += 1
            continue

        try:
            shutil.move(str(jpg), str(dest))
            moved += 1
        except Exception as e:
            print(f"  Move failed {jpg} → {dest}: {e}")

    for row in rows:
        tier = row.get("value_tier", "unlabeled")
        dest = TIER_DIRS.get(tier, TIER_DIRS["unlabeled"]) / f"{row['diamond_id']}.jpg"
        if not dest.exists():
            missing += 1

    action = "Would move" if dry_run else "Moved"
    print(f"Images: {action} {moved:,} files into tier folders")
    if missing:
        print(f"  Not yet downloaded: {missing:,}  (run download_images.py)")


# ─────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────
def print_report(rows: list[dict],
                 nat_bounds: tuple, lab_bounds: tuple | None,
                 skipped_natural: int, skipped_lab: int):

    tier_counts  = Counter(r.get("value_tier") for r in rows)
    total        = len(rows)
    n_natural    = sum(1 for r in rows if not str(r.get("is_lab_diamond","")).lower() in ("true","1","yes"))
    n_lab        = total - n_natural

    print("\n" + "=" * 60)
    print("BRILLIANT EARTH — VALUE TIER DISTRIBUTION")
    print("=" * 60)
    for tier in ["budget", "mid_range", "premium", "investment_grade", "unlabeled"]:
        n   = tier_counts.get(tier, 0)
        pct = 100 * n / total if total else 0
        bar = "█" * int(pct / 2)
        print(f"  {tier:20s} {n:>7,} ({pct:5.1f}%)  {bar}")

    print(f"\n  Natural:  {n_natural:,}  |  Lab-grown: {n_lab:,}  |  Total: {total:,}")

    skipped_total = skipped_natural + skipped_lab
    if skipped_total:
        print(f"\n  Fancy-colored diamonds skipped: {skipped_total:,}")
        print(f"    Natural: {skipped_natural:,}  |  Lab: {skipped_lab:,}")

    print("\nTIER BOUNDARIES (computed independently per subset):")
    p25n, p75n, p90n = nat_bounds
    print(f"  NATURAL  — Budget ≤${p25n:,.0f}  Mid ≤${p75n:,.0f}  Premium ≤${p90n:,.0f}  Investment >${p90n:,.0f}")
    if lab_bounds:
        p25l, p75l, p90l = lab_bounds
        print(f"  LAB      — Budget ≤${p25l:,.0f}  Mid ≤${p75l:,.0f}  Premium ≤${p90l:,.0f}  Investment >${p90l:,.0f}")

    print("\nDATA QUALITY:")
    has_image = sum(1 for r in rows if r.get("image_url"))
    has_cut   = sum(1 for r in rows if r.get("cut"))
    has_color = sum(1 for r in rows if r.get("color"))
    has_cert  = sum(1 for r in rows if r.get("cert_lab"))
    for label, n in [
        ("image_url populated",  has_image),
        ("cut populated",        has_cut),
        ("color populated",      has_color),
        ("cert_lab populated",   has_cert),
    ]:
        pct = 100 * n / total if total else 0
        print(f"  {label:28s} {n:>7,}/{total:,}  ({pct:.1f}%)")

    print("\nDOWNLOADED IMAGES PER TIER:")
    for tier, d in TIER_DIRS.items():
        n = sum(1 for p in d.glob("*.jpg") if p.stat().st_size >= 5_000)
        if n > 0:
            print(f"  {tier:20s} {n:>7,} images")

    has_both = sum(
        1 for r in rows
        if r.get("value_tier") not in (None, "unlabeled") and r.get("image_url")
    )
    print(f"\nML READINESS:")
    print(f"  Rows with tier + image_url: {has_both:,}")
    if has_both >= 2000:
        print(f"  ✓ Sufficient for Stage 2 training (≥2,000)")
    elif has_both >= 500:
        print(f"  ⚠ Marginal — continue scraping")
    else:
        print(f"  ✗ Too few — continue scraping before training")

    # Save stats JSON
    stats = {
        "site":           "brilliant_earth",
        "total_diamonds": total,
        "n_natural":      n_natural,
        "n_lab":          n_lab,
        "tier_counts":    dict(tier_counts),
        "boundaries": {
            "natural": {
                "p25": round(p25n, 2),
                "p75": round(p75n, 2),
                "p90": round(p90n, 2),
            },
            "lab": {
                "p25": round(lab_bounds[0], 2),
                "p75": round(lab_bounds[1], 2),
                "p90": round(lab_bounds[2], 2),
            } if lab_bounds else None,
        },
        "data_quality": {
            "has_image_url": has_image,
            "has_cut":       has_cut,
            "has_color":     has_color,
            "has_cert":      has_cert,
        },
        "ml_ready_rows":  has_both,
        "fancy_skipped":  skipped_total,
    }
    STATS_FILE.write_text(json.dumps(stats, indent=2))
    print(f"\nStats saved → {STATS_FILE}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Label Brilliant Earth diamonds into 4 value tiers"
    )
    parser.add_argument("--natural-only", action="store_true",
                        help="Only process natural diamonds CSV")
    parser.add_argument("--lab-only",     action="store_true",
                        help="Only process lab-grown diamonds CSV")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Print report without writing files or moving images")
    args = parser.parse_args()

    # ── Load ──
    print("Loading raw CSVs ...")
    nat_rows, lab_rows = [], []
    if not args.lab_only:
        if NAT_RAW_CSV.exists():
            nat_rows = load_csv(NAT_RAW_CSV)
        else:
            print(f"  {NAT_RAW_CSV} not found — skipping natural diamonds")
    if not args.natural_only:
        if LAB_RAW_CSV.exists():
            lab_rows = load_csv(LAB_RAW_CSV)
        else:
            print(f"  {LAB_RAW_CSV} not found — skipping lab-grown diamonds")

    if not nat_rows and not lab_rows:
        print("ERROR: No data found. Run scrape.py first.")
        sys.exit(1)

    # ── Dedup ──
    nat_rows = deduplicate(nat_rows)
    lab_rows = deduplicate(lab_rows)

    # ── Filter fancy colors ──
    print("\nFiltering to D–Z color scale ...")
    nat_rows, skip_nat = filter_standard_color(nat_rows)
    lab_rows, skip_lab = filter_standard_color(lab_rows)
    if skip_nat:
        print(f"  Natural: skipped {skip_nat:,} fancy-colored diamonds")
    if skip_lab:
        print(f"  Lab:     skipped {skip_lab:,} fancy-colored diamonds")

    # ── Compute boundaries INDEPENDENTLY per subset ──
    nat_bounds = None
    lab_bounds = None

    if nat_rows:
        nat_bounds = compute_boundaries(nat_rows, "Natural")
        for row in nat_rows:
            price = parse_price(row.get("price_usd"))
            row["value_tier"] = assign_tier(price, *nat_bounds)
            row["site"]       = "brilliant_earth"

    if lab_rows:
        lab_bounds = compute_boundaries(lab_rows, "Lab-grown")
        for row in lab_rows:
            price = parse_price(row.get("price_usd"))
            row["value_tier"] = assign_tier(price, *lab_bounds)
            row["site"]       = "brilliant_earth"

    all_rows = nat_rows + lab_rows

    if not all_rows:
        print("ERROR: No rows after filtering.")
        sys.exit(1)

    # ── Write ──
    write_labeled(all_rows, args.dry_run)
    organize_images(all_rows, args.dry_run)
    print_report(
        all_rows,
        nat_bounds or (0, 0, 0),
        lab_bounds,
        skip_nat,
        skip_lab,
    )

    if not args.dry_run:
        print(f"\nDone. Next step: python be_audit.py")


if __name__ == "__main__":
    main()
