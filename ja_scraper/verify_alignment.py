"""
verify_alignment.py — Dataset alignment verification
CSC-481 Stage 2 dataset validation

Checks that CSV rows and downloaded images are aligned.
"""

import csv
import json
import os
import random
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path("output")
NAT_CSV    = OUTPUT_DIR / "diamonds_natural_raw.csv"
LAB_CSV    = OUTPUT_DIR / "diamonds_lab_raw.csv"
IMAGES_DIR = OUTPUT_DIR / "images"
REPORT     = OUTPUT_DIR / "alignment_report.json"

TIER_DIRS = ["budget", "mid_range", "premium", "investment_grade", "unlabeled"]


def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def scan_images():
    """Return dict: image_stem -> Path"""
    img_map = {}
    for jpg in IMAGES_DIR.rglob("*.jpg"):
        img_map[jpg.stem] = jpg
    return img_map


def tier_from_path(path):
    """Extract tier from parent directory name."""
    parent = path.parent.name
    if parent in TIER_DIRS:
        return parent
    return "unlabeled"


def main():
    print("=" * 60)
    print("ALIGNMENT VERIFICATION")
    print("=" * 60)

    # ── Load CSVs ──────────────────────────────────────────────
    nat_rows = load_csv(NAT_CSV)
    lab_rows = load_csv(LAB_CSV)
    all_rows = nat_rows + lab_rows
    total_csv = len(all_rows)

    nat_ids = {r["diamond_id"]: r for r in nat_rows}
    lab_ids = {r["diamond_id"]: r for r in lab_rows}
    all_ids = {r["diamond_id"]: r for r in all_rows}

    print(f"\nCSV rows (natural):  {len(nat_rows):>7}")
    print(f"CSV rows (lab):      {len(lab_rows):>7}")
    print(f"CSV rows (total):    {total_csv:>7}")

    # ── Scan images ─────────────────────────────────────────────
    print("\nScanning images on disk...")
    img_map = scan_images()
    total_images = len(img_map)
    print(f"Images on disk:      {total_images:>7}")

    # ── Compute alignment ────────────────────────────────────────
    csv_ids_set = set(all_ids.keys())
    img_ids_set = set(img_map.keys())

    matched_ids       = csv_ids_set & img_ids_set        # a: CSV rows with matching image
    orphan_csv_ids    = csv_ids_set - img_ids_set        # b: CSV rows with no image
    orphan_image_ids  = img_ids_set - csv_ids_set        # c: images with no CSV row

    # d: rows with image_url populated but file missing
    url_but_missing = [
        r for r in all_rows
        if r.get("image_url") and r["diamond_id"] not in img_ids_set
    ]

    # e: rows where image_url is empty
    no_url = [r for r in all_rows if not r.get("image_url")]

    match_rate = 100.0 * len(matched_ids) / total_csv if total_csv else 0
    orphan_img_pct = 100.0 * len(orphan_image_ids) / total_images if total_images else 0

    print(f"\n{'─'*55}")
    print(f"  a) CSV rows with matching image:        {len(matched_ids):>7}")
    print(f"  b) Orphan CSV rows (no image on disk):  {len(orphan_csv_ids):>7}")
    print(f"  c) Orphan images (no CSV row):          {len(orphan_image_ids):>7}")
    print(f"  d) image_url set but file missing:      {len(url_but_missing):>7}")
    print(f"  e) Rows where image_url is empty:       {len(no_url):>7}")
    print(f"  Match rate:                             {match_rate:>6.2f}%")
    print(f"  Orphan image rate:                      {orphan_img_pct:>6.2f}%")

    # ── Per-tier image counts ────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  f) Per-tier image counts:")
    tier_counts = defaultdict(int)
    for stem, path in img_map.items():
        tier = tier_from_path(path)
        tier_counts[tier] += 1
    for tier in TIER_DIRS:
        print(f"     {tier:20s}  {tier_counts.get(tier, 0):>7}")

    # ── Sample 5 matched pairs ───────────────────────────────────
    print(f"\n{'─'*55}")
    print("  g) Sample matched pairs (5 random):")
    sample_matched = random.sample(list(matched_ids), min(5, len(matched_ids)))
    for did in sample_matched:
        row  = all_ids[did]
        path = img_map[did]
        sz   = path.stat().st_size
        print(f"     diamond_id={did}  price=${row.get('price_usd','?')}  "
              f"cut={row.get('cut','?')}  color={row.get('color','?')}  "
              f"clarity={row.get('clarity','?')}")
        print(f"       path={path}  size={sz} bytes")

    # ── Sample 5 orphan CSV rows ─────────────────────────────────
    print(f"\n{'─'*55}")
    print("  h) Sample orphan CSV rows (5):")
    if orphan_csv_ids:
        sample_orphan_csv = random.sample(list(orphan_csv_ids), min(5, len(orphan_csv_ids)))
        for did in sample_orphan_csv:
            row = all_ids[did]
            print(f"     diamond_id={did}  image_url={row.get('image_url','')[:60]}")
    else:
        print("     (none)")

    # ── Sample 5 orphan images ───────────────────────────────────
    print(f"\n{'─'*55}")
    print("  i) Sample orphan images (5):")
    if orphan_image_ids:
        sample_orphan_img = random.sample(list(orphan_image_ids), min(5, len(orphan_image_ids)))
        for did in sample_orphan_img:
            print(f"     {img_map[did]}")
    else:
        print("     (none)")

    # ── PASS / WARN / FAIL ───────────────────────────────────────
    print(f"\n{'='*55}")
    if match_rate >= 99 and orphan_img_pct < 1:
        result = "PASS"
    elif match_rate >= 95:
        result = "WARN"
    else:
        result = "FAIL"
    print(f"  RESULT: {result}")
    print(f"  Match rate {match_rate:.2f}%  |  Orphan image rate {orphan_img_pct:.2f}%")
    print("=" * 55)

    # ── Save JSON report ─────────────────────────────────────────
    report = {
        "csv_natural":        len(nat_rows),
        "csv_lab":            len(lab_rows),
        "csv_total":          total_csv,
        "images_on_disk":     total_images,
        "matched":            len(matched_ids),
        "orphan_csv":         len(orphan_csv_ids),
        "orphan_images":      len(orphan_image_ids),
        "url_set_but_missing":len(url_but_missing),
        "no_url":             len(no_url),
        "match_rate_pct":     round(match_rate, 4),
        "orphan_image_pct":   round(orphan_img_pct, 4),
        "tier_counts":        dict(tier_counts),
        "result":             result,
    }
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {REPORT}")


if __name__ == "__main__":
    main()
