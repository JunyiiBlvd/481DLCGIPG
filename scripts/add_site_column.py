"""
add_site_column.py — Add site identifier column to both labeled CSVs
DLCGIPG project — cross-domain generalization experiment

Adds a 'site' column to:
  - be_scraper/output/be_labeled.csv        → site = "brilliant_earth"
  - ja_scraper/output/diamonds_labeled.csv  → site = "james_allen"

This enables clean DataFrame merging for cross-domain train/test splits:

    import pandas as pd
    be = pd.read_csv("be_scraper/output/be_labeled.csv")
    ja = pd.read_csv("ja_scraper/output/diamonds_labeled.csv")
    df = pd.concat([be, ja], ignore_index=True)
    # train on JA, test on BE:
    train = df[df.site == "james_allen"]
    test  = df[df.site == "brilliant_earth"]

If the column already exists its value is overwritten (idempotent).

Usage:
    python scripts/add_site_column.py
    python scripts/add_site_column.py --dry-run   # print counts without writing
"""

import argparse
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # project root

TARGETS = [
    {
        "path":  ROOT / "be_scraper" / "output" / "be_labeled.csv",
        "site":  "brilliant_earth",
    },
    {
        "path":  ROOT / "ja_scraper" / "output" / "diamonds_labeled.csv",
        "site":  "james_allen",
    },
]


def add_site(csv_path: Path, site_value: str, dry_run: bool) -> int:
    """
    Adds or overwrites the 'site' column in csv_path.
    Returns the number of rows processed.
    """
    if not csv_path.exists():
        print(f"  SKIP: {csv_path} — file not found")
        return 0

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(row)

    if not rows:
        print(f"  SKIP: {csv_path} — empty file")
        return 0

    # Ensure 'site' is in fieldnames
    if "site" not in fieldnames:
        fieldnames.append("site")

    for row in rows:
        row["site"] = site_value

    n = len(rows)

    if dry_run:
        existing_sites = set(r.get("site", "") for r in rows)
        print(f"  [dry-run] {csv_path.name}: {n:,} rows, "
              f"site column would be set to {site_value!r} "
              f"(was: {existing_sites})")
        return n

    # Write to a temp file first, then atomically replace
    tmp = csv_path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    shutil.move(str(tmp), str(csv_path))
    print(f"  {csv_path.name}: {n:,} rows updated — site={site_value!r}")
    return n


def main():
    parser = argparse.ArgumentParser(
        description="Add site column to both labeled CSVs"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing files")
    args = parser.parse_args()

    total = 0
    for target in TARGETS:
        n = add_site(target["path"], target["site"], args.dry_run)
        total += n

    if not args.dry_run and total > 0:
        print(f"\nDone. {total:,} rows updated across {len(TARGETS)} files.")
        print("Both CSVs now have site column — safe to pd.concat().")
    elif args.dry_run:
        print("\n[dry-run complete — no files written]")


if __name__ == "__main__":
    main()
