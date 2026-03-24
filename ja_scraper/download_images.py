"""
download_images.py — Download diamond images from scraped CSV
CSC-481 Stage 2 dataset collection

Reads: output/diamonds_raw.csv  (must exist and have image_url column)
Reads: output/diamonds_labeled.csv  (if exists, uses tier-named folders)
Writes: output/images/{tier}/{diamond_id}.jpg

Can run in parallel with scrape.py once ~500 rows exist.
Resumes safely — skips already-downloaded images.

Usage:
    python download_images.py
    python download_images.py --workers 8   # parallel downloads (default: 6)
    python download_images.py --limit 5000  # first N rows only
    python download_images.py --missing     # only retry failed downloads
"""

import argparse
import csv
import os
import sys
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
OUTPUT_DIR   = Path("output")
IMAGES_DIR   = OUTPUT_DIR / "images"
RAW_CSV      = OUTPUT_DIR / "diamonds_natural_raw.csv"
LABELED_CSV  = OUTPUT_DIR / "diamonds_labeled.csv"
FAILED_LOG   = OUTPUT_DIR / "download_failures.csv"

# Image dimensions JA uses — we want the highest res still image
IMAGE_CDN_PATTERNS = [
    # Modern JA CDN — still image at high resolution
    "https://cdn1.jamesallen.com/rings/{SHAPE_CODE}/{ID}/still-hi.jpg",
    "https://cdn1.jamesallen.com/rings/{SHAPE_CODE}/{ID}/still.jpg",
    "https://cdn1.jamesallen.com/rings/{SHAPE_CODE}/{ID}/image.jpg",
    # Fallback: use scraped image_url directly
    "{IMAGE_URL}",
]

SHAPE_CODES = {
    "round": "RD", "princess": "PR", "cushion": "CU", "oval": "OV",
    "emerald": "EM", "pear": "PE", "marquise": "MQ", "radiant": "RA",
    "asscher": "AS", "heart": "HT",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.jamesallen.com/",
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
}

MIN_FILE_SIZE = 5_000   # bytes — reject if smaller (broken/placeholder image)


# ─────────────────────────────────────────
# Setup directories
# ─────────────────────────────────────────
TIER_DIRS = {
    "budget":            IMAGES_DIR / "budget",
    "mid_range":         IMAGES_DIR / "mid_range",
    "premium":           IMAGES_DIR / "premium",
    "investment_grade":  IMAGES_DIR / "investment_grade",
    "unlabeled":         IMAGES_DIR / "unlabeled",
}

for d in TIER_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────
# Load CSV
# ─────────────────────────────────────────
def load_diamonds(csv_path_override=None, limit=0):
    # Prefer provided path, then labeled, then default natural raw
    csv_path = csv_path_override or (LABELED_CSV if LABELED_CSV.exists() else RAW_CSV)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run scrape.py first.")
        sys.exit(1)

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("image_url"):
                rows.append(row)
            if limit and len(rows) >= limit:
                break

    print(f"Loaded {len(rows)} rows from {csv_path.name}")
    return rows


# ─────────────────────────────────────────
# Determine output path for a diamond
# ─────────────────────────────────────────
def image_path(row):
    tier = row.get("value_tier", "unlabeled") or "unlabeled"
    tier = tier.lower().replace(" ", "_").replace("-", "_")
    folder = TIER_DIRS.get(tier, TIER_DIRS["unlabeled"])
    return folder / f"{row['diamond_id']}.jpg"


# ─────────────────────────────────────────
# Build candidate image URLs
# ─────────────────────────────────────────
def candidate_urls(row):
    did = row["diamond_id"]
    shape = row.get("shape", "round").lower().replace("_cut", "")
    shape_code = SHAPE_CODES.get(shape, "RD")
    image_url = row.get("image_url", "")

    urls = []
    for pattern in IMAGE_CDN_PATTERNS:
        url = (pattern
               .replace("{ID}", did)
               .replace("{SHAPE_CODE}", shape_code)
               .replace("{IMAGE_URL}", image_url))
        if url and url.startswith("http"):
            urls.append(url)

    return urls


# ─────────────────────────────────────────
# Download one image
# ─────────────────────────────────────────
def download_image(row, session):
    path = image_path(row)
    did = row["diamond_id"]

    # Skip if already downloaded
    if path.exists() and path.stat().st_size >= MIN_FILE_SIZE:
        return did, "skip", None

    urls = candidate_urls(row)
    if not urls:
        return did, "fail", "no_urls"

    for url in urls:
        try:
            r = session.get(url, timeout=20, stream=True)
            if r.status_code != 200:
                continue
            ct = r.headers.get("content-type", "")
            if "image" not in ct:
                continue
            data = r.content
            if len(data) < MIN_FILE_SIZE:
                continue
            path.write_bytes(data)
            return did, "ok", url
        except Exception:
            pass
        time.sleep(0.1)

    return did, "fail", f"tried {len(urls)} urls"


# ─────────────────────────────────────────
# Stats tracking
# ─────────────────────────────────────────
class Stats:
    def __init__(self):
        self.ok = 0
        self.skip = 0
        self.fail = 0
        self.start = time.time()

    def report(self, total):
        elapsed = time.time() - self.start
        rate = (self.ok + self.skip) / elapsed if elapsed > 0 else 0
        done = self.ok + self.skip + self.fail
        pct = 100 * done / total if total else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(
            f"\r  {done}/{total} ({pct:.1f}%)  "
            f"ok={self.ok} skip={self.skip} fail={self.fail}  "
            f"rate={rate:.1f}/s  ETA={eta/60:.1f}min     ",
            end="", flush=True
        )


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     type=str, default=None, help="Path to CSV (e.g. output/diamonds_lab_raw.csv)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit",   type=int, default=0)
    parser.add_argument("--missing", action="store_true",
                        help="Only download rows where image is missing/broken")
    args = parser.parse_args()

    csv_override = Path(args.csv) if args.csv else None
    rows = load_diamonds(csv_override, args.limit)

    if args.missing:
        rows = [r for r in rows if not (image_path(r).exists()
                and image_path(r).stat().st_size >= MIN_FILE_SIZE)]
        print(f"Missing images: {len(rows)}")

    stats = Stats()
    failures = []

    # Thread-local sessions for parallel downloads
    def make_session():
        s = requests.Session()
        s.headers.update(HEADERS)
        return s

    print(f"Starting download: {len(rows)} images, {args.workers} workers")
    print(f"Output: {IMAGES_DIR}")

    sessions = [make_session() for _ in range(args.workers)]

    def worker(i_row):
        idx, row = i_row
        session = sessions[idx % args.workers]
        return download_image(row, session)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(worker, (i, row)): row for i, row in enumerate(rows)}
        for future in as_completed(futures):
            did, status, detail = future.result()
            if status == "ok":
                stats.ok += 1
            elif status == "skip":
                stats.skip += 1
            else:
                stats.fail += 1
                failures.append({"diamond_id": did, "reason": detail,
                                  "image_url": futures[future].get("image_url", "")})
            stats.report(len(rows))

    print()  # newline after progress bar

    if failures:
        with open(FAILED_LOG, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["diamond_id", "reason", "image_url"])
            w.writeheader()
            w.writerows(failures)
        print(f"Failures logged: {FAILED_LOG} ({len(failures)} rows)")

    total = sum(1 for p in IMAGES_DIR.rglob("*.jpg") if p.stat().st_size >= MIN_FILE_SIZE)
    print(f"\nDownload complete.")
    print(f"  Valid images on disk: {total}")
    print(f"  By tier:")
    for tier, d in TIER_DIRS.items():
        n = sum(1 for p in d.glob("*.jpg") if p.stat().st_size >= MIN_FILE_SIZE)
        if n > 0:
            print(f"    {tier:20s}: {n}")
    print(f"\nNext step: python label_tiers.py  (if not run yet)")


if __name__ == "__main__":
    main()
