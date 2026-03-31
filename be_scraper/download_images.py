"""
download_images.py — Download Brilliant Earth diamond images
DLCGIPG project

Reads image_url from output/diamonds_{natural|lab}_raw.csv and downloads
each image to output/images/{diamond_id}.jpg using plain requests.

The image CDN (image.brilliantearth.com) is NOT behind Cloudflare, so
no browser automation is needed here — plain HTTP works fine.

Can be run in parallel with scrape.py once ~500 rows exist.
Resumes safely — skips already-downloaded images.

Usage:
    python download_images.py                     # natural diamonds
    python download_images.py --lab               # lab-grown diamonds
    python download_images.py --workers 8         # parallel workers (default: 6)
    python download_images.py --limit 5000        # first N rows only
    python download_images.py --missing           # retry failed only
    python download_images.py --csv path/to/f.csv # explicit CSV path

NOTE: If image_url is empty for many rows, run scrape.py --probe first
to confirm the image field name, then update parse_diamond() in scrape.py.
"""

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output")
IMAGES_DIR = OUTPUT_DIR / "images"
FAILED_LOG = OUTPUT_DIR / "download_failures.csv"
MIN_SIZE   = 4_000   # bytes — reject blank/broken images

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.brilliantearth.com/",
    "Accept":  "image/webp,image/apng,image/*,*/*;q=0.8",
}

IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# Load CSV
# ─────────────────────────────────────────────────────────────────────
def load_diamonds(csv_path: Path, limit: int = 0):
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run scrape.py first.")
        sys.exit(1)

    rows = []
    empty_img = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("image_url"):
                rows.append(row)
            else:
                empty_img += 1
            if limit and len(rows) >= limit:
                break

    print(f"Loaded {len(rows):,} rows with image URLs from {csv_path.name}")
    if empty_img:
        print(
            f"  ({empty_img:,} rows skipped — empty image_url; "
            f"check parse_diamond() field names after --probe)"
        )
    return rows


# ─────────────────────────────────────────────────────────────────────
# Image path on disk
# ─────────────────────────────────────────────────────────────────────
def image_path(row: dict) -> Path:
    return IMAGES_DIR / f"{row['diamond_id']}.jpg"


# ─────────────────────────────────────────────────────────────────────
# Download one image
# ─────────────────────────────────────────────────────────────────────
def download_image(row: dict, session: requests.Session):
    """Returns (diamond_id, status, detail) where status ∈ {ok, skip, fail}."""
    path = image_path(row)
    did  = row["diamond_id"]

    # Already on disk and valid — skip
    if path.exists() and path.stat().st_size >= MIN_SIZE:
        return did, "skip", None

    url = row.get("image_url", "").strip()
    if not url:
        return did, "fail", "no_url"

    try:
        r = session.get(url, timeout=20, stream=True)
        if r.status_code != 200:
            return did, "fail", f"http_{r.status_code}"

        ct = r.headers.get("content-type", "")
        if "image" not in ct:
            return did, "fail", f"bad_content_type:{ct[:60]}"

        data = r.content
        if len(data) < MIN_SIZE:
            return did, "fail", f"too_small:{len(data)}b"

        path.write_bytes(data)
        return did, "ok", url

    except Exception as e:
        return did, "fail", str(e)[:120]


# ─────────────────────────────────────────────────────────────────────
# Progress stats
# ─────────────────────────────────────────────────────────────────────
class Stats:
    def __init__(self):
        self.ok = self.skip = self.fail = 0
        self.start = time.time()

    def report(self, total: int):
        elapsed = time.time() - self.start
        done    = self.ok + self.skip + self.fail
        rate    = done / elapsed if elapsed > 0 else 0
        eta     = (total - done) / rate if rate > 0 else 0
        pct     = 100 * done / total if total else 0
        print(
            f"\r  {done:,}/{total:,} ({pct:.1f}%)  "
            f"ok={self.ok:,} skip={self.skip:,} fail={self.fail:,}  "
            f"{rate:.1f}/s  ETA={eta/60:.1f}min     ",
            end="", flush=True,
        )


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download Brilliant Earth diamond images from scraped CSV"
    )
    parser.add_argument("--lab",     action="store_true",
                        help="Use lab-grown CSV (default: natural)")
    parser.add_argument("--csv",     default=None,
                        help="Explicit path to CSV (overrides --lab)")
    parser.add_argument("--workers", type=int, default=6,
                        help="Parallel download workers (default: 6)")
    parser.add_argument("--limit",   type=int, default=0,
                        help="Only process first N rows (0 = all)")
    parser.add_argument("--missing", action="store_true",
                        help="Only download rows whose image is missing on disk")
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
    else:
        prefix   = "lab" if args.lab else "natural"
        csv_path = OUTPUT_DIR / f"diamonds_{prefix}_raw.csv"

    rows = load_diamonds(csv_path, args.limit)

    if args.missing:
        rows = [r for r in rows
                if not (image_path(r).exists()
                        and image_path(r).stat().st_size >= MIN_SIZE)]
        print(f"Missing images to fetch: {len(rows):,}")

    if not rows:
        print("Nothing to download.")
        return

    # Per-worker sessions (thread-safe: one session per worker)
    sessions = []
    for _ in range(args.workers):
        s = requests.Session()
        s.headers.update(HEADERS)
        sessions.append(s)

    stats    = Stats()
    failures = []

    def worker(i_row):
        idx, row = i_row
        return download_image(row, sessions[idx % args.workers])

    print(f"Output dir: {IMAGES_DIR}")
    print(f"Downloading {len(rows):,} images with {args.workers} workers ...")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(worker, (i, r)): r for i, r in enumerate(rows)}
        for fut in as_completed(futures):
            did, status, detail = fut.result()
            if status == "ok":
                stats.ok += 1
            elif status == "skip":
                stats.skip += 1
            else:
                stats.fail += 1
                failures.append({
                    "diamond_id": did,
                    "reason":     detail,
                    "image_url":  futures[fut].get("image_url", ""),
                })
            stats.report(len(rows))

    print()  # newline after progress bar

    if failures:
        with open(FAILED_LOG, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["diamond_id", "reason", "image_url"])
            w.writeheader()
            w.writerows(failures)
        print(f"Failures logged: {FAILED_LOG}  ({len(failures):,} rows)")
        print("  Retry with: python download_images.py --missing")

    valid = sum(1 for p in IMAGES_DIR.glob("*.jpg") if p.stat().st_size >= MIN_SIZE)
    print(f"\nDownload complete.")
    print(f"  Valid images on disk: {valid:,}")
    print(f"  Output dir:           {IMAGES_DIR}")


if __name__ == "__main__":
    main()
