"""
scrape_v2.py — Production James Allen diamond scraper
CSC-481 Stage 2 dataset collection

Uses the confirmed internal GraphQL API directly via requests.
No Playwright required. No auth tokens. Just content-type + referer.

Endpoint: POST https://www.jamesallen.com/service-api/ja-product-api/diamond/v/2/
Strategy:  Sweep carat bands, all colors/cuts/clarities per band.
           ~1,900 API calls total, ~23 minutes for all metadata.
           Then run download_images.py separately for images.

Output:    output/diamonds_raw.csv  (append-safe, checkpointed)

Usage:
    python scrape_v2.py                    # full run, natural diamonds
    python scrape_v2.py --lab              # lab diamonds instead
    python scrape_v2.py --limit 5000       # test run
    python scrape_v2.py --reset            # clear checkpoint, start fresh
    python scrape_v2.py --shapes all       # all shapes (default: round only)
"""

import argparse
import csv
import json
import time
import random
import sys
from datetime import datetime
from pathlib import Path

import requests

# ─────────────────────────────────────────────────────────────────────
# CONFIRMED ID MAPPINGS  (from probe3.py item schema)
# ─────────────────────────────────────────────────────────────────────
COLOR_IDS = {
    "D": 1, "E": 2, "F": 3, "G": 4, "H": 5,
    "I": 6, "J": 7, "K": 8, "L": 9, "M": 10,
}
CLARITY_IDS = {
    "FL":   1, "IF":  2, "VVS1": 3, "VVS2": 4,
    "VS1":  5, "VS2": 6, "SI1":  7, "SI2":  8,
    "I1":   9, "I2": 10, "I3":  11,
}
CUT_IDS = {
    "Ideal": 1, "Excellent": 2, "Very Good": 3, "Good": 4, "Fair": 5,
}
SHAPE_IDS = {
    "round": 1, "princess": 2, "cushion": 3, "oval": 4,
    "emerald": 5, "pear": 6, "marquise": 7, "radiant": 8,
    "asscher": 9, "heart": 10,
}

# ─────────────────────────────────────────────────────────────────────
# SCRAPE CONFIG
# ─────────────────────────────────────────────────────────────────────
CONFIG = {
    # Shapes to collect
    "SHAPES": ["round"],

    # Grade ranges (inclusive) — collect full quality spectrum
    "COLOR_RANGE":   {"from": 1, "to": 7},   # D → J
    "CUT_RANGE":     {"from": 1, "to": 4},   # Ideal → Good
    "CLARITY_RANGE": {"from": 1, "to": 9},   # FL → I1

    # Carat bands — sweep in bands to keep result counts manageable
    "CARAT_BANDS": [
        (0.25, 0.40), (0.40, 0.55), (0.55, 0.70),
        (0.70, 0.90), (0.90, 1.10), (1.10, 1.40),
        (1.40, 1.80), (1.80, 2.50), (2.50, 5.00),
    ],

    # API pagination
    "PAGE_SIZE": 96,      # try 96; API will cap at its max (likely 96 or 48)

    # Rate limiting
    "RATE_LIMIT": 0.65,   # seconds between requests
    "JITTER":     0.30,   # max random jitter added

    # Natural diamonds only (False) or lab diamonds (True)
    "IS_LAB": False,

    # Image: which view to save ("stage"=front, "supperZoom"=40x, "thumb"=small)
    "IMAGE_VIEW": "supperZoom",   # _0_first_.jpg — highest detail, best for ML
}

# ─────────────────────────────────────────────────────────────────────
# GRAPHQL QUERY  (reconstructed from api_request.json)
# Full query as used by JA frontend — variables control filtering
# ─────────────────────────────────────────────────────────────────────
GQL_QUERY = """
query (
  $currency: currencies,
  $isOnSale: Boolean,
  $sort: sortBy,
  $lab: [Int],
  $price: intRange,
  $page: pager,
  $carat: floatRange,
  $color: intRange,
  $cut: intRange,
  $shapeID: [Int],
  $clarity: intRange,
  $isExpressShipping: Boolean,
  $addBannerPlaceholder: Boolean,
  $isFancy: Boolean,
  $isLabDiamond: Boolean,
  $polish: [Int],
  $symmetry: [Int],
  $flour: [Int]
) {
  searchByIDs(
    currency: $currency,
    lab: $lab,
    isOnSale: $isOnSale,
    sort: $sort,
    price: $price,
    page: $page,
    carat: $carat,
    color: $color,
    cut: $cut,
    shapeID: $shapeID,
    clarity: $clarity,
    isExpressShipping: $isExpressShipping,
    addBannerPlaceholder: $addBannerPlaceholder,
    isFancy: $isFancy,
    isLabDiamond: $isLabDiamond,
    polish: $polish,
    symmetry: $symmetry,
    flour: $flour
  ) {
    hits
    pageNumber
    numberOfPages
    total
    items {
      productID
      sku
      itemID
      title
      usdPrice
      usdSalePrice
      url
      status { id name }
      media {
        thumb
        stage
        supperZoom
        gallery
        tab
        sideView
        segomaPhotoID
        galleryDisplayType
      }
      stone {
        carat
        isLabDiamond
        certNumber
        measurements
        ratio
        depth
        tableSize
        shape   { id name }
        color   { id name isFancy }
        cut     { id name }
        clarity { id name }
        lab     { id name }
        flour   { id name fullName }
        symmetry { id name fullName }
        polish  { id name fullName }
        girdle
        culet   { id name fullName }
      }
    }
  }
}
"""

IMAGE_CDN = "https://ion.jamesallen.com/"

# ─────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
RAW_CSV    = OUTPUT_DIR / "diamonds_raw.csv"
CHECKPOINT = OUTPUT_DIR / "scrape_v2_checkpoint.json"

CSV_FIELDS = [
    "diamond_id", "product_id", "sku", "shape",
    "carat", "cut", "color", "clarity",
    "cert_lab", "cert_number",
    "price_usd", "is_lab_diamond",
    "depth_pct", "table_pct", "measurements",
    "fluorescence", "symmetry", "polish", "girdle",
    "image_url", "image_view",
    "all_media_paths",
    "product_url", "scraped_at",
]

# ─────────────────────────────────────────────────────────────────────
# HTTP SESSION
# ─────────────────────────────────────────────────────────────────────
def make_session():
    s = requests.Session()
    s.headers.update({
        "Content-Type":   "application/json",
        "Referer":        "https://www.jamesallen.com/loose-diamonds/all-diamonds/",
        "Origin":         "https://www.jamesallen.com",
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 "
            "-- CSC481-academic-research"
        ),
        "Accept":          "application/json, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "DNT":             "1",
    })
    return s


API_URL = "https://www.jamesallen.com/service-api/ja-product-api/diamond/v/2/"
_session = None

def get_session():
    global _session
    if _session is None:
        _session = make_session()
    return _session


# ─────────────────────────────────────────────────────────────────────
# RATE LIMITING
# ─────────────────────────────────────────────────────────────────────
def wait():
    t = CONFIG["RATE_LIMIT"] + random.uniform(0, CONFIG["JITTER"])
    time.sleep(t)


# ─────────────────────────────────────────────────────────────────────
# BUILD VARIABLES
# ─────────────────────────────────────────────────────────────────────
def build_variables(shape_id, carat_min, carat_max, page_num, page_size):
    return {
        "currency":     "USD",
        "isLabDiamond": CONFIG["IS_LAB"],
        "shapeID":      [shape_id],
        "color":        CONFIG["COLOR_RANGE"],
        "cut":          CONFIG["CUT_RANGE"],
        "clarity":      CONFIG["CLARITY_RANGE"],
        "carat":        {"from": carat_min, "to": carat_max},
        "page":         {"number": page_num, "size": page_size},
        "sort":         "price_asc",
        "isFancy":      False,
        "isOnSale":     None,
        "addBannerPlaceholder": False,
    }


# ─────────────────────────────────────────────────────────────────────
# API CALL
# ─────────────────────────────────────────────────────────────────────
def call_api(variables, retries=4):
    payload = {"query": GQL_QUERY, "variables": variables}
    s = get_session()

    for attempt in range(retries):
        try:
            r = s.post(API_URL, json=payload, timeout=25)

            if r.status_code == 200:
                data = r.json()
                # Handle both nested and flat response structures
                search = (
                    data.get("data", {}).get("searchByIDs") or
                    data.get("searchByIDs")
                )
                if search:
                    return search
                # If query fields don't match, the items may be under different nesting
                _log(f"  Unexpected response structure: {list(data.keys())}")
                return None

            if r.status_code == 429:
                wait_t = 30 * (attempt + 1)
                _log(f"  429 rate-limited — sleeping {wait_t}s")
                time.sleep(wait_t)
                continue

            if r.status_code in (401, 403):
                _log(f"  {r.status_code} auth error — API may need session cookie")
                _log(f"  Response: {r.text[:200]}")
                return None

            _log(f"  HTTP {r.status_code} — {r.text[:150]}")
            time.sleep(2 ** attempt)

        except requests.exceptions.RequestException as e:
            _log(f"  Request error attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)

    return None


# ─────────────────────────────────────────────────────────────────────
# PARSE ONE ITEM → CSV ROW
# ─────────────────────────────────────────────────────────────────────
def parse_item(item):
    """Extract all fields from a diamond item using confirmed field paths."""
    stone = item.get("stone") or {}
    media = item.get("media") or {}

    # Core IDs
    sku        = str(item.get("sku") or item.get("itemID") or "")
    product_id = str(item.get("productID") or "")

    # Stone attributes — use .name for human-readable values
    shape     = (stone.get("shape")   or {}).get("name", "")
    cut       = (stone.get("cut")     or {}).get("name", "")
    color     = (stone.get("color")   or {}).get("name", "")
    clarity   = (stone.get("clarity") or {}).get("name", "")
    lab       = (stone.get("lab")     or {}).get("name", "")
    flour     = (stone.get("flour")   or {}).get("name", "")
    symmetry  = (stone.get("symmetry") or {}).get("name", "")
    polish    = (stone.get("polish")   or {}).get("name", "")

    carat     = stone.get("carat")
    depth     = stone.get("depth")
    table_sz  = stone.get("tableSize")
    measures  = stone.get("measurements", "")
    girdle    = stone.get("girdle", "")
    cert_num  = stone.get("certNumber", "")
    is_lab    = stone.get("isLabDiamond", False)

    # Price — prefer usdPrice (currency-normalized)
    price = item.get("usdPrice") or item.get("price")

    # Image URL construction
    # media.supperZoom gives the partial path, prefix with CDN base
    view = CONFIG["IMAGE_VIEW"]
    media_path = media.get(view) or media.get("stage") or media.get("thumb") or ""
    if media_path:
        # Normalize: paths start with sgmdirect/ (no leading slash)
        if media_path.startswith("/"):
            media_path = media_path.lstrip("/")
        image_url = IMAGE_CDN + media_path
    else:
        # Reconstruct from segomaPhotoID if partial paths are missing
        photo_id = media.get("segomaPhotoID")
        if photo_id and sku and carat and shape:
            carat_str = f"{float(carat):.2f}".rstrip("0").rstrip(".")
            suffix_map = {
                "supperZoom": "0", "thumb": "1", "tab": "2",
                "gallery": "3", "stage": "4", "sideView": "3",
            }
            suffix = suffix_map.get(view, "4")
            image_url = (
                f"{IMAGE_CDN}sgmdirect/photoID/{photo_id}/Diamond/{sku}/"
                f"Diamond-{shape}-{carat_str}-Carat_{suffix}_first_.jpg"
            )
        else:
            image_url = ""

    # Collect all available media paths for reference
    media_paths = {
        k: IMAGE_CDN + v if v and not v.startswith("http") else v
        for k, v in media.items()
        if isinstance(v, str) and ("sgmdirect" in v or "certs" in v)
    }

    # Product page URL
    url_path = item.get("url", "")
    product_url = (
        f"https://www.jamesallen.com/{url_path}"
        if url_path and not url_path.startswith("http")
        else url_path or f"https://www.jamesallen.com/loose-diamonds/{shape}/{sku}-sku/"
    )

    if not sku:
        return None

    return {
        "diamond_id":      sku,
        "product_id":      product_id,
        "sku":             sku,
        "shape":           shape,
        "carat":           carat,
        "cut":             cut,
        "color":           color,
        "clarity":         clarity,
        "cert_lab":        lab,
        "cert_number":     cert_num,
        "price_usd":       price,
        "is_lab_diamond":  is_lab,
        "depth_pct":       depth,
        "table_pct":       table_sz,
        "measurements":    measures,
        "fluorescence":    flour,
        "symmetry":        symmetry,
        "polish":          polish,
        "girdle":          girdle,
        "image_url":       image_url,
        "image_view":      view,
        "all_media_paths": json.dumps(media_paths),
        "product_url":     product_url,
        "scraped_at":      datetime.utcnow().isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────────────────────────────
def load_checkpoint():
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {"completed_bands": [], "total_collected": 0}


def save_checkpoint(state):
    with open(CHECKPOINT, "w") as f:
        json.dump(state, f, indent=2)


def load_seen_ids():
    if not RAW_CSV.exists():
        return set()
    seen = set()
    with open(RAW_CSV, newline="") as f:
        for row in csv.DictReader(f):
            seen.add(row.get("diamond_id", ""))
    return seen


# ─────────────────────────────────────────────────────────────────────
# CSV WRITER
# ─────────────────────────────────────────────────────────────────────
def get_writer():
    write_header = not RAW_CSV.exists() or RAW_CSV.stat().st_size == 0
    f = open(RAW_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if write_header:
        writer.writeheader()
        f.flush()
    return f, writer


# ─────────────────────────────────────────────────────────────────────
# SCRAPE ONE CARAT BAND
# ─────────────────────────────────────────────────────────────────────
def scrape_band(shape, shape_id, carat_min, carat_max,
                seen_ids, writer, file_handle, limit=0, total_so_far=0):
    page_size   = CONFIG["PAGE_SIZE"]
    page_num    = 1
    band_count  = 0
    actual_size = page_size  # may be reduced after first response

    while True:
        variables = build_variables(shape_id, carat_min, carat_max, page_num, actual_size)
        wait()
        result = call_api(variables)

        if not result:
            _log(f"  No result — skipping remainder of band")
            break

        # On first page: adapt to actual page size returned
        items_raw = result.get("items", [])
        n_pages   = result.get("numberOfPages", 1)
        hits      = result.get("hits", 0)

        # items may be list-of-lists (probe showed nested structure)
        items = []
        for item in items_raw:
            if isinstance(item, list):
                items.extend(item)
            elif isinstance(item, dict):
                items.append(item)

        if not items:
            break

        # Adapt page size to actual items returned on first page
        if page_num == 1:
            actual_size = len(items)
            if actual_size < page_size:
                _log(f"  API returned {actual_size} items/page (requested {page_size})")

        new_count = 0
        for item in items:
            row = parse_item(item)
            if not row:
                continue
            did = row["diamond_id"]
            if did and did not in seen_ids:
                writer.writerow(row)
                seen_ids.add(did)
                new_count += 1
                band_count += 1
                total_so_far += 1

        file_handle.flush()

        _log(
            f"  p{page_num:>3}/{n_pages}  "
            f"+{new_count:>3} new  |  band={band_count}  total={total_so_far}"
            + (f"  (hits={hits})" if page_num == 1 else "")
        )

        if page_num >= n_pages:
            break

        if limit and total_so_far >= limit:
            break

        page_num += 1

    return band_count, total_so_far


# ─────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────
def _log(msg):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="James Allen Diamond Scraper v2")
    parser.add_argument("--limit",  type=int, default=0, help="Stop after N diamonds total")
    parser.add_argument("--reset",  action="store_true", help="Clear checkpoint and CSV")
    parser.add_argument("--lab",    action="store_true", help="Collect lab diamonds instead")
    parser.add_argument("--shapes", default="round",
                        help="Comma-separated shapes: round,oval,cushion,... or 'all'")
    args = parser.parse_args()

    if args.reset:
        CHECKPOINT.unlink(missing_ok=True)
        RAW_CSV.unlink(missing_ok=True)
        _log("Checkpoint and CSV cleared — starting fresh")

    if args.lab:
        CONFIG["IS_LAB"] = True
        _log("Lab diamond mode enabled")

    if args.shapes == "all":
        CONFIG["SHAPES"] = list(SHAPE_IDS.keys())
    else:
        CONFIG["SHAPES"] = [s.strip() for s in args.shapes.split(",")]

    # ── Load state ───────────────────────────────────────────────────
    state       = load_checkpoint()
    seen_ids    = load_seen_ids()
    completed   = set(tuple(b) for b in state["completed_bands"])
    total       = state["total_collected"]

    _log(f"Resuming: {len(seen_ids)} existing, {len(completed)} bands done")
    _log(f"Shapes: {CONFIG['SHAPES']}")
    _log(f"IS_LAB: {CONFIG['IS_LAB']}  |  IMAGE_VIEW: {CONFIG['IMAGE_VIEW']}")

    f, writer = get_writer()

    try:
        for shape in CONFIG["SHAPES"]:
            shape_id = SHAPE_IDS.get(shape)
            if not shape_id:
                _log(f"Unknown shape '{shape}' — skipping")
                continue

            for carat_min, carat_max in CONFIG["CARAT_BANDS"]:
                band_key = (shape, carat_min, carat_max)
                if band_key in completed:
                    _log(f"SKIP {shape} {carat_min:.2f}–{carat_max:.2f}ct (done)")
                    continue

                _log(f"\n── {shape.upper()}  {carat_min:.2f}–{carat_max:.2f}ct ──")

                n, total = scrape_band(
                    shape, shape_id, carat_min, carat_max,
                    seen_ids, writer, f,
                    limit=args.limit, total_so_far=total,
                )

                _log(f"  Band done: +{n}  →  running total: {total}")

                completed.add(band_key)
                state["completed_bands"]  = [list(b) for b in completed]
                state["total_collected"]  = total
                save_checkpoint(state)

                if args.limit and total >= args.limit:
                    _log(f"Limit {args.limit} reached — stopping")
                    break

            if args.limit and total >= args.limit:
                break

    except KeyboardInterrupt:
        _log("Interrupted — checkpoint saved safely")
    finally:
        f.close()

    _log(f"\nFinal total: {total} diamonds in {RAW_CSV}")
    _log(f"Next step:   python label_tiers.py  (assign value tier labels)")
    _log(f"Then:        python download_images.py  (download images)")


if __name__ == "__main__":
    main()
