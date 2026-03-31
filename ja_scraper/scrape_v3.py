"""
scrape_v3.py — Fixed James Allen diamond scraper
CSC-481 Stage 2 dataset collection

Fixes from debug_api.py output:
  - Uses EXACT query from api_request.json (not reconstructed)
  - page must include "count" field: {"count": 50, "size": 8, "number": N}
    → 50 rows × 8 items = 400 diamonds per API call
  - No "sort" field (enum value doesn't exist in their schema)
  - depth, tableSize, shippingDays, polish, symmetry, flour are required
  - items structure: list[count] of list[8] — flatten both levels

Expected: ~1,900 calls for 185k diamonds, ~23 minutes total metadata.

Usage:
    python scrape_v3.py                  # full natural diamond run
    python scrape_v3.py --limit 500      # test run
    python scrape_v3.py --reset          # clear checkpoint, start fresh
    python scrape_v3.py --lab            # lab diamonds
    python scrape_v3.py --shapes all     # all cuts (default: round only)
    python scrape_v3.py --count 100      # 800 items/call (if API allows)
"""

import argparse
import csv
import json
import time
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

# ─────────────────────────────────────────────────────────────────────
# LOAD EXACT QUERY FROM CAPTURED REQUEST
# ─────────────────────────────────────────────────────────────────────
_REQ_FILE = Path("api_request.json")
if not _REQ_FILE.exists():
    print("ERROR: api_request.json not found.")
    print("Run probe3.py first from this directory.")
    sys.exit(1)

_captured    = json.loads(_REQ_FILE.read_text())
GQL_QUERY    = (_captured.get("body") or {}).get("query", "")
if not GQL_QUERY:
    print("ERROR: No query in api_request.json. Re-run probe3.py.")
    sys.exit(1)

print(f"[init] Loaded GQL query: {len(GQL_QUERY)} chars from api_request.json")

# ─────────────────────────────────────────────────────────────────────
# SHAPE / GRADE ID MAPPINGS
# ─────────────────────────────────────────────────────────────────────
SHAPE_IDS = {
    "round": 1, "princess": 2, "cushion": 3, "oval": 4,
    "emerald": 5, "pear": 6, "marquise": 7, "radiant": 8,
    "asscher": 9, "heart": 10,
}

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
CONFIG = {
    "SHAPES": ["round"],

    # Full grade spectrum
    "COLOR_RANGE":   {"from": 1, "to": 10},   # D → M  (covers whole spectrum)
    "CUT_RANGE":     {"from": 1, "to": 5},    # Ideal → Fair
    "CLARITY_RANGE": {"from": 1, "to": 11},   # FL → I3

    # Required filter fields — use wide/permissive defaults from original request
    "DEPTH_RANGE":     {"from": 40, "to": 85},
    "TABLE_RANGE":     {"from": 40, "to": 90},
    "SHIPPING_DAYS":   999,
    "POLISH":          [5, 4, 3, 2, 1],       # all grades
    "SYMMETRY":        [5, 4, 3, 2, 1],
    "FLOUR":           [8, 5, 4, 3, 2, 1],    # all fluorescence

    # Carat bands — micro-bands (0.01ct) to stay under 400-item API limit
    "CARAT_BANDS": [], # populated in main()

    # Pagination: count × size = items per call
    # size=8 appears fixed by JA. count=50 → 400 per call.
    "PAGE_COUNT": 50,
    "PAGE_SIZE":  8,

    # Rate limiting
    "RATE_LIMIT": 0.65,
    "JITTER":     0.35,

    "IS_LAB": False,
    "IMAGE_VIEW": "supperZoom",   # _0_first_.jpg = highest detail
}

# ─────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
RAW_CSV    = None
CHECKPOINT = None

CSV_FIELDS = [
    "diamond_id", "product_id", "sku", "shape",
    "carat", "cut", "color", "clarity",
    "cert_lab", "cert_number",
    "price_usd", "is_lab_diamond",
    "depth_pct", "table_pct", "measurements",
    "fluorescence", "symmetry", "polish", "girdle",
    "image_url", "image_view",
    "product_url", "scraped_at",
]

IMAGE_CDN = "https://ion.jamesallen.com/"
API_URL   = "https://www.jamesallen.com/service-api/ja-product-api/diamond/v/2/"

# ─────────────────────────────────────────────────────────────────────
# SESSION
# ─────────────────────────────────────────────────────────────────────
def make_session():
    s = requests.Session()
    s.headers.update({
        "Content-Type":    "application/json",
        "Referer":         "https://www.jamesallen.com/loose-diamonds/all-diamonds/",
        "Origin":          "https://www.jamesallen.com",
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

_session = None
def get_session():
    global _session
    if _session is None:
        _session = make_session()
    return _session

def wait():
    time.sleep(CONFIG["RATE_LIMIT"] + random.uniform(0, CONFIG["JITTER"]))

# ─────────────────────────────────────────────────────────────────────
# BUILD VARIABLES — exact field names confirmed by debug_api.py Test 2
# ─────────────────────────────────────────────────────────────────────
def build_variables(shape_id, carat_min, carat_max, page_num):
    return {
        "currency":     "USD",
        "isLabDiamond": CONFIG["IS_LAB"],
        "isFancy":      False,
        "isOnSale":     None,
        "addBannerPlaceholder": False,
        "shapeID":      [shape_id],
        "color":        CONFIG["COLOR_RANGE"],
        "cut":          CONFIG["CUT_RANGE"],
        "clarity":      CONFIG["CLARITY_RANGE"],
        "carat":        {"from": carat_min, "to": carat_max},
        "depth":        CONFIG["DEPTH_RANGE"],
        "tableSize":    CONFIG["TABLE_RANGE"],
        "shippingDays": CONFIG["SHIPPING_DAYS"],
        "polish":       CONFIG["POLISH"],
        "symmetry":     CONFIG["SYMMETRY"],
        "flour":        CONFIG["FLOUR"],
        "price":        {"from": 0, "to": 9999999},
        "page": {
            "count":  CONFIG["PAGE_COUNT"],
            "size":   CONFIG["PAGE_SIZE"],
            "number": page_num,
        },
        # No "sort" — enum value doesn't exist in their schema
    }

# ─────────────────────────────────────────────────────────────────────
# API CALL
# ─────────────────────────────────────────────────────────────────────
def call_api(variables, retries=4):
    payload = {"query": GQL_QUERY, "variables": variables}
    s = get_session()

    for attempt in range(retries):
        try:
            r = s.post(API_URL, json=payload, timeout=30)

            if r.status_code == 200:
                try:
                    data = r.json()
                except json.JSONDecodeError:
                    # Empty/HTML response = session stale or rate-limited.
                    # Drop the session so the next get_session() creates a fresh one.
                    global _session
                    _session = None
                    pause = 20 * (attempt + 1)
                    _log(f"  JSON decode error — session reset, sleeping {pause}s (attempt {attempt+1}/{retries})")
                    time.sleep(pause)
                    s = get_session()
                    continue

                # Check for GraphQL errors
                if "errors" in data:
                    errs = data["errors"]
                    _log(f"  GQL errors: {json.dumps(errs)[:200]}")
                    return None

                search = (data.get("data") or {}).get("searchByIDs")
                if search:
                    return search

                _log(f"  No searchByIDs in response. Keys: {list((data.get('data') or data).keys())}")
                return None

            if r.status_code == 429:
                pause = 30 * (attempt + 1)
                _log(f"  429 rate-limited — sleeping {pause}s")
                time.sleep(pause)
                continue

            if r.status_code == 500:
                _log(f"  500 — likely bad variable. Response: {r.text[:200]}")
                return None

            _log(f"  HTTP {r.status_code}: {r.text[:100]}")
            time.sleep(2 ** attempt)

        except requests.exceptions.RequestException as e:
            _log(f"  Request error attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)

    return False

# ─────────────────────────────────────────────────────────────────────
# PARSE ONE ITEM
# ─────────────────────────────────────────────────────────────────────
def parse_item(item):
    if not isinstance(item, dict):
        return None

    stone = item.get("stone") or {}
    media = item.get("media") or {}

    sku        = str(item.get("sku") or item.get("itemID") or "")
    product_id = str(item.get("productID") or "")
    if not sku:
        return None

    shape    = (stone.get("shape")    or {}).get("name", "")
    cut      = (stone.get("cut")      or {}).get("name", "")
    color    = (stone.get("color")    or {}).get("name", "")
    clarity  = (stone.get("clarity")  or {}).get("name", "")
    lab      = (stone.get("lab")      or {}).get("name", "")
    flour    = (stone.get("flour")    or {}).get("name", "")
    symmetry = (stone.get("symmetry") or {}).get("name", "")
    polish   = (stone.get("polish")   or {}).get("name", "")

    carat    = stone.get("carat")
    depth    = stone.get("depth")
    table_sz = stone.get("tableSize")
    measures = stone.get("measurements", "")
    girdle   = stone.get("girdle", "")
    cert_num = stone.get("certNumber", "")
    is_lab   = stone.get("isLabDiamond", False)

    price = item.get("usdPrice") or item.get("price")

    # Image URL — supperZoom (_0_first_) is 40x magnification, best for ML
    view       = CONFIG["IMAGE_VIEW"]
    media_path = media.get(view) or media.get("stage") or media.get("thumb") or ""

    if media_path:
        media_path = media_path.lstrip("/")
        image_url = IMAGE_CDN + media_path
    else:
        # Reconstruct from parts if path is missing
        photo_id = media.get("segomaPhotoID")
        if photo_id and sku and carat and shape:
            carat_str = f"{float(carat):.2f}".rstrip("0").rstrip(".")
            suffix    = {"supperZoom": "0", "thumb": "1",
                         "tab": "2", "gallery": "3",
                         "stage": "4"}.get(view, "4")
            image_url = (
                f"{IMAGE_CDN}sgmdirect/photoID/{photo_id}/Diamond/{sku}/"
                f"Diamond-{shape}-{carat_str}-Carat_{suffix}_first_.jpg"
            )
        else:
            image_url = ""

    url_path    = item.get("url", "")
    product_url = (
        f"https://www.jamesallen.com/{url_path.lstrip('/')}"
        if url_path else
        f"https://www.jamesallen.com/loose-diamonds/{shape}-cut/{sku}-sku/"
    )

    return {
        "diamond_id":     sku,
        "product_id":     product_id,
        "sku":            sku,
        "shape":          shape,
        "carat":          carat,
        "cut":            cut,
        "color":          color,
        "clarity":        clarity,
        "cert_lab":       lab,
        "cert_number":    cert_num,
        "price_usd":      price,
        "is_lab_diamond": is_lab,
        "depth_pct":      depth,
        "table_pct":      table_sz,
        "measurements":   measures,
        "fluorescence":   flour,
        "symmetry":       symmetry,
        "polish":         polish,
        "girdle":         girdle,
        "image_url":      image_url,
        "image_view":     view,
        "product_url":    product_url,
        "scraped_at":     datetime.now(timezone.utc).isoformat(),
    }

# ─────────────────────────────────────────────────────────────────────
# FLATTEN ITEMS — API returns list[count] of list[size]
# ─────────────────────────────────────────────────────────────────────
def flatten_items(raw_items):
    flat = []
    for item in (raw_items or []):
        if isinstance(item, list):
            flat.extend(item)
        elif isinstance(item, dict):
            flat.append(item)
    return flat

# ─────────────────────────────────────────────────────────────────────
# CHECKPOINT + CSV
# ─────────────────────────────────────────────────────────────────────
def load_checkpoint():
    global CHECKPOINT
    if CHECKPOINT and CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {"completed_bands": [], "total_collected": 0}

def save_checkpoint(state):
    global CHECKPOINT
    with open(CHECKPOINT, "w") as f:
        json.dump(state, f, indent=2)

def load_seen_ids():
    global RAW_CSV
    if not RAW_CSV or not RAW_CSV.exists():
        return set()
    seen = set()
    with open(RAW_CSV, newline="") as f:
        for row in csv.DictReader(f):
            seen.add(row.get("diamond_id", ""))
    return seen

def get_writer():
    global RAW_CSV
    write_header = not RAW_CSV.exists() or RAW_CSV.stat().st_size == 0
    f = open(RAW_CSV, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if write_header:
        w.writeheader()
        f.flush()
    return f, w

# ─────────────────────────────────────────────────────────────────────
# SCRAPE ONE BAND
# ─────────────────────────────────────────────────────────────────────
def scrape_band(shape, shape_id, carat_min, carat_max,
                seen_ids, writer, file_handle, limit, total):

    page_num   = 1
    band_total = 0
    items_per_page = CONFIG["PAGE_COUNT"] * CONFIG["PAGE_SIZE"]  # 50 × 8 = 400

    while True:
        variables = build_variables(shape_id, carat_min, carat_max, page_num)
        wait()
        result = call_api(variables)

        if result is False:
            _log(f"  API failure (JSON error) — skipping band completion")
            return band_total, total, False
        if result is None:
            _log(f"  API returned None — stopping band")
            break

        n_pages  = result.get("numberOfPages", 1)
        hits     = result.get("hits", 0)
        raw_items = result.get("items", [])
        items    = flatten_items(raw_items)

        if not items:
            if page_num == 1:
                _log(f"  0 items on page 1 (hits={hits}) — band empty")
            break

        new = 0
        for item in items:
            row = parse_item(item)
            if not row:
                continue
            did = row["diamond_id"]
            if did and did not in seen_ids:
                writer.writerow(row)
                seen_ids.add(did)
                new += 1
                band_total += 1
                total += 1

        file_handle.flush()

        _log(
            f"  p{page_num:>3}/{n_pages}  "
            f"got={len(items):>3}  +new={new:>3}  "
            f"band={band_total}  total={total}"
            + (f"  hits={hits}" if page_num == 1 else "")
        )

        if page_num >= n_pages or len(items) < items_per_page:
            break

        if limit and total >= limit:
            break

        page_num += CONFIG["PAGE_COUNT"]

    return band_total, total, True

# ─────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────
def _log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)

# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    global RAW_CSV, CHECKPOINT
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",  type=int, default=0)
    parser.add_argument("--reset",  action="store_true")
    parser.add_argument("--lab",    action="store_true")
    parser.add_argument("--count",  type=int, default=50,
                        help="Rows per API call (count × 8 = items). Default 50=400 items.")
    parser.add_argument("--shapes", default="round")
    args = parser.parse_args()

    prefix = "lab" if args.lab else "natural"

    CONFIG["IS_LAB"]     = args.lab
    CONFIG["PAGE_COUNT"] = args.count
    CONFIG["SHAPES"]     = (
        list(SHAPE_IDS.keys()) if args.shapes == "all"
        else [s.strip() for s in args.shapes.split(",")]
    )

    # 0.25 to 6.00 in 0.01 increments (575 bands)
    bands = []
    curr = 0.25
    while curr < 6.00:
        nxt = round(curr + 0.01, 2)
        bands.append((curr, nxt))
        curr = nxt
    CONFIG["CARAT_BANDS"] = bands

    _log(f"Config: shapes={CONFIG['SHAPES']}  lab={CONFIG['IS_LAB']}  "
         f"items/call={CONFIG['PAGE_COUNT']*CONFIG['PAGE_SIZE']}")

    try:
        for shape in CONFIG["SHAPES"]:
            shape_id = SHAPE_IDS.get(shape)
            if not shape_id:
                _log(f"Unknown shape: {shape}")
                continue

            # ── per-shape file paths ──
            RAW_CSV    = OUTPUT_DIR / f"diamonds_{prefix}_{shape}_raw.csv"
            CHECKPOINT = OUTPUT_DIR / f"scrape_v3_{prefix}_{shape}_checkpoint.json"

            if args.reset:
                CHECKPOINT.unlink(missing_ok=True)
                RAW_CSV.unlink(missing_ok=True)
                _log(f"[{shape}] Cleared checkpoint and CSV")

            state     = load_checkpoint()
            seen_ids  = load_seen_ids()
            completed = set(tuple(b) for b in state["completed_bands"])
            total     = state["total_collected"]

            _log(f"\n{'='*55}")
            _log(f"SHAPE: {shape.upper()}  →  {RAW_CSV.name}")
            _log(f"Resume: {len(seen_ids):,} existing  {len(completed)} bands done")

            f, writer = get_writer()
            shape_interrupted = False

            try:
                for cmin, cmax in CONFIG["CARAT_BANDS"]:
                    band_key = (cmin, cmax)
                    if band_key in completed:
                        continue

                    _log(f"\n── {shape.upper()}  {cmin:.2f}–{cmax:.2f}ct ──")

                    n, total, success = scrape_band(
                        shape, shape_id, cmin, cmax,
                        seen_ids, writer, f,
                        limit=args.limit, total=total,
                    )

                    _log(f"  Band done +{n}  running total={total}")

                    if not success:
                        _log(f"  Band FAILED — not marking as completed")
                        continue

                    completed.add(band_key)
                    state["completed_bands"] = [list(b) for b in completed]
                    state["total_collected"] = total
                    save_checkpoint(state)

                    if args.limit and total >= args.limit:
                        _log(f"Limit {args.limit} reached")
                        break

            except KeyboardInterrupt:
                _log(f"[{shape}] Interrupted — checkpoint saved")
                shape_interrupted = True
            finally:
                f.close()

            n_rows = sum(1 for _ in open(str(RAW_CSV))) - 1 if RAW_CSV.exists() else 0
            _log(f"\n[{shape}] Done. {n_rows:,} diamonds → {RAW_CSV.name}")

            if shape_interrupted:
                break

    except KeyboardInterrupt:
        _log("Interrupted — all shape checkpoints saved")

    _log("Next: python label_tiers.py  then  python download_images.py")

if __name__ == "__main__":
    main()
