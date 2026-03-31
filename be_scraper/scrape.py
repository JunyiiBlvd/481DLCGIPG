"""
scrape.py — Brilliant Earth diamond scraper
DLCGIPG project — cross-domain generalization experiment

Uses Playwright (headless Chromium) to bypass Cloudflare's managed-challenge
protection. API calls are made via page.evaluate() fetch() inside the live
browser context, which inherits the Cloudflare session cookies.

DISCOVERED API (via probe_network.py on 2026-03-28):
  Endpoint:   GET https://www.brilliantearth.com/api/v1/plp/products/
  Natural:    product_class=Loose Diamonds
  Lab:        product_class=Lab Created Colorless Diamonds
  Shape:      shapes=Round  (single value; one shape per call)
  Pagination: display=50&page=1  (1-indexed page numbers)
  Filters:    cuts=, colors=, clarities=, min_carat=, max_carat=, etc.
  Image:      images.real_images[0].src  (real photo; prefer over images[0])

Strategy: micro-band carat sweeps (0.01ct increments) + page pagination.
Output:   one CSV + checkpoint per shape, e.g.:
            output/diamonds_natural_oval_raw.csv
            output/scrape_natural_oval_checkpoint.json

FIRST RUN — probe to verify API is reachable:
    python scrape.py --probe

NORMAL USAGE:
    python scrape.py --shapes oval                     # single shape
    python scrape.py --shapes oval,cushion,pear        # multiple shapes sequentially
    python scrape.py --shapes oval,cushion,pear --lab  # lab-grown
    python scrape.py --limit 500                       # stop after N records per shape (test)
    python scrape.py --reset --shapes oval             # wipe oval checkpoint + CSV
    python scrape.py --no-headless                     # show browser window (CF debug)
    python scrape.py --display 100                     # items per API call (default: 50)

PARALLEL RUNS (natural + lab simultaneously):
    python scrape.py --shapes oval,cushion,pear &
    python scrape.py --shapes oval,cushion,pear --lab &
"""

import argparse
import asyncio
import csv
import json
import random
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────
API_PATH     = "/api/v1/plp/products/"
PRODUCT_BASE = "https://www.brilliantearth.com"
IMAGE_BASE   = "https://image.brilliantearth.com"
NAT_URL      = "https://www.brilliantearth.com/loose-diamonds/"
LAB_URL      = "https://www.brilliantearth.com/lab-diamonds-search/"

# product_class values confirmed via network interception
PRODUCT_CLASS_NATURAL = "Loose Diamonds"
PRODUCT_CLASS_LAB     = "Lab Created Colorless Diamonds"

# Shape names — BE-canonical capitalization
ALL_SHAPES = [
    "Round", "Oval", "Cushion", "Pear",
    "Princess", "Emerald", "Marquise", "Asscher", "Radiant", "Heart",
]

# Comma-separated filter strings (confirmed from captured API calls)
CUTS         = "Fair,Good,Very Good,Ideal,Super Ideal"
COLORS       = "J,I,H,G,F,E,D"
CLARITIES    = "SI2,SI1,VS2,VS1,VVS2,VVS1,IF,FL"
POLISHES     = "Good,Very Good,Excellent"
SYMMETRIES   = "Good,Very Good,Excellent"
FLUORESCENCES= "Very Strong,Strong,Medium,Faint,None"

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# CONFIG  (mutated by main() after arg parsing)
# ─────────────────────────────────────────────────────────────────────
CONFIG = {
    "SHAPES":        ["Round"],
    "IS_LAB":        False,
    "PRODUCT_CLASS": PRODUCT_CLASS_NATURAL,
    "DISPLAY":       50,       # items per API call
    "RATE_LIMIT":    1.0,      # base delay between calls (seconds)
    "JITTER":        0.5,      # additional random delay [0, JITTER]
    "CARAT_BANDS":   [],       # populated in main()
}

# CSV field order — mirrors ja_scraper/scrape_v3.py CSV_FIELDS for
# cross-domain experiment compatibility
CSV_FIELDS = [
    "diamond_id", "sku", "shape",
    "carat", "cut", "color", "clarity",
    "cert_lab", "cert_number",
    "price_usd", "is_lab_diamond",
    "depth_pct", "table_pct", "measurements",
    "fluorescence", "symmetry", "polish", "girdle",
    "image_url", "product_url", "scraped_at",
]

# (CSV and checkpoint paths are per-shape; constructed in run_scrape_shape)


# ─────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────
def _log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────
# BROWSER SETUP
# ─────────────────────────────────────────────────────────────────────
async def make_browser(pw, headless: bool):
    """Launch Chromium with anti-detection settings."""
    browser = await pw.chromium.launch(
        headless=headless,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
        ],
    )
    context = await browser.new_context(
        viewport={"width": 1920, "height": 1080},
        user_agent=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        locale="en-US",
        timezone_id="America/New_York",
        extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
    )
    # Remove navigator.webdriver flag
    await context.add_init_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    page = await context.new_page()
    return browser, page


async def navigate_to_search(page, is_lab: bool):
    """Navigate to the BE search page to establish a Cloudflare session."""
    url = LAB_URL if is_lab else NAT_URL
    _log(f"Navigating to {url} ...")
    try:
        await page.goto(url, wait_until="networkidle", timeout=60_000)
    except PWTimeout:
        _log("  networkidle timed out — retrying with domcontentloaded")
        await page.goto(url, wait_until="domcontentloaded", timeout=30_000)

    title = await page.title()
    _log(f"  Page title: {title!r}")

    # Allow Cloudflare challenge JS to complete if needed
    await asyncio.sleep(4)

    content = await page.content()
    if "Just a moment" in content or "Enable JavaScript" in content:
        _log("  WARNING: Cloudflare challenge page. Waiting 15s ...")
        await asyncio.sleep(15)
        content = await page.content()
        if "Just a moment" in content:
            _log("  Still on challenge page. Try --no-headless to solve manually.")


# ─────────────────────────────────────────────────────────────────────
# BUILD QUERY STRING
# ─────────────────────────────────────────────────────────────────────
def _enc(val: str) -> str:
    """URL-encode a value (spaces → %20, commas → %2C)."""
    return urllib.parse.quote(str(val), safe="")


def build_params(shape: str, carat_min: float, carat_max: float,
                 page_num: int) -> str:
    """
    Build the query string for one paginated API call.

    Parameter format confirmed via network interception 2026-03-28:
    - shapes=Round                (single value, no brackets)
    - cuts=Fair,Good,...          (comma-separated string)
    - display=50&page=1           (page-based pagination, 1-indexed)
    - product_class=Loose Diamonds | Lab Created Colorless Diamonds
    """
    pc = CONFIG["PRODUCT_CLASS"]

    parts = [
        f"display={CONFIG['DISPLAY']}",
        f"page={page_num}",
        "currency=USD",
        f"product_class={_enc(pc)}",
        f"cuts={_enc(CUTS)}",
        f"colors={_enc(COLORS)}",
        f"clarities={_enc(CLARITIES)}",
        f"polishes={_enc(POLISHES)}",
        f"symmetries={_enc(SYMMETRIES)}",
        f"fluorescences={_enc(FLUORESCENCES)}",
        "real_diamond_view=",
        "quick_ship_diamond=",
        "hearts_and_arrows_diamonds=",
        "min_price=0",
        "max_price=9999999",
        "min_table=45",
        "max_table=83",
        "min_depth=43",
        "max_depth=83.2",
        "min_ratio=1",
        "max_ratio=2.75",
        f"min_carat={carat_min:.2f}",
        f"max_carat={carat_max:.2f}",
        f"shapes={_enc(shape)}",
        "order_by=price",
        "order_method=asc",
    ]
    return "&".join(parts)


# ─────────────────────────────────────────────────────────────────────
# API CALL  via page.evaluate() fetch()
# ─────────────────────────────────────────────────────────────────────
_FETCH_JS = """
    async (url) => {
        try {
            const r = await fetch(url, {
                method: 'GET',
                credentials: 'include',
                headers: {
                    'Accept': 'application/json, text/javascript, */*; q=0.01',
                    'X-Requested-With': 'XMLHttpRequest',
                }
            });
            if (!r.ok) {
                let txt = '';
                try { txt = await r.text(); } catch (_) {}
                return {__error: r.status, __text: txt.slice(0, 300)};
            }
            return await r.json();
        } catch (e) {
            return {__error: -1, __text: String(e)};
        }
    }
"""


async def call_api(page, qs: str, retries: int = 4):
    """Make one GET call to the BE API. Returns parsed JSON dict or None."""
    url = f"{API_PATH}?{qs}"

    for attempt in range(retries):
        await asyncio.sleep(CONFIG["RATE_LIMIT"] + random.uniform(0, CONFIG["JITTER"]))
        try:
            result = await page.evaluate(_FETCH_JS, url)
        except Exception as e:
            _log(f"  evaluate() exception attempt {attempt+1}: {e}")
            await asyncio.sleep(2 ** attempt)
            continue

        if isinstance(result, dict) and "__error" in result:
            status  = result["__error"]
            snippet = result.get("__text", "")[:200]
            if status == 429:
                wait = 30 * (attempt + 1)
                _log(f"  429 rate-limited — sleeping {wait}s")
                await asyncio.sleep(wait)
                continue
            if status == 403:
                _log(f"  403 — Cloudflare may have re-challenged. Re-navigating ...")
                await navigate_to_search(page, CONFIG["IS_LAB"])
                continue
            _log(f"  HTTP {status} (attempt {attempt+1}): {snippet!r}")
            await asyncio.sleep(2 ** attempt)
            continue

        return result

    return None


# ─────────────────────────────────────────────────────────────────────
# PARSE ONE DIAMOND RECORD
# ─────────────────────────────────────────────────────────────────────
def parse_diamond(d: dict):
    """
    Map one BE /api/v1/plp/products/ record → CSV row dict.

    Confirmed field names (from probe_network.py + plp_response.json):
      id, upc, shape, carat, cut, color, clarity, report,
      certificate_number, price, depth, table, measurements,
      fluorescence, symmetry, polish, girdle, product_class,
      images.real_images[0].src, images.images[0].src,
      real_diamond_image
    """
    if not isinstance(d, dict):
        return None

    did = str(d.get("id") or "").strip()
    if not did:
        return None

    sku         = str(d.get("upc") or did)
    shape       = str(d.get("shape") or "").strip()
    carat       = d.get("carat")
    cut         = str(d.get("cut") or "").strip()
    color       = str(d.get("color") or "").strip()
    clarity     = str(d.get("clarity") or "").strip()
    cert_lab    = str(d.get("report") or "").strip()
    cert_number = str(d.get("certificate_number") or "").strip()
    price_usd   = d.get("price")
    is_lab      = str(d.get("product_class") or "").lower().startswith("lab")

    depth_pct    = d.get("depth")        # already a percentage float
    table_pct    = d.get("table")
    measurements = str(d.get("measurements") or "").strip()
    fluorescence = str(d.get("fluorescence") or "").strip()
    symmetry     = str(d.get("symmetry") or "").strip()
    polish       = str(d.get("polish") or "").strip()
    girdle       = str(d.get("girdle") or "").strip()

    # Image URL — prefer real_images (actual stone photo) over product images
    image_url = ""
    imgs = d.get("images") or {}
    real = imgs.get("real_images") or []
    if real and isinstance(real[0], dict):
        src = str(real[0].get("src") or "").strip()
        if src:
            image_url = ("https:" + src) if src.startswith("//") else src

    if not image_url:
        main_imgs = imgs.get("images") or []
        if main_imgs and isinstance(main_imgs[0], dict):
            src = str(main_imgs[0].get("src") or "").strip()
            if src:
                image_url = ("https:" + src) if src.startswith("//") else src

    # Product URL — construct from diamond id
    product_url = f"{PRODUCT_BASE}/loose-diamonds/view_detail/{did}/"

    return {
        "diamond_id":     did,
        "sku":            sku,
        "shape":          shape,
        "carat":          carat,
        "cut":            cut,
        "color":          color,
        "clarity":        clarity,
        "cert_lab":       cert_lab,
        "cert_number":    cert_number,
        "price_usd":      price_usd,
        "is_lab_diamond": is_lab,
        "depth_pct":      depth_pct,
        "table_pct":      table_pct,
        "measurements":   measurements,
        "fluorescence":   fluorescence,
        "symmetry":       symmetry,
        "polish":         polish,
        "girdle":         girdle,
        "image_url":      image_url,
        "product_url":    product_url,
        "scraped_at":     datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────
# PROBE MODE — verify API is reachable and print response structure
# ─────────────────────────────────────────────────────────────────────
async def run_probe(page, is_lab: bool):
    _log("=== PROBE MODE ===")
    shape = "Round"
    qs    = build_params(shape, 1.00, 1.01, page_num=1)
    _log(f"GET {PRODUCT_BASE}{API_PATH}?{qs[:100]}...")

    resp = await call_api(page, qs)
    if resp is None:
        _log("No response received.")
        _log("Try --no-headless to debug Cloudflare interaction.")
        return

    products = resp.get("products", [])
    total    = resp.get("total")

    print("\n" + "=" * 60)
    print("API RESPONSE SUMMARY")
    print("=" * 60)
    print(f"Top-level keys : {list(resp.keys())}")
    print(f"total          : {total}")
    print(f"products count : {len(products)}")

    if products:
        p0 = products[0]
        print(f"\nFirst product keys:\n  {sorted(p0.keys())}")
        print(f"\nFirst product (raw):\n{json.dumps(p0, indent=2)[:2000]}")
        print("\n--- Parsed records ---")
        for i, d in enumerate(products[:3]):
            row = parse_diamond(d)
            print(f"\n[{i}] {'OK' if row else 'PARSE FAILED'}")
            if row:
                print(json.dumps(row, indent=2, default=str))
    else:
        print(f"\nNo products. Full response:\n{json.dumps(resp, indent=2)[:2000]}")

    probe_file = OUTPUT_DIR / "probe_response.json"
    probe_file.write_text(json.dumps(resp, indent=2))
    _log(f"\nFull response saved → {probe_file}")


# ─────────────────────────────────────────────────────────────────────
# CHECKPOINT + CSV  (path-explicit versions — no module globals needed)
# ─────────────────────────────────────────────────────────────────────
def load_checkpoint(ckpt_path: Path) -> dict:
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            return json.load(f)
    return {"completed_bands": [], "total_collected": 0}


def save_checkpoint(state: dict, ckpt_path: Path):
    with open(ckpt_path, "w") as f:
        json.dump(state, f, indent=2)


def load_seen_ids(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    seen = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            seen.add(row.get("diamond_id", ""))
    return seen


def get_writer(csv_path: Path):
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    fh = open(csv_path, "a", newline="", encoding="utf-8")
    w  = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if write_header:
        w.writeheader()
        fh.flush()
    return fh, w


# ─────────────────────────────────────────────────────────────────────
# SCRAPE ONE CARAT BAND  (single shape)
# ─────────────────────────────────────────────────────────────────────
async def scrape_band(page, shape: str, carat_min: float, carat_max: float,
                      seen_ids, writer, fh, limit, total):
    """Page through all results for one shape in (carat_min, carat_max]."""
    band_total = 0
    display    = CONFIG["DISPLAY"]
    page_num   = 1

    while True:
        qs   = build_params(shape, carat_min, carat_max, page_num)
        resp = await call_api(page, qs)

        if resp is None:
            _log(f"  API None — skipping rest of band")
            break

        products  = resp.get("products") or []
        api_total = resp.get("total") or 0
        max_pages = (int(api_total) + display - 1) // display if api_total else 1

        if not products:
            if page_num == 1:
                _log(f"  0 items (API total={api_total}) — band empty")
            break

        new = 0
        for d in products:
            row = parse_diamond(d)
            if not row:
                continue
            did = row["diamond_id"]
            if did and did not in seen_ids:
                writer.writerow(row)
                seen_ids.add(did)
                new        += 1
                band_total += 1
                total      += 1

        fh.flush()

        status = (
            f"  p{page_num}/{max_pages}  "
            f"got={len(products):>3}  +new={new:>3}  "
            f"band={band_total}  total={total}"
        )
        if page_num == 1:
            status += f"  (API total={api_total})"
        _log(status)

        if page_num >= max_pages or len(products) < display:
            break
        if limit and total >= limit:
            break

        page_num += 1

    return band_total, total


# ─────────────────────────────────────────────────────────────────────
# SCRAPE ONE SHAPE  (all carat bands, own CSV + checkpoint)
# ─────────────────────────────────────────────────────────────────────
async def run_scrape_shape(page, shape: str, prefix: str, args):
    raw_csv = OUTPUT_DIR / f"diamonds_{prefix}_{shape.lower()}_raw.csv"
    ckpt    = OUTPUT_DIR / f"scrape_{prefix}_{shape.lower()}_checkpoint.json"

    if args.reset:
        ckpt.unlink(missing_ok=True)
        raw_csv.unlink(missing_ok=True)
        _log(f"  [{shape}] Cleared checkpoint and CSV")

    state     = load_checkpoint(ckpt)
    seen_ids  = load_seen_ids(raw_csv)
    completed = set(tuple(b) for b in state["completed_bands"])
    total     = state["total_collected"]

    _log(f"  [{shape}] {len(seen_ids):,} existing  {len(completed)} bands done  →  {raw_csv.name}")

    fh, writer = get_writer(raw_csv)
    interrupted = False

    try:
        for cmin, cmax in CONFIG["CARAT_BANDS"]:
            if (cmin, cmax) in completed:
                continue

            _log(f"\n── [{shape}] {cmin:.2f}–{cmax:.2f}ct ──")
            n, total = await scrape_band(
                page, shape, cmin, cmax,
                seen_ids, writer, fh,
                limit=args.limit, total=total,
            )
            _log(f"  [{shape}] Band done +{n}  running total={total}")

            completed.add((cmin, cmax))
            state["completed_bands"] = [list(b) for b in completed]
            state["total_collected"] = total
            save_checkpoint(state, ckpt)

            if args.limit and total >= args.limit:
                _log(f"  [{shape}] Limit {args.limit} reached")
                break

    except (KeyboardInterrupt, asyncio.CancelledError):
        _log(f"  [{shape}] Interrupted — checkpoint saved")
        interrupted = True
        raise
    finally:
        fh.close()

    if not interrupted:
        n_rows = (sum(1 for _ in open(str(raw_csv))) - 1) if raw_csv.exists() else 0
        _log(f"  [{shape}] Done. {n_rows:,} diamonds → {raw_csv.name}")


# ─────────────────────────────────────────────────────────────────────
# MAIN SCRAPE LOOP  (iterates over all configured shapes)
# ─────────────────────────────────────────────────────────────────────
async def run_scrape(page, args):
    prefix = "lab" if CONFIG["IS_LAB"] else "natural"
    _log(
        f"Config: product_class={CONFIG['PRODUCT_CLASS']!r}  "
        f"display={CONFIG['DISPLAY']}  "
        f"shapes={CONFIG['SHAPES']}"
    )

    try:
        for shape in CONFIG["SHAPES"]:
            _log(f"\n{'='*55}")
            _log(f"SHAPE: {shape.upper()}")
            await run_scrape_shape(page, shape, prefix, args)
    except (KeyboardInterrupt, asyncio.CancelledError):
        _log("Interrupted — all shape checkpoints saved, safe to resume")

    _log(f"\nAll shapes done. Next: python download_images.py")


# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(
        description="Brilliant Earth diamond scraper (Playwright + Cloudflare bypass)"
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="Make one test API call, print response, then exit",
    )
    parser.add_argument(
        "--lab", action="store_true",
        help="Scrape lab-grown diamonds (product_class=Lab Created Colorless Diamonds)",
    )
    parser.add_argument(
        "--shapes", default="round",
        help="Comma-separated shape names, or 'all'. Default: round",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Stop after N records (0 = unlimited). Use for test runs.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Clear checkpoint and CSV before starting",
    )
    parser.add_argument(
        "--no-headless", dest="headless", action="store_false", default=True,
        help="Show browser window (useful for debugging Cloudflare challenges)",
    )
    parser.add_argument(
        "--display", type=int, default=50,
        help="Items per API call (default: 50)",
    )
    args = parser.parse_args()

    # ── config ──
    CONFIG["IS_LAB"]        = args.lab
    CONFIG["PRODUCT_CLASS"] = PRODUCT_CLASS_LAB if args.lab else PRODUCT_CLASS_NATURAL
    CONFIG["DISPLAY"]       = args.display

    if args.shapes == "all":
        CONFIG["SHAPES"] = ALL_SHAPES
    else:
        CONFIG["SHAPES"] = [s.strip().title() for s in args.shapes.split(",")]

    invalid = [s for s in CONFIG["SHAPES"] if s not in ALL_SHAPES]
    if invalid:
        print(f"ERROR: Unknown shapes: {invalid}")
        print(f"Valid: {ALL_SHAPES}")
        sys.exit(1)

    # Carat bands: 0.20 → 6.00 in 0.01ct steps (580 bands)
    bands, curr = [], 0.20
    while curr < 6.00:
        nxt = round(curr + 0.01, 2)
        bands.append((curr, nxt))
        curr = nxt
    CONFIG["CARAT_BANDS"] = bands

    prefix = "lab" if args.lab else "natural"
    _log(f"Brilliant Earth scraper — {prefix} diamonds")
    _log(f"Shapes: {CONFIG['SHAPES']}")
    _log(f"Product class: {CONFIG['PRODUCT_CLASS']!r}")
    _log(f"Carat bands: {len(bands)} ({bands[0][0]:.2f}–{bands[-1][1]:.2f}ct)")

    async with async_playwright() as pw:
        browser, page = await make_browser(pw, args.headless)
        try:
            await navigate_to_search(page, args.lab)
            if args.probe:
                await run_probe(page, args.lab)
            else:
                await run_scrape(page, args)
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
