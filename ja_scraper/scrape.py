"""
scrape.py — Main James Allen diamond scraper
CSC-481 Stage 2 dataset collection

Collects: diamond_id, shape, carat, cut, color, clarity, price, image_url
Writes:   output/diamonds_raw.csv  (append-safe, resumes from checkpoint)

Usage:
    python scrape.py               # full run
    python scrape.py --limit 500   # test run (500 diamonds only)
    python scrape.py --reset       # clear checkpoint and restart
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import random
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ─────────────────────────────────────────
# CONFIGURATION — edit after running probe.py
# ─────────────────────────────────────────
SCRAPE_CONFIG = {
    # Set True if probe.py shows Cloudflare blocking direct requests
    "USE_PLAYWRIGHT": False,

    # Seconds between requests — DO NOT set below 0.5
    "RATE_LIMIT": 0.7,

    # Max random jitter added to rate limit (0 = no jitter)
    "JITTER": 0.4,

    # Page size for search results
    "PAGE_SIZE": 50,

    # Shapes to collect (start with round — largest inventory)
    "SHAPES": ["round-cut"],

    # Full 4C grade space to cover
    "COLORS":    ["D", "E", "F", "G", "H", "I", "J"],
    "CUTS":      ["Ideal", "Very+Good", "Good"],
    "CLARITIES": ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2"],

    # Carat sweep — iterate in 0.5ct bands to keep result pages manageable
    "CARAT_BANDS": [
        (0.30, 0.50), (0.50, 0.75), (0.75, 1.00),
        (1.00, 1.25), (1.25, 1.50), (1.50, 2.00),
        (2.00, 2.50), (2.50, 3.00), (3.00, 5.00),
    ],
}

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
RAW_CSV     = OUTPUT_DIR / "diamonds_raw.csv"
CHECKPOINT  = OUTPUT_DIR / "scrape_checkpoint.json"

CSV_FIELDS = [
    "diamond_id", "shape", "carat", "cut", "color",
    "clarity", "price_usd", "cert_lab", "image_url",
    "product_url", "scraped_at",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 "
        "-- CSC481-academic-research"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "DNT": "1",
}


# ─────────────────────────────────────────
# Checkpoint management
# ─────────────────────────────────────────
def load_checkpoint():
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {"completed_combos": [], "total_collected": 0}


def save_checkpoint(state):
    with open(CHECKPOINT, "w") as f:
        json.dump(state, f, indent=2)


def load_seen_ids():
    if not RAW_CSV.exists():
        return set()
    seen = set()
    with open(RAW_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seen.add(row["diamond_id"])
    return seen


# ─────────────────────────────────────────
# CSV writer (append mode)
# ─────────────────────────────────────────
def get_writer():
    write_header = not RAW_CSV.exists()
    f = open(RAW_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if write_header:
        writer.writeheader()
        f.flush()
    return f, writer


# ─────────────────────────────────────────
# Rate limiting
# ─────────────────────────────────────────
def wait():
    delay = SCRAPE_CONFIG["RATE_LIMIT"] + random.uniform(0, SCRAPE_CONFIG["JITTER"])
    time.sleep(delay)


# ─────────────────────────────────────────
# HTTP session
# ─────────────────────────────────────────
def make_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


# ─────────────────────────────────────────
# Build search URL
# ─────────────────────────────────────────
def search_url(shape, color, cut, clarity, carat_min, carat_max, page=1):
    base = f"https://www.jamesallen.com/loose-diamonds/{shape}/"
    params = (
        f"?CaratFrom={carat_min:.2f}&CaratTo={carat_max:.2f}"
        f"&Color={color}&Cut={cut}&Clarity={clarity}"
        f"&PriceFrom=0&PriceTo=9999999"
        f"&ViewsOptions=List"
        f"&page={page}"
    )
    return base + params


# ─────────────────────────────────────────
# Parse search results page → list of diamond dicts
# ─────────────────────────────────────────
def parse_search_page(html, shape):
    """
    Try multiple extraction strategies in priority order:
    1. __NEXT_DATA__ JSON (Next.js — most modern, most reliable)
    2. application/ld+json Product schema
    3. HTML parsing of list items
    """
    diamonds = []
    soup = BeautifulSoup(html, "html.parser")

    # Strategy 1: Next.js __NEXT_DATA__
    next_tag = soup.find("script", id="__NEXT_DATA__")
    if next_tag:
        try:
            data = json.loads(next_tag.string)
            # Walk the props tree looking for diamond arrays
            diamonds = _extract_from_next_data(data, shape)
            if diamonds:
                return diamonds
        except Exception as e:
            _log(f"  __NEXT_DATA__ parse failed: {e}")

    # Strategy 2: LD+JSON product listings
    for ld_tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(ld_tag.string)
            if isinstance(data, list):
                for item in data:
                    d = _ld_to_diamond(item, shape)
                    if d:
                        diamonds.append(d)
            elif isinstance(data, dict):
                d = _ld_to_diamond(data, shape)
                if d:
                    diamonds.append(d)
        except Exception:
            pass
    if diamonds:
        return diamonds

    # Strategy 3: HTML item cards (fallback)
    diamonds = _parse_html_list(soup, shape)
    return diamonds


def _extract_from_next_data(data, shape):
    """Walk Next.js data tree for diamond arrays."""
    diamonds = []

    def walk(obj, depth=0):
        if depth > 10:
            return
        if isinstance(obj, list):
            for item in obj:
                walk(item, depth + 1)
        elif isinstance(obj, dict):
            # Look for arrays with diamond-like keys
            for key in ["diamonds", "products", "items", "results", "data"]:
                if key in obj and isinstance(obj[key], list):
                    for item in obj[key]:
                        d = _dict_to_diamond(item, shape)
                        if d:
                            diamonds.append(d)
            # Also recurse
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    walk(v, depth + 1)

    walk(data)
    return diamonds


def _dict_to_diamond(item, shape):
    """Attempt to build a diamond record from an arbitrary dict."""
    if not isinstance(item, dict):
        return None

    # Normalize key names (JA may use camelCase or snake_case)
    lower = {k.lower().replace("_", "").replace("-", ""): v for k, v in item.items()}

    did = (
        item.get("id") or item.get("diamond_id") or item.get("diamondId") or
        item.get("sku") or lower.get("id") or lower.get("diamondid")
    )
    carat = (
        item.get("carat") or item.get("caratWeight") or item.get("carat_weight") or
        lower.get("carat") or lower.get("caratweight")
    )
    cut = (
        item.get("cut") or item.get("cutGrade") or item.get("cut_grade") or
        lower.get("cut") or lower.get("cutgrade")
    )
    color = (
        item.get("color") or item.get("colorGrade") or
        lower.get("color") or lower.get("colorgrade")
    )
    clarity = (
        item.get("clarity") or item.get("clarityGrade") or
        lower.get("clarity") or lower.get("claritygrade")
    )
    price = (
        item.get("price") or item.get("price_usd") or item.get("retailPrice") or
        lower.get("price") or lower.get("retailprice")
    )
    image = (
        item.get("image") or item.get("imageUrl") or item.get("image_url") or
        item.get("thumbnail") or lower.get("imageurl") or lower.get("image")
    )
    # image may be a dict with a 'url' key
    if isinstance(image, dict):
        image = image.get("url") or image.get("src") or image.get("href")

    cert = item.get("cert") or item.get("lab") or item.get("certLab") or lower.get("cert")

    if not (did and carat and clarity):
        return None

    return {
        "diamond_id":  str(did),
        "shape":       shape.replace("-cut", "").replace("-", "_"),
        "carat":       float(carat) if carat else None,
        "cut":         str(cut) if cut else "",
        "color":       str(color) if color else "",
        "clarity":     str(clarity) if clarity else "",
        "price_usd":   float(str(price).replace(",", "").replace("$", "")) if price else None,
        "cert_lab":    str(cert) if cert else "",
        "image_url":   str(image) if image else "",
        "product_url": f"https://www.jamesallen.com/loose-diamonds/{shape}/{did}-sku/",
        "scraped_at":  datetime.utcnow().isoformat(),
    }


def _ld_to_diamond(data, shape):
    if data.get("@type") not in ("Product", "Offer"):
        return None
    return _dict_to_diamond(data, shape)


def _parse_html_list(soup, shape):
    """HTML fallback: find product cards by CSS patterns JA has used historically."""
    diamonds = []

    # JA has used data-id, data-diamond-id, or similar attributes
    for card in soup.find_all(attrs={"data-id": True}):
        did = card.get("data-id") or card.get("data-diamond-id")
        if not did:
            continue

        text = card.get_text(" ", strip=True)

        def find_val(patterns):
            for pat in patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    return m.group(1)
            return None

        carat   = find_val([r"([\d.]+)\s*ct", r"([\d.]+)\s*carat"])
        cut     = find_val([r"\b(Ideal|Excellent|Very\s*Good|Good|Fair)\b"])
        color   = find_val([r"\bColor[:\s]+([D-Z])\b", r"\b([D-Z])\s+Color\b"])
        clarity = find_val([r"\b(IF|VVS1|VVS2|VS1|VS2|SI1|SI2|I1|I2|I3)\b"])
        price   = find_val([r"\$([\d,]+)"])

        img_tag = card.find("img")
        image = img_tag.get("src") or img_tag.get("data-src") if img_tag else None

        if not (did and carat and clarity):
            continue

        diamonds.append({
            "diamond_id":  str(did),
            "shape":       shape.replace("-cut", ""),
            "carat":       float(carat) if carat else None,
            "cut":         cut or "",
            "color":       color or "",
            "clarity":     clarity or "",
            "price_usd":   float(price.replace(",", "")) if price else None,
            "cert_lab":    "",
            "image_url":   str(image) if image else "",
            "product_url": f"https://www.jamesallen.com/loose-diamonds/{shape}/{did}-sku/",
            "scraped_at":  datetime.utcnow().isoformat(),
        })

    return diamonds


def get_total_results(html):
    """Try to extract total result count from page."""
    soup = BeautifulSoup(html, "html.parser")

    # Next.js data
    next_tag = soup.find("script", id="__NEXT_DATA__")
    if next_tag:
        try:
            data = json.loads(next_tag.string)
            txt = json.dumps(data)
            for key in ["totalCount", "total_count", "totalResults", "count"]:
                m = re.search(rf'"{key}"\s*:\s*(\d+)', txt)
                if m:
                    return int(m.group(1))
        except Exception:
            pass

    # HTML text
    text = soup.get_text()
    m = re.search(r"([\d,]+)\s+diamonds?\s+found", text, re.IGNORECASE)
    if m:
        return int(m.group(1).replace(",", ""))

    return None


# ─────────────────────────────────────────
# Playwright mode (if requests gets blocked)
# ─────────────────────────────────────────
_pw_browser = None
_pw_context = None


def get_playwright_page():
    global _pw_browser, _pw_context
    if _pw_browser is None:
        from playwright.sync_api import sync_playwright
        pw = sync_playwright().start()
        _pw_browser = pw.chromium.launch(headless=True)
        _pw_context = _pw_browser.new_context(
            user_agent=HEADERS["User-Agent"],
            extra_http_headers={"DNT": "1"},
        )
    return _pw_context.new_page()


def fetch_with_playwright(url):
    page = get_playwright_page()
    try:
        page.goto(url, wait_until="networkidle", timeout=30000)
        time.sleep(1.5)
        return page.content()
    finally:
        page.close()


# ─────────────────────────────────────────
# Fetch one URL (HTML or Playwright)
# ─────────────────────────────────────────
_session = None


def fetch(url, retries=3):
    global _session
    if _session is None:
        _session = make_session()

    if SCRAPE_CONFIG["USE_PLAYWRIGHT"]:
        for attempt in range(retries):
            try:
                return fetch_with_playwright(url)
            except Exception as e:
                _log(f"  Playwright attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)
        return None

    for attempt in range(retries):
        try:
            r = _session.get(url, timeout=20)
            if r.status_code == 200:
                return r.text
            if r.status_code == 429:
                wait_time = 30 + 30 * attempt
                _log(f"  429 rate-limited — waiting {wait_time}s")
                time.sleep(wait_time)
                continue
            if r.status_code == 403:
                _log(f"  403 Forbidden — switching to Playwright mode")
                SCRAPE_CONFIG["USE_PLAYWRIGHT"] = True
                return fetch(url, retries)
            _log(f"  HTTP {r.status_code} for {url}")
            return None
        except requests.exceptions.RequestException as e:
            _log(f"  Request error attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return None


# ─────────────────────────────────────────
# Core scrape loop for one combo
# ─────────────────────────────────────────
def scrape_combo(shape, color, cut, clarity, carat_min, carat_max, seen_ids, writer, file_handle):
    collected = 0
    page = 1

    while True:
        url = search_url(shape, color, cut, clarity, carat_min, carat_max, page)
        wait()
        html = fetch(url)

        if not html:
            _log(f"  No response, skipping combo")
            break

        diamonds = parse_search_page(html, shape)

        if not diamonds:
            if page == 1:
                total = get_total_results(html)
                _log(f"  No diamonds parsed (total={total}). May need probe re-run.")
            break

        new_count = 0
        for d in diamonds:
            did = d.get("diamond_id", "")
            if did and did not in seen_ids:
                writer.writerow(d)
                seen_ids.add(did)
                new_count += 1
                collected += 1

        file_handle.flush()

        total = get_total_results(html)
        _log(f"  page {page} → {len(diamonds)} parsed, {new_count} new"
             + (f" / {total} total" if total else ""))

        # Stop if we got fewer results than expected (last page)
        if len(diamonds) < SCRAPE_CONFIG["PAGE_SIZE"] // 2:
            break

        page += 1

        # Safety: don't go past 100 pages per combo
        if page > 100:
            break

    return collected


# ─────────────────────────────────────────
# Logging
# ─────────────────────────────────────────
def _log(msg):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Stop after N diamonds (0=no limit)")
    parser.add_argument("--reset", action="store_true", help="Clear checkpoint and restart")
    parser.add_argument("--playwright", action="store_true", help="Force Playwright mode")
    args = parser.parse_args()

    if args.reset:
        CHECKPOINT.unlink(missing_ok=True)
        RAW_CSV.unlink(missing_ok=True)
        _log("Checkpoint and CSV cleared.")

    if args.playwright:
        SCRAPE_CONFIG["USE_PLAYWRIGHT"] = True
        _log("Playwright mode enabled.")

    state = load_checkpoint()
    seen_ids = load_seen_ids()
    completed = set(tuple(c) for c in state["completed_combos"])
    total_collected = state["total_collected"]

    _log(f"Resuming: {len(seen_ids)} existing diamonds, {len(completed)} completed combos")

    cfg = SCRAPE_CONFIG
    f, writer = get_writer()

    try:
        for shape in cfg["SHAPES"]:
            for cmin, cmax in cfg["CARAT_BANDS"]:
                for color in cfg["COLORS"]:
                    for cut in cfg["CUTS"]:
                        for clarity in cfg["CLARITIES"]:
                            combo = (shape, color, cut, clarity, cmin, cmax)

                            if combo in completed:
                                continue

                            _log(f"{shape} | {cmin:.2f}-{cmax:.2f}ct | {color} | {cut} | {clarity}")

                            n = scrape_combo(shape, color, cut, clarity, cmin, cmax, seen_ids, writer, f)
                            total_collected += n

                            completed.add(combo)
                            state["completed_combos"] = [list(c) for c in completed]
                            state["total_collected"] = total_collected
                            save_checkpoint(state)

                            _log(f"  → +{n} | total: {total_collected}")

                            if args.limit and total_collected >= args.limit:
                                _log(f"Limit of {args.limit} reached. Stopping.")
                                return

    except KeyboardInterrupt:
        _log("Interrupted — checkpoint saved, safe to resume.")
    finally:
        f.close()
        _log(f"Final count: {total_collected} diamonds in {RAW_CSV}")
        _log("Next step: python download_images.py")


if __name__ == "__main__":
    main()
