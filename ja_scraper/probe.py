"""
probe.py — Site structure probe for JamesAllen.com
Run this ONCE before the full scrape to verify endpoints still work
and inspect response structure.

Usage: python probe.py
"""

import json
import time
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
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

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# ─────────────────────────────────────────
# Test 1: Search list page (HTML)
# ─────────────────────────────────────────
def probe_search_html():
    print("\n[1] Probing search list page (HTML)...")
    url = (
        "https://www.jamesallen.com/loose-diamonds/round-cut/"
        "?CaratFrom=0.50&CaratTo=0.60"
        "&Color=G,H&Cut=Ideal,Very+Good"
        "&Clarity=VS1,VS2"
        "&PriceFrom=0&PriceTo=99999"
        "&ViewsOptions=List"
    )
    try:
        r = SESSION.get(url, timeout=15)
        print(f"    Status: {r.status_code}")
        print(f"    Content-Type: {r.headers.get('content-type', 'unknown')}")
        print(f"    Body length: {len(r.text)} chars")

        # Look for JSON embedded in page
        soup = BeautifulSoup(r.text, "html.parser")
        scripts = soup.find_all("script", type="application/json")
        print(f"    JSON script tags found: {len(scripts)}")
        for i, s in enumerate(scripts[:3]):
            try:
                data = json.loads(s.string)
                print(f"    Script[{i}] keys: {list(data.keys())[:8]}")
            except Exception:
                pass

        # Look for __NEXT_DATA__ (Next.js)
        next_data = soup.find("script", id="__NEXT_DATA__")
        if next_data:
            print("    Found __NEXT_DATA__ (Next.js page)")
            try:
                data = json.loads(next_data.string)
                _dig(data, "__NEXT_DATA__", depth=3)
            except Exception as e:
                print(f"    Parse error: {e}")

        # Look for window.__INITIAL_STATE__ or similar
        for script in soup.find_all("script"):
            txt = script.string or ""
            if "diamonds" in txt.lower() and ("carat" in txt.lower() or "clarity" in txt.lower()):
                print(f"    Found diamond-related script block ({len(txt)} chars)")
                print(f"    Preview: {txt[:300]}")
                break

        return r.status_code == 200

    except Exception as e:
        print(f"    ERROR: {e}")
        return False


# ─────────────────────────────────────────
# Test 2: Known internal API endpoints
# ─────────────────────────────────────────
def probe_api_endpoints():
    print("\n[2] Probing known API-style endpoints...")

    candidates = [
        # Historical endpoint from prior scrapers
        "https://www.jamesallen.com/loose-diamonds/round-cut/?CaratFrom=0.50&CaratTo=0.51&Color=G&Cut=Ideal&Clarity=VS1&ViewsOptions=List&format=json",
        # Possible REST paths
        "https://www.jamesallen.com/api/diamonds?shape=round&caratMin=0.5&caratMax=0.6",
        "https://www.jamesallen.com/api/v1/diamonds/search",
        "https://api.jamesallen.com/diamonds",
        # GraphQL
        "https://www.jamesallen.com/graphql",
    ]

    for url in candidates:
        try:
            r = SESSION.get(url, timeout=10)
            ct = r.headers.get("content-type", "")
            print(f"    {r.status_code} | {ct[:50]:50s} | {url[-60:]}")
            if "json" in ct and r.status_code == 200:
                print(f"    >>> JSON RESPONSE FOUND: {r.text[:500]}")
        except Exception as e:
            print(f"    ERR | {str(e)[:40]:40s} | {url[-60:]}")
        time.sleep(0.8)


# ─────────────────────────────────────────
# Test 3: Single product page
# ─────────────────────────────────────────
def probe_product_page():
    print("\n[3] Probing a product detail page...")
    # JA diamond IDs are numeric; these are representative
    test_ids = ["1566940", "1604221"]

    for did in test_ids:
        url = f"https://www.jamesallen.com/loose-diamonds/round-cut/{did}-sku/"
        try:
            r = SESSION.get(url, timeout=15)
            print(f"    [{did}] Status: {r.status_code}")
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                # Look for structured data
                for ld in soup.find_all("script", type="application/ld+json"):
                    try:
                        data = json.loads(ld.string)
                        print(f"    LD+JSON type: {data.get('@type', 'unknown')}")
                        if data.get("@type") == "Product":
                            print(f"    Product data: {json.dumps(data, indent=2)[:800]}")
                    except Exception:
                        pass
                # Image URLs
                imgs = soup.find_all("img", src=lambda s: s and "diamond" in s.lower())
                print(f"    Diamond images found: {len(imgs)}")
                for img in imgs[:3]:
                    print(f"    IMG: {img.get('src', '')[:100]}")
        except Exception as e:
            print(f"    ERROR: {e}")
        time.sleep(1.0)


# ─────────────────────────────────────────
# Test 4: Image CDN pattern
# ─────────────────────────────────────────
def probe_image_cdn():
    print("\n[4] Checking image CDN patterns...")
    # JA images historically at cdn1.jamesallen.com or similar
    cdn_patterns = [
        "https://cdn1.jamesallen.com/rings/RD/1566940/video.mp4",
        "https://cdn1.jamesallen.com/rings/RD/1566940/still.jpg",
        "https://cdn1.jamesallen.com/wr/rings/RD/1566940/still.jpg",
        "https://videos.jamesallen.com/rings/RD/1566940/video.mp4",
    ]
    for url in cdn_patterns:
        try:
            r = SESSION.head(url, timeout=8)
            ct = r.headers.get("content-type", "")
            cl = r.headers.get("content-length", "?")
            print(f"    {r.status_code} | {ct[:30]:30s} | {cl:>8} bytes | {url[-60:]}")
        except Exception as e:
            print(f"    ERR | {str(e)[:40]:40s} | {url[-60:]}")
        time.sleep(0.5)


# ─────────────────────────────────────────
# Helper: walk nested dict for keys
# ─────────────────────────────────────────
def _dig(obj, path, depth=0, max_depth=3):
    if depth > max_depth:
        return
    if isinstance(obj, dict):
        print(f"    {'  ' * depth}{path}: dict keys = {list(obj.keys())[:10]}")
        for k in list(obj.keys())[:5]:
            _dig(obj[k], k, depth + 1, max_depth)
    elif isinstance(obj, list):
        print(f"    {'  ' * depth}{path}: list[{len(obj)}]")
        if obj:
            _dig(obj[0], f"{path}[0]", depth + 1, max_depth)
    else:
        val = str(obj)
        print(f"    {'  ' * depth}{path}: {val[:80]}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("JA Site Structure Probe")
    print("=" * 60)

    results = {
        "search_html": probe_search_html(),
    }
    time.sleep(2)
    probe_api_endpoints()
    time.sleep(2)
    probe_product_page()
    time.sleep(2)
    probe_image_cdn()

    print("\n" + "=" * 60)
    print("PROBE COMPLETE")
    print("Review output above before running scrape.py")
    print("Key things to look for:")
    print("  1. Does search return 200? (if 403/429 → need Playwright)")
    print("  2. Is data in __NEXT_DATA__ JSON? (most common modern pattern)")
    print("  3. What is the image CDN URL pattern?")
    print("  4. Does LD+JSON on product page give structured 4C data?")
    print("=" * 60)
    print()
    print("NEXT STEP: Edit scrape.py SCRAPE_CONFIG based on probe results.")
    print("  If probe shows Cloudflare block → set USE_PLAYWRIGHT = True")
    print("  If __NEXT_DATA__ found → data extraction path is confirmed")
