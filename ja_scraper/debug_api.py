"""
debug_api.py — Fire one real API call using the exact query from api_request.json
and print the full response structure so scrape_v2.py can be fixed.

Run from the diamond-scraper directory (same folder as api_request.json):
    python debug_api.py
"""

import json
import sys
from pathlib import Path

import requests

# ── Load the exact captured request ──────────────────────────────────
req_file = Path("api_request.json")
if not req_file.exists():
    print("ERROR: api_request.json not found.")
    print("Run probe3.py first from this directory.")
    sys.exit(1)

captured = json.loads(req_file.read_text())
original_body = captured.get("body") or {}
original_query = original_body.get("query", "")

if not original_query:
    print("ERROR: No query found in api_request.json body.")
    print("Contents:", json.dumps(captured, indent=2)[:500])
    sys.exit(1)

print(f"Query loaded: {len(original_query)} chars")
print(f"Query preview: {original_query[:200]}")
print()

# ── Build a fresh variables payload (small test: 0.5–0.6ct, G, Ideal, VS1) ──
# Swap in simple variables — keep original query structure
variables = {
    "currency":     "USD",
    "isLabDiamond": False,
    "shapeID":      [1],           # round
    "color":        {"from": 4, "to": 5},   # G-H
    "cut":          {"from": 1, "to": 2},   # Ideal-Excellent
    "clarity":      {"from": 5, "to": 6},   # VS1-VS2
    "carat":        {"from": 0.50, "to": 0.60},
    "page":         {"number": 1, "size": 10},
    "sort":         "price_asc",
    "isFancy":      False,
    "isOnSale":     None,
    "addBannerPlaceholder": False,
}

# ── Also try with the original variables from the captured request ────
original_vars = original_body.get("variables", {})
print("Original captured variables:")
print(json.dumps(original_vars, indent=2)[:600])
print()

# ── Fire the request ──────────────────────────────────────────────────
API_URL = "https://www.jamesallen.com/service-api/ja-product-api/diamond/v/2/"

headers = {
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
}

print("=" * 60)
print("TEST 1: Exact original query + fresh simple variables")
print("=" * 60)
payload1 = {"query": original_query, "variables": variables}
r1 = requests.post(API_URL, json=payload1, headers=headers, timeout=20)
print(f"Status: {r1.status_code}")
print(f"Content-Type: {r1.headers.get('content-type', '?')}")
print(f"Body length: {len(r1.text)}")
try:
    d1 = r1.json()
    print(f"Top-level keys: {list(d1.keys())}")
    # Walk the response tree
    def show(obj, path="", depth=0):
        if depth > 6:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                show(v, f"{path}.{k}" if path else k, depth+1)
        elif isinstance(obj, list):
            print(f"  {path}  →  list[{len(obj)}]")
            if obj and depth < 5:
                show(obj[0], f"{path}[0]", depth+1)
        else:
            val = str(obj)
            print(f"  {path}  =  {val[:100]}")
    show(d1)
except Exception as e:
    print(f"JSON parse error: {e}")
    print(f"Raw response: {r1.text[:500]}")

print()
print("=" * 60)
print("TEST 2: Exact original query + exact original variables")
print("=" * 60)
payload2 = {"query": original_query, "variables": original_vars}
r2 = requests.post(API_URL, json=payload2, headers=headers, timeout=20)
print(f"Status: {r2.status_code}")
try:
    d2 = r2.json()
    print(f"Top-level keys: {list(d2.keys())}")
    search = (d2.get("data") or {}).get("searchByIDs") or d2.get("searchByIDs")
    if search:
        print(f"searchByIDs keys: {list(search.keys())}")
        print(f"hits: {search.get('hits')}")
        print(f"total: {search.get('total')}")
        print(f"numberOfPages: {search.get('numberOfPages')}")
        items = search.get("items", [])
        print(f"items type: {type(items).__name__}  len={len(items)}")
        if items:
            first = items[0]
            print(f"items[0] type: {type(first).__name__}")
            if isinstance(first, list):
                print(f"items[0] is a list of len={len(first)}")
                if first:
                    print(f"items[0][0] keys: {list(first[0].keys()) if isinstance(first[0], dict) else type(first[0])}")
                    # Show first diamond
                    d = first[0]
                    print(f"\nFirst diamond:")
                    print(f"  sku:       {d.get('sku')}")
                    print(f"  usdPrice:  {d.get('usdPrice')}")
                    stone = d.get("stone") or {}
                    print(f"  carat:     {stone.get('carat')}")
                    print(f"  cut:       {(stone.get('cut') or {}).get('name')}")
                    print(f"  color:     {(stone.get('color') or {}).get('name')}")
                    print(f"  clarity:   {(stone.get('clarity') or {}).get('name')}")
                    media = d.get("media") or {}
                    print(f"  supperZoom: {media.get('supperZoom', 'MISSING')}")
                    print(f"  segomaPhotoID: {media.get('segomaPhotoID', 'MISSING')}")
            elif isinstance(first, dict):
                print(f"items[0] keys: {list(first.keys())[:15]}")
                print(f"\nFirst diamond:")
                print(f"  sku:       {first.get('sku')}")
                print(f"  usdPrice:  {first.get('usdPrice')}")
                stone = first.get("stone") or {}
                print(f"  carat:     {stone.get('carat')}")
                print(f"  cut:       {(stone.get('cut') or {}).get('name')}")
                media = first.get("media") or {}
                print(f"  supperZoom: {media.get('supperZoom', 'MISSING')}")
    else:
        print("searchByIDs not found in response")
        print(f"Full response (first 1000 chars): {r2.text[:1000]}")
except Exception as e:
    print(f"JSON parse error: {e}")
    print(f"Raw: {r2.text[:500]}")

print()
print("=" * 60)
print("TEST 3: Check if items field name differs (try 'diamonds' or 'results')")
print("=" * 60)
# Try a minimal query to see what fields actually exist on searchByIDs
minimal_query = """
query($page: pager, $shapeID: [Int], $isLabDiamond: Boolean) {
  searchByIDs(page: $page, shapeID: $shapeID, isLabDiamond: $isLabDiamond) {
    hits
    total
    pageNumber
    numberOfPages
  }
}
"""
minimal_vars = {
    "page": {"number": 1, "size": 5},
    "shapeID": [1],
    "isLabDiamond": False,
}
r3 = requests.post(API_URL,
    json={"query": minimal_query, "variables": minimal_vars},
    headers=headers, timeout=20)
print(f"Status: {r3.status_code}")
try:
    d3 = r3.json()
    errors = d3.get("errors", [])
    if errors:
        print(f"GraphQL errors: {json.dumps(errors, indent=2)[:600]}")
    search3 = (d3.get("data") or {}).get("searchByIDs") or {}
    print(f"searchByIDs pagination: hits={search3.get('hits')} total={search3.get('total')}")
    print(f"Full minimal response: {json.dumps(d3, indent=2)[:800]}")
except Exception as e:
    print(f"Error: {e} | Raw: {r3.text[:300]}")

print()
print("SAVE: Writing full Test 2 response to debug_response.json")
Path("debug_response.json").write_text(r2.text)
print("Done. Paste this output back to get the fixed scrape_v2.py")
