"""
probe2.py — Network interception probe for JamesAllen.com
Run this to find the real internal API endpoint and image CDN pattern.

Requires Playwright: pip install playwright && playwright install chromium

Usage: python probe2.py
"""

import json
import time
from urllib.parse import urlparse, parse_qs

# ─────────────────────────────────────────────────────────────────────
# Storage for intercepted requests
# ─────────────────────────────────────────────────────────────────────
captured = {
    "api_calls":    [],   # XHR/fetch calls that returned JSON
    "image_urls":   [],   # image requests
    "all_requests": [],   # everything (for inspection)
}


def record_request(route, request):
    """Intercept and log every outgoing request."""
    url = request.url
    rtype = request.resource_type

    entry = {"type": rtype, "url": url}
    captured["all_requests"].append(entry)

    if rtype == "image" and "diamond" in url.lower():
        captured["image_urls"].append(url)

    route.continue_()


def record_response(response):
    """Intercept responses — look for JSON containing diamond data."""
    url = response.url
    ct = response.headers.get("content-type", "")

    if "json" not in ct:
        return

    try:
        body = response.json()
        body_str = json.dumps(body)

        # Look for diamond-like payloads
        diamond_signals = ["carat", "clarity", "diamond", "cut", "color", "price"]
        hits = sum(1 for sig in diamond_signals if sig in body_str.lower())

        if hits >= 3:
            captured["api_calls"].append({
                "url":    url,
                "status": response.status,
                "hits":   hits,
                "keys":   list(body.keys())[:15] if isinstance(body, dict) else f"list[{len(body)}]",
                "preview": body_str[:600],
            })
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────
# Main probe
# ─────────────────────────────────────────────────────────────────────
def run_probe():
    from playwright.sync_api import sync_playwright

    search_url = (
        "https://www.jamesallen.com/loose-diamonds/round-cut/"
        "?CaratFrom=0.50&CaratTo=0.70"
        "&Color=G,H&Cut=Ideal,Very+Good"
        "&Clarity=VS1,VS2"
        "&PriceFrom=0&PriceTo=99999"
        "&ViewsOptions=List"
    )

    print("Launching Playwright (headless Chromium)...")
    print(f"Target: {search_url}\n")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 "
                "-- CSC481-academic-research"
            ),
            extra_http_headers={
                "DNT": "1",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        page = context.new_page()

        # Intercept ALL requests
        page.route("**/*", record_request)

        # Intercept ALL responses (looking for JSON API calls)
        page.on("response", record_response)

        print("[1] Loading search page — waiting for network idle...")
        try:
            page.goto(search_url, wait_until="networkidle", timeout=45000)
        except Exception as e:
            print(f"  Warning: {e} (continuing anyway)")

        # Give JS extra time to fire lazy requests
        print("[2] Waiting 4 more seconds for lazy API calls...")
        time.sleep(4)

        # Scroll down to trigger any lazy-load
        print("[3] Scrolling to trigger lazy loads...")
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(2)
        page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
        time.sleep(2)

        # Grab page source for manual inspection
        html = page.content()

        # ── Also probe one product page ──
        print("\n[4] Loading a product detail page...")
        product_url = "https://www.jamesallen.com/loose-diamonds/round-cut/1566940-sku/"
        try:
            page.goto(product_url, wait_until="networkidle", timeout=30000)
            time.sleep(3)
        except Exception as e:
            print(f"  Warning: {e}")

        # ── Try fetching an image with Referer set ──
        print("\n[5] Testing CDN image with Referer header...")
        img_result = context.request.get(
            "https://cdn1.jamesallen.com/rings/RD/1566940/still.jpg",
            headers={"Referer": "https://www.jamesallen.com/"},
        )
        print(f"  CDN still.jpg with Referer: {img_result.status} "
              f"| {img_result.headers.get('content-type', '?')} "
              f"| {len(img_result.body())} bytes")

        # Try hi-res variant
        img_result2 = context.request.get(
            "https://cdn1.jamesallen.com/rings/RD/1566940/still-hi.jpg",
            headers={"Referer": "https://www.jamesallen.com/"},
        )
        print(f"  CDN still-hi.jpg with Referer: {img_result2.status} "
              f"| {img_result2.headers.get('content-type', '?')} "
              f"| {len(img_result2.body())} bytes")

        browser.close()

    # ── Results ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("NETWORK INTERCEPT RESULTS")
    print("=" * 60)

    # API calls
    print(f"\nJSON API CALLS WITH DIAMOND DATA ({len(captured['api_calls'])} found):")
    if captured["api_calls"]:
        for i, call in enumerate(captured["api_calls"]):
            print(f"\n  [{i+1}] {call['url']}")
            print(f"       Status: {call['status']}  |  Diamond signals: {call['hits']}/6")
            print(f"       Keys:   {call['keys']}")
            print(f"       Preview: {call['preview'][:300]}")
    else:
        print("  NONE FOUND — diamond data may be in page HTML or a non-JSON format")
        print("  Check html_snippet.txt for clues")

    # Image URLs
    print(f"\nDIAMOND IMAGE REQUESTS INTERCEPTED ({len(captured['image_urls'])} found):")
    for url in captured["image_urls"][:10]:
        print(f"  {url}")

    # All unique domains called
    domains = set()
    for r in captured["all_requests"]:
        try:
            domains.add(urlparse(r["url"]).netloc)
        except Exception:
            pass
    print(f"\nALL DOMAINS CALLED ({len(domains)} unique):")
    for d in sorted(domains):
        count = sum(1 for r in captured["all_requests"]
                    if urlparse(r["url"]).netloc == d)
        print(f"  {count:>4}x  {d}")

    # Resource type breakdown
    from collections import Counter
    types = Counter(r["type"] for r in captured["all_requests"])
    print(f"\nREQUEST TYPES:")
    for t, n in types.most_common():
        print(f"  {n:>4}x  {t}")

    # XHR/fetch URLs
    xhr_urls = [r["url"] for r in captured["all_requests"]
                if r["type"] in ("xhr", "fetch")]
    print(f"\nALL XHR/FETCH CALLS ({len(xhr_urls)}):")
    for url in sorted(set(xhr_urls)):
        print(f"  {url}")

    # Save HTML snippet for manual inspection
    with open("html_snippet.txt", "w") as f:
        f.write(f"Page HTML length: {len(html)}\n\n")
        f.write(html[:8000])
    print(f"\nHTML saved to html_snippet.txt (first 8000 chars)")

    # Save full capture
    with open("probe2_results.json", "w") as f:
        json.dump({
            "api_calls":    captured["api_calls"],
            "image_urls":   captured["image_urls"][:50],
            "xhr_urls":     list(set(xhr_urls)),
            "domains":      list(domains),
        }, f, indent=2)
    print("Full results saved to probe2_results.json")

    print("\n" + "=" * 60)
    print("WHAT TO LOOK FOR IN RESULTS:")
    print("  API CALLS section → real endpoint URL → update scrape.py")
    print("  IMAGE REQUESTS → CDN URL pattern → update download_images.py")
    print("  XHR/FETCH section → find the diamond data endpoint")
    print("  html_snippet.txt → search for 'carat' or 'clarity' manually")
    print("=" * 60)


if __name__ == "__main__":
    run_probe()
