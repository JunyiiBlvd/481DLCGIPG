"""
probe_network.py — Intercept ALL XHR/fetch requests from BE diamond search page.

Loads the loose diamonds page, waits for network activity, and prints every
request that looks like it could be a diamond data API (JSON responses).

Usage:
    python probe_network.py
    python probe_network.py --url https://www.brilliantearth.com/loose-diamonds/
    python probe_network.py --wait 15       # wait longer for lazy-loaded requests

Output: output/network_requests.json — all captured requests + first 2KB of each response
"""

import asyncio
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import async_playwright

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
OUT_FILE   = OUTPUT_DIR / "network_requests.json"

# Heuristic: response content-type suggests JSON API
JSON_TYPES = ("application/json", "text/javascript", "text/json")


async def main():
    url     = sys.argv[sys.argv.index("--url") + 1] if "--url" in sys.argv else \
              "https://www.brilliantearth.com/loose-diamonds/"
    wait_s  = int(sys.argv[sys.argv.index("--wait") + 1]) if "--wait" in sys.argv else 12

    captured = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        await context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        page = await context.new_page()

        # Intercept all responses
        async def on_response(response):
            ct = response.headers.get("content-type", "")
            if not any(t in ct for t in JSON_TYPES):
                return
            req_url = response.url
            # Skip CDN assets, analytics, etc.
            skip_patterns = (
                "google", "facebook", "analytics", "fonts", "jquery",
                "cdn-cgi", "cloudflare", "segment", "hotjar", "qualtrics",
                ".js", ".css", ".woff", ".png", ".jpg", ".svg",
            )
            if any(p in req_url.lower() for p in skip_patterns):
                return

            try:
                body = await response.body()
                text = body[:3000].decode("utf-8", errors="replace")
            except Exception:
                text = "<unreadable>"

            entry = {
                "url":         req_url,
                "status":      response.status,
                "method":      response.request.method,
                "content_type": ct,
                "post_data":   response.request.post_data or "",
                "response_snippet": text,
            }
            captured.append(entry)
            parsed = urlparse(req_url)
            print(f"  [{response.status}] {response.request.method} {parsed.path}  ct={ct[:40]}")
            if text.strip().startswith("{") or text.strip().startswith("["):
                # Print first 400 chars of JSON to identify field names
                print(f"         JSON preview: {text[:400]}")

        page.on("response", on_response)

        print(f"Loading: {url}")
        try:
            await page.goto(url, wait_until="networkidle", timeout=60_000)
        except Exception as e:
            print(f"  goto: {e}")
            await asyncio.sleep(5)

        print(f"Waiting {wait_s}s for additional XHR/fetch requests ...")
        await asyncio.sleep(wait_s)

        # Also try scrolling / triggering the diamond grid to load
        print("Scrolling page to trigger lazy-load requests ...")
        for _ in range(3):
            await page.evaluate("window.scrollBy(0, window.innerHeight)")
            await asyncio.sleep(2)

        await asyncio.sleep(3)
        await browser.close()

    OUT_FILE.write_text(json.dumps(captured, indent=2, ensure_ascii=False))
    print(f"\nCaptured {len(captured)} JSON requests → {OUT_FILE}")

    # Summary: unique paths
    print("\nUnique paths seen:")
    seen = {}
    for e in captured:
        p = urlparse(e["url"]).path
        if p not in seen:
            seen[p] = e["status"]
    for path, status in sorted(seen.items()):
        print(f"  [{status}] {path}")


if __name__ == "__main__":
    asyncio.run(main())
