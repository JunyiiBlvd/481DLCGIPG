"""
probe3.py — Capture full request body and complete item schema from JA diamond API
Run this to get everything needed to write scrape_v2.py

Outputs:
  api_request.json   — full request method, headers, body sent TO the API
  api_response.json  — full first-page response FROM the API
  item_schema.txt    — all fields in a single diamond item (flat)

Usage: python probe3.py
"""

import json
import time
from pathlib import Path


def run():
    from playwright.sync_api import sync_playwright

    captured_request  = None  # the POST to ja-product-api
    captured_response = None  # the JSON body back

    TARGET = "ja-product-api/diamond/v/2/"

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 "
                "-- CSC481-academic-research"
            )
        )
        page = context.new_page()

        # ── Intercept: capture outgoing request body ──────────────────
        def handle_request(request):
            nonlocal captured_request
            if TARGET in request.url and captured_request is None:
                try:
                    body = request.post_data
                    body_json = None
                    if body:
                        try:
                            body_json = json.loads(body)
                        except Exception:
                            body_json = body  # raw string if not JSON
                    captured_request = {
                        "url":     request.url,
                        "method":  request.method,
                        "headers": dict(request.headers),
                        "body":    body_json,
                        "body_raw": body,
                    }
                    print(f"[REQUEST CAPTURED] {request.method} {request.url}")
                except Exception as e:
                    print(f"[REQUEST CAPTURE ERROR] {e}")

        # ── Intercept: capture full response ──────────────────────────
        def handle_response(response):
            nonlocal captured_response
            if TARGET in response.url and captured_response is None:
                try:
                    data = response.json()
                    captured_response = data
                    print(f"[RESPONSE CAPTURED] {response.status} — {len(json.dumps(data))} chars")
                except Exception as e:
                    print(f"[RESPONSE CAPTURE ERROR] {e}")

        page.on("request",  handle_request)
        page.on("response", handle_response)

        print("Loading search page...")
        url = (
            "https://www.jamesallen.com/loose-diamonds/round-cut/"
            "?CaratFrom=0.50&CaratTo=0.70"
            "&Color=G,H&Cut=Ideal,Very+Good"
            "&Clarity=VS1,VS2"
            "&PriceFrom=0&PriceTo=99999"
            "&ViewsOptions=List"
        )
        try:
            page.goto(url, wait_until="networkidle", timeout=45000)
        except Exception as e:
            print(f"  (navigation warning: {e})")
        time.sleep(4)

        browser.close()

    # ── Save raw files ────────────────────────────────────────────────
    if captured_request:
        Path("api_request.json").write_text(
            json.dumps(captured_request, indent=2, default=str)
        )
        print("\n[1] api_request.json saved")
        print(f"    Method:  {captured_request['method']}")
        print(f"    URL:     {captured_request['url']}")
        print(f"    Body:    {json.dumps(captured_request.get('body'), indent=4)[:800]}")
        print("\n    ── Key request headers ──")
        for k, v in captured_request["headers"].items():
            if k.lower() in (
                "content-type", "authorization", "x-api-key",
                "x-request-id", "cookie", "origin", "referer",
                "x-client-id", "x-session", "x-token"
            ):
                print(f"    {k}: {v[:120]}")
    else:
        print("\n[1] !! api_request.json — NOT CAPTURED")
        print("    The diamond API call may have fired before page.on() registered.")
        print("    Try running again — or the API may require a specific trigger.")

    if captured_response:
        Path("api_response.json").write_text(
            json.dumps(captured_response, indent=2, default=str)
        )
        print("\n[2] api_response.json saved")

        # ── Walk item schema ──────────────────────────────────────────
        try:
            search = captured_response["data"]["searchByIDs"]
            print(f"\n    Pagination:")
            print(f"      hits:          {search.get('hits')}")
            print(f"      total:         {search.get('total')}")
            print(f"      pageNumber:    {search.get('pageNumber')}")
            print(f"      numberOfPages: {search.get('numberOfPages')}")

            items = search.get("items", [])
            # items is a list of lists — flatten one level
            flat = []
            for item in items:
                if isinstance(item, list):
                    flat.extend(item)
                elif isinstance(item, dict):
                    flat.append(item)

            print(f"      items on page: {len(flat)}")

            if flat:
                first = flat[0]
                print(f"\n[3] COMPLETE ITEM SCHEMA (first diamond):")
                schema_lines = []
                def walk(obj, prefix=""):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            walk(v, f"{prefix}.{k}" if prefix else k)
                    elif isinstance(obj, list):
                        if obj:
                            walk(obj[0], f"{prefix}[0]")
                    else:
                        line = f"    {prefix:50s}  =  {str(obj)[:80]}"
                        schema_lines.append(line)
                        print(line)

                walk(first)
                Path("item_schema.txt").write_text(
                    "\n".join(schema_lines)
                )
                print("\n    item_schema.txt saved")

                # ── Find photo-related fields ─────────────────────────
                print("\n[4] PHOTO / IMAGE FIELDS:")
                for key, val in _flatten(first).items():
                    if any(x in key.lower() for x in
                           ("photo", "image", "img", "media", "cdn", "url",
                            "gallery", "thumb", "still", "video", "ion")):
                        print(f"    {key:50s}  =  {str(val)[:100]}")

                # ── Confirm 4C fields ─────────────────────────────────
                print("\n[5] 4C + PRICE FIELDS:")
                for key, val in _flatten(first).items():
                    if any(x in key.lower() for x in
                           ("carat", "cut", "color", "clarity", "price",
                            "sku", "productid", "cert", "lab", "shape")):
                        print(f"    {key:50s}  =  {str(val)[:100]}")

        except Exception as e:
            print(f"    Schema walk error: {e}")
            print("    Raw response keys:", list(captured_response.keys()))
    else:
        print("\n[2] !! api_response.json — NOT CAPTURED")

    print("\n" + "=" * 60)
    print("DONE. Files written:")
    for f in ["api_request.json", "api_response.json", "item_schema.txt"]:
        p = Path(f)
        if p.exists():
            print(f"  {f}  ({p.stat().st_size:,} bytes)")
    print("\nNEXT: paste this output back — scrape_v2.py will be written")
    print("      based on the exact request body and item schema.")
    print("=" * 60)


def _flatten(obj, prefix=""):
    """Flatten nested dict to dot-notation keys."""
    result = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            result.update(_flatten(v, key))
    elif isinstance(obj, list):
        if obj:
            result.update(_flatten(obj[0], f"{prefix}[0]"))
    else:
        result[prefix] = obj
    return result


if __name__ == "__main__":
    run()
