# James Allen Diamond Scraper
## CSC-481 — Stage 2 Dataset Collection

Collects GIA-certified diamond listings from JamesAllen.com:
image URL + cut + color + clarity + carat + price per diamond.
Outputs a labeled CSV and downloads images into tier-named folders.

---

## Setup (protouno)

```bash
cd /mnt/storage/projects/
mkdir diamond-scraper && cd diamond-scraper
python3 -m venv venv
source venv/bin/activate
pip install requests beautifulsoup4 playwright tqdm pandas
playwright install chromium
```

Copy all `.py` files from this project into that directory.

---

## Run Order

```bash
# Step 1 — probe the site structure (run once, inspect output before full scrape)
python probe.py

# Step 2 — full scrape (runs for ~2-4 hours, resumes if interrupted)
python scrape.py

# Step 3 — download images (can run in parallel with scrape.py after ~500 rows exist)
python download_images.py

# Step 4 — compute value tiers and finalize dataset
python label_tiers.py

# Step 5 — audit what you collected
python audit.py
```

---

## Output Structure

```
output/
  diamonds_raw.csv          ← all scraped metadata (no images yet)
  diamonds_labeled.csv      ← with value_tier column added
  images/
    budget/                 ← ≤25th price percentile
    mid_range/              ← 25–75th
    premium/                ← 75–90th
    investment_grade/       ← >90th
  scrape_checkpoint.json    ← resume state
  audit_report.txt
```

---

## Expected Collection

- ~500,000 natural round-cut GIA diamonds on JA at any time
- Target: 15,000–25,000 images (covers full 4C grade space)
- At 1.5 req/sec: ~3–4 hours for metadata, ~6–8 hours for images
- Storage: ~8–15 GB for images at full resolution

---

## Notes

- Rate limited to 1.5 req/sec by default (polite, avoids bans)
- Playwright handles JS rendering on product pages
- Checkpointed — safe to Ctrl+C and resume
- Academic use only — non-commercial research project
