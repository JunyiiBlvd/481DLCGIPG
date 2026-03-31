"""
Image audit script for DLCGIPG diamond datasets.
JA images:  ja_scraper/output/images/{tier}/{diamond_id}.jpg
BE images:  be_scraper/output/images/{tier}/{diamond_id}.jpg
CSVs:       ja_scraper/output/diamonds_labeled.csv
            be_scraper/output/be_labeled.csv
Output:     docs/eda/images/
"""

import random
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE     = Path("/mnt/storage/projects/DLCGIPG")
JA_IMGS  = BASE / "ja_scraper/output/images"
BE_IMGS  = BASE / "be_scraper/output/images"
JA_CSV   = BASE / "ja_scraper/output/diamonds_labeled.csv"
BE_CSV   = BASE / "be_scraper/output/be_labeled.csv"
OUT_DIR  = BASE / "docs/eda/images"

TIERS      = ["budget", "mid_range", "premium", "investment_grade"]
TIER_LABELS = {"budget": "Budget", "mid_range": "Mid-Range",
               "premium": "Premium", "investment_grade": "Investment Grade"}
SUSPECT_KB = 10
SAMPLE_N   = 2_000

random.seed(42)

def sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ── Load CSVs ─────────────────────────────────────────────────────────────────
ja_df = pd.read_csv(JA_CSV)
be_df = pd.read_csv(BE_CSV)

# ── Collect all image paths ───────────────────────────────────────────────────
def collect_paths(img_root: Path) -> dict[str, list[Path]]:
    """Returns {tier: [path, ...]} for all tiers."""
    result = {}
    for tier in TIERS:
        result[tier] = sorted((img_root / tier).glob("*.jpg"))
    return result

ja_by_tier = collect_paths(JA_IMGS)
be_by_tier = collect_paths(BE_IMGS)

ja_all = [p for paths in ja_by_tier.values() for p in paths]
be_all = [p for paths in be_by_tier.values() for p in paths]

# ── 1. File count vs CSV row count ────────────────────────────────────────────
sep("1. FILE COUNT vs CSV ROW COUNT")

def count_report(name, csv_df, all_paths, by_tier):
    csv_ids   = set(csv_df["diamond_id"].astype(str))
    file_ids  = {p.stem for p in all_paths}
    n_files   = len(all_paths)
    n_csv     = len(csv_df)
    delta     = n_files - n_csv
    in_csv_no_file  = len(csv_ids - file_ids)
    in_file_no_csv  = len(file_ids - csv_ids)

    print(f"\n[{name}]")
    print(f"  CSV rows:          {n_csv:>8,}")
    print(f"  Image files:       {n_files:>8,}")
    print(f"  Delta (files-CSV): {delta:>+8,}")
    print(f"  In CSV, no file:   {in_csv_no_file:>8,}")
    print(f"  In files, no CSV:  {in_file_no_csv:>8,}")
    print(f"  Per-tier file counts:")
    for tier in TIERS:
        print(f"    {TIER_LABELS[tier]:22s}: {len(by_tier[tier]):>8,}")

count_report("James Allen (JA)", ja_df, ja_all, ja_by_tier)
count_report("Brilliant Earth (BE)", be_df, be_all, be_by_tier)

# ── 2. File size distribution ─────────────────────────────────────────────────
sep("2. FILE SIZE DISTRIBUTION")

def get_sizes_kb(paths):
    return [p.stat().st_size / 1024 for p in paths]

ja_sizes = get_sizes_kb(ja_all)
be_sizes = get_sizes_kb(be_all)

ja_suspect = [p for p, s in zip(ja_all, ja_sizes) if s < SUSPECT_KB]
be_suspect = [p for p, s in zip(be_all, be_sizes) if s < SUSPECT_KB]

print(f"\n[JA]  total={len(ja_sizes):,}  suspect(<{SUSPECT_KB}KB)={len(ja_suspect):,}")
print(f"  size: min={min(ja_sizes):.1f}KB  median={np.median(ja_sizes):.1f}KB  "
      f"p99={np.percentile(ja_sizes,99):.1f}KB  max={max(ja_sizes):.1f}KB")
if ja_suspect:
    print(f"  Suspect files:")
    for p in sorted(ja_suspect, key=lambda x: x.stat().st_size)[:10]:
        print(f"    {p.relative_to(BASE)}  ({p.stat().st_size/1024:.2f} KB)")

print(f"\n[BE]  total={len(be_sizes):,}  suspect(<{SUSPECT_KB}KB)={be_suspect and len(be_suspect) or 0:,}")
print(f"  size: min={min(be_sizes):.1f}KB  median={np.median(be_sizes):.1f}KB  "
      f"p99={np.percentile(be_sizes,99):.1f}KB  max={max(be_sizes):.1f}KB")
if be_suspect:
    print(f"  Suspect files:")
    for p in sorted(be_suspect, key=lambda x: x.stat().st_size)[:10]:
        print(f"    {p.relative_to(BASE)}  ({p.stat().st_size/1024:.2f} KB)")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, sizes, suspect, title, color in zip(
        axes,
        [ja_sizes, be_sizes],
        [ja_suspect, be_suspect],
        ["JA — File Size Distribution", "BE — File Size Distribution"],
        ["steelblue", "coral"]):
    cap = np.percentile(sizes, 99.5)
    clipped = [min(s, cap) for s in sizes]
    ax.hist(clipped, bins=80, color=color, edgecolor="none", alpha=0.85)
    ax.axvline(SUSPECT_KB, color="red", linestyle="--", linewidth=1.2,
               label=f"Suspect threshold ({SUSPECT_KB} KB)\nn={len(suspect):,}")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("File size (KB)  [capped at p99.5]")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "filesize_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[saved] filesize_distribution.png")

# ── 3. Resolution distribution ────────────────────────────────────────────────
sep("3. RESOLUTION DISTRIBUTION (sample 2,000 per dataset)")

def sample_resolutions(paths, n=SAMPLE_N):
    sample = random.sample(paths, min(n, len(paths)))
    dims, errors = [], []
    for p in sample:
        try:
            with Image.open(p) as img:
                dims.append(img.size)  # (width, height)
        except (UnidentifiedImageError, Exception):
            errors.append(p)
    return dims, errors

print("  Sampling JA resolutions...")
ja_dims, ja_res_errors = sample_resolutions(ja_all)
print("  Sampling BE resolutions...")
be_dims, be_res_errors = sample_resolutions(be_all)

def resolution_report(name, dims, errors):
    from collections import Counter
    counter = Counter(dims)
    most_common_res, most_common_n = counter.most_common(1)[0]
    deviating = [(w, h) for w, h in dims if (w, h) != most_common_res]
    print(f"\n[{name}]  sampled={len(dims):,}  read_errors={len(errors):,}")
    print(f"  Most common resolution: {most_common_res[0]}×{most_common_res[1]}  "
          f"({most_common_n:,}/{len(dims):,} = {most_common_n/len(dims)*100:.1f}%)")
    print(f"  Deviating from mode:    {len(deviating):,}  "
          f"({len(deviating)/len(dims)*100:.1f}%)")
    print(f"  Top 5 resolutions:")
    for res, cnt in counter.most_common(5):
        print(f"    {res[0]:>5}×{res[1]:<5}  {cnt:>6,}  ({cnt/len(dims)*100:.1f}%)")
    return most_common_res, len(deviating)

ja_mode_res, ja_outlier_n = resolution_report("JA", ja_dims, ja_res_errors)
be_mode_res, be_outlier_n = resolution_report("BE", be_dims, be_res_errors)

# Plot — scatter of widths vs heights, coloured by dataset
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, dims, mode_res, title, color in zip(
        axes,
        [ja_dims, be_dims],
        [ja_mode_res, be_mode_res],
        ["JA — Resolution Scatter (sampled)", "BE — Resolution Scatter (sampled)"],
        ["steelblue", "coral"]):
    ws = [d[0] for d in dims]
    hs = [d[1] for d in dims]
    ax.scatter(ws, hs, s=4, alpha=0.25, color=color)
    ax.scatter([mode_res[0]], [mode_res[1]], s=120, color="red",
               zorder=5, label=f"mode {mode_res[0]}×{mode_res[1]}")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "resolution_scatter.png", dpi=150, bbox_inches="tight")
plt.close()

# Also a width histogram
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, dims, title, color in zip(
        axes,
        [ja_dims, be_dims],
        ["JA — Width Distribution (sampled)", "BE — Width Distribution (sampled)"],
        ["steelblue", "coral"]):
    ws = [d[0] for d in dims]
    ax.hist(ws, bins=40, color=color, edgecolor="none", alpha=0.85)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout()
plt.savefig(OUT_DIR / "resolution_width_histogram.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] resolution_scatter.png, resolution_width_histogram.png")

# ── 4. Sample image grid (4 tiers × 8 images: 4 JA + 4 BE) ───────────────────
sep("4. SAMPLE IMAGE GRID (4 tiers × 8 images)")

GRID_ROWS = len(TIERS)   # 4 tiers
GRID_COLS = 8            # 4 JA + 4 BE per row
THUMB_SIZE = (200, 200)

fig, axes = plt.subplots(GRID_ROWS, GRID_COLS,
                         figsize=(GRID_COLS * 2.2, GRID_ROWS * 2.4))
fig.patch.set_facecolor("#1a1a1a")

for row_idx, tier in enumerate(TIERS):
    ja_pool = ja_by_tier[tier]
    be_pool = be_by_tier[tier]

    ja_sample = random.sample(ja_pool, min(4, len(ja_pool)))
    be_sample = random.sample(be_pool, min(4, len(be_pool)))
    all_sample = ja_sample + be_sample  # 4 JA then 4 BE

    for col_idx, img_path in enumerate(all_sample):
        ax = axes[row_idx][col_idx]
        source = "JA" if col_idx < 4 else "BE"
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                img_rgb.thumbnail(THUMB_SIZE, Image.LANCZOS)
                ax.imshow(np.array(img_rgb))
        except Exception:
            ax.set_facecolor("#333333")
            ax.text(0.5, 0.5, "ERR", ha="center", va="center",
                    color="white", fontsize=8, transform=ax.transAxes)

        ax.axis("off")
        # Column header on first row
        if row_idx == 0:
            ax.set_title(f"[{source}]", color="white", fontsize=8, pad=3)
        # Tier label on leftmost column
        if col_idx == 0:
            ax.set_ylabel(TIER_LABELS[tier], color="white", fontsize=9,
                          rotation=90, labelpad=6)

    # Vertical divider between JA and BE (draw after all cols rendered)
    # We'll use a line between col 3 and 4
    div_ax = axes[row_idx][3]
    div_ax.axvline(x=div_ax.get_xlim()[1] if hasattr(div_ax, '_xlim') else 1,
                   color="yellow", linewidth=0)  # placeholder; use fig line below

# Draw a vertical divider line at the boundary between JA and BE cols
# Use figure coordinates
for row_idx in range(GRID_ROWS):
    ax_left  = axes[row_idx][3]
    ax_right = axes[row_idx][4]
    # get figure coords of the gap between col 3 and col 4
    x_left  = ax_left.get_position().x1
    x_right = ax_right.get_position().x0
    x_mid   = (x_left + x_right) / 2
    y_bot   = axes[row_idx][0].get_position().y0
    y_top   = axes[row_idx][0].get_position().y1
    line = plt.Line2D([x_mid, x_mid], [y_bot, y_top],
                      transform=fig.transFigure, color="yellow",
                      linewidth=1.5, linestyle="--")
    fig.add_artist(line)

# Overall title and legend
fig.suptitle("Sample Images by Tier — JA (left) vs BE (right)",
             color="white", fontsize=14, y=1.01)

plt.tight_layout(pad=0.4)
plt.savefig(OUT_DIR / "sample_grid_by_tier.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("[saved] sample_grid_by_tier.png")

# ── 5. Final summary ──────────────────────────────────────────────────────────
sep("5. FINAL SUMMARY")

total_checked = len(ja_dims) + len(be_dims)
total_suspect = len(ja_suspect) + len(be_suspect)
total_res_outliers = ja_outlier_n + be_outlier_n

print(f"""
  Total images on disk:    JA={len(ja_all):>8,}   BE={len(be_all):>8,}
  CSV rows:                JA={len(ja_df):>8,}   BE={len(be_df):>8,}
  Delta (files - CSV):     JA={len(ja_all)-len(ja_df):>+8,}   BE={len(be_all)-len(be_df):>+8,}

  Suspect files (<{SUSPECT_KB}KB):    JA={len(ja_suspect):>8,}   BE={len(be_suspect):>8,}
                           total = {total_suspect:,}

  Resolution sample size:  JA={len(ja_dims):>8,}   BE={len(be_dims):>8,}
  Mode resolution:         JA={ja_mode_res[0]}×{ja_mode_res[1]}   BE={be_mode_res[0]}×{be_mode_res[1]}
  Resolution outliers:     JA={ja_outlier_n:>8,}   BE={be_outlier_n:>8,}
                           total = {total_res_outliers:,}  ({total_res_outliers/total_checked*100:.1f}% of sampled)

  Plots saved to: {OUT_DIR}
    filesize_distribution.png
    resolution_scatter.png
    resolution_width_histogram.png
    sample_grid_by_tier.png
""")
