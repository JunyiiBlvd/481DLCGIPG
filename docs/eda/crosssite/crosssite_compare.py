"""
Cross-site comparability audit: James Allen (JA) vs Brilliant Earth (BE).
Output: docs/eda/crosssite/
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path("/mnt/storage/projects/DLCGIPG")
JA_CSV  = BASE / "ja_scraper/output/diamonds_labeled.csv"
BE_CSV  = BASE / "be_scraper/output/be_labeled.csv"
JA_IMGS = BASE / "ja_scraper/output/images"
BE_IMGS = BASE / "be_scraper/output/images"
OUT     = BASE / "docs/eda/crosssite"

TIERS        = ["budget", "mid_range", "premium", "investment_grade"]
TIER_LABELS  = {"budget": "Budget", "mid_range": "Mid-Range",
                "premium": "Premium", "investment_grade": "Investment Grade"}
CUT_ORDER    = ["Fair", "Good", "Very Good", "Ideal", "Excellent", "Astor Ideal", "Super Ideal"]
COLOR_ORDER  = list("DEFGHIJKLMN")
CLARITY_ORDER = ["FL", "IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1", "I2", "I3"]

PALETTE = {"JA": "#4C9BE8", "BE": "#E87B4C"}

sns.set_theme(style="whitegrid", palette="muted")

def sep(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def fmt(n): return f"{n:,}"

# ── Load ──────────────────────────────────────────────────────────────────────
ja = pd.read_csv(JA_CSV)
be = pd.read_csv(BE_CSV)

ja["_src"] = "JA"
be["_src"] = "BE"

# ── 1. Tier distribution (normalised %) ───────────────────────────────────────
sep("1. TIER DISTRIBUTION (% of each dataset)")

def tier_pct(df):
    counts = df["value_tier"].value_counts().reindex(TIERS).fillna(0).astype(int)
    return counts, counts / counts.sum() * 100

ja_tier_n, ja_tier_pct = tier_pct(ja)
be_tier_n, be_tier_pct = tier_pct(be)

tbl = pd.DataFrame({
    "JA count": ja_tier_n,
    "JA %":     ja_tier_pct.round(2),
    "BE count": be_tier_n,
    "BE %":     be_tier_pct.round(2),
}, index=TIERS)
print(tbl.to_string())

x     = np.arange(len(TIERS))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
bars_ja = ax.bar(x - width/2, ja_tier_pct.values, width, label="JA", color=PALETTE["JA"], alpha=0.9)
bars_be = ax.bar(x + width/2, be_tier_pct.values, width, label="BE", color=PALETTE["BE"], alpha=0.9)
for bars in (bars_ja, bars_be):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(x)
ax.set_xticklabels([TIER_LABELS[t] for t in TIERS])
ax.set_ylabel("% of dataset")
ax.set_title("Tier Distribution — JA vs BE (normalised)", fontsize=13)
ax.legend()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
plt.tight_layout()
plt.savefig(OUT / "tier_distribution_pct.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[saved] tier_distribution_pct.png")

# ── 2. Shape vocabulary ───────────────────────────────────────────────────────
sep("2. SHAPE VOCABULARY COMPARISON")

ja_shapes = set(ja["shape"].dropna().str.strip().str.lower())
be_shapes = set(be["shape"].dropna().str.strip().str.lower())

ja_only = sorted(ja_shapes - be_shapes)
be_only = sorted(be_shapes - ja_shapes)
common  = sorted(ja_shapes & be_shapes)

print(f"\nJA shapes ({len(ja_shapes)}): {sorted(ja_shapes)}")
print(f"BE shapes ({len(be_shapes)}): {sorted(be_shapes)}")
print(f"\nShared ({len(common)}):      {common}")
print(f"JA only ({len(ja_only)}):     {ja_only if ja_only else '—'}")
print(f"BE only ({len(be_only)}):     {be_only if be_only else '—'}")

# Shape count comparison table
all_shapes = sorted(ja_shapes | be_shapes)
shape_tbl = pd.DataFrame({
    "JA count": [ja["shape"].str.lower().value_counts().get(s, 0) for s in all_shapes],
    "BE count": [be["shape"].str.lower().value_counts().get(s, 0) for s in all_shapes],
}, index=all_shapes)
shape_tbl["JA only"] = shape_tbl["BE count"] == 0
shape_tbl["BE only"] = shape_tbl["JA count"] == 0
print(f"\n[Shape counts]\n{shape_tbl.to_string()}")

# ── 3. Carat range per tier — box plot JA vs BE ───────────────────────────────
sep("3. CARAT RANGE PER TIER")

combined = pd.concat([
    ja[["value_tier", "carat", "_src"]],
    be[["value_tier", "carat", "_src"]],
], ignore_index=True)
combined = combined[combined["value_tier"].isin(TIERS)]

# Print summary stats
for tier in TIERS:
    subset = combined[combined["value_tier"] == tier]
    ja_sub = subset[subset["_src"] == "JA"]["carat"]
    be_sub = subset[subset["_src"] == "BE"]["carat"]
    print(f"\n[{TIER_LABELS[tier]}]")
    print(f"  JA: n={len(ja_sub):>6,}  min={ja_sub.min():.2f}  "
          f"med={ja_sub.median():.2f}  max={ja_sub.max():.2f}  p99={ja_sub.quantile(0.99):.2f}")
    print(f"  BE: n={len(be_sub):>6,}  min={be_sub.min():.2f}  "
          f"med={be_sub.median():.2f}  max={be_sub.max():.2f}  p99={be_sub.quantile(0.99):.2f}")

fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
for ax, tier in zip(axes, TIERS):
    tier_data = combined[combined["value_tier"] == tier]
    groups    = [tier_data[tier_data["_src"] == src]["carat"].dropna() for src in ["JA", "BE"]]
    bp = ax.boxplot(groups, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", linewidth=2),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3))
    for patch, color in zip(bp["boxes"], PALETTE.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_title(TIER_LABELS[tier], fontsize=11)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["JA", "BE"])
    ax.set_ylabel("Carat" if tier == TIERS[0] else "")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}"))

fig.suptitle("Carat Distribution per Tier — JA vs BE", fontsize=13)
plt.tight_layout()
plt.savefig(OUT / "carat_per_tier_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[saved] carat_per_tier_boxplot.png")

# ── 4. Grade overlap: cut / color / clarity ───────────────────────────────────
sep("4. CUT / COLOR / CLARITY GRADE OVERLAP")

for attr, order in [("cut", CUT_ORDER), ("color", COLOR_ORDER), ("clarity", CLARITY_ORDER)]:
    ja_vals = set(ja[attr].dropna().str.strip())
    be_vals = set(be[attr].dropna().str.strip())
    ja_only_v = sorted(ja_vals - be_vals)
    be_only_v = sorted(be_vals - ja_vals)
    shared_v  = sorted(ja_vals & be_vals, key=lambda v: order.index(v) if v in order else 99)

    print(f"\n[{attr.upper()}]")
    print(f"  JA unique ({len(ja_vals)}): {sorted(ja_vals)}")
    print(f"  BE unique ({len(be_vals)}): {sorted(be_vals)}")
    print(f"  Shared:   ({len(shared_v)}): {shared_v}")
    if ja_only_v:
        print(f"  JA only:  {ja_only_v}")
    if be_only_v:
        print(f"  BE only:  {be_only_v}")

    # Count table for grades present in either dataset
    all_grades = [v for v in order if v in ja_vals | be_vals]
    count_tbl = pd.DataFrame({
        "JA count": [int(ja[attr].value_counts().get(g, 0)) for g in all_grades],
        "BE count": [int(be[attr].value_counts().get(g, 0)) for g in all_grades],
    }, index=all_grades)
    count_tbl["JA only"] = count_tbl["BE count"] == 0
    count_tbl["BE only"] = count_tbl["JA count"] == 0
    print(count_tbl.to_string())

    # Grouped bar chart
    fig, ax = plt.subplots(figsize=(max(8, len(all_grades) * 1.1), 5))
    xpos = np.arange(len(all_grades))
    ax.bar(xpos - 0.2, count_tbl["JA count"], 0.38, label="JA", color=PALETTE["JA"], alpha=0.9)
    ax.bar(xpos + 0.2, count_tbl["BE count"], 0.38, label="BE", color=PALETTE["BE"], alpha=0.9)
    ax.set_xticks(xpos)
    ax.set_xticklabels(all_grades, rotation=30, ha="right")
    ax.set_title(f"{attr.capitalize()} Grade Distribution — JA vs BE", fontsize=13)
    ax.set_ylabel("Count")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    plt.tight_layout()
    plt.savefig(OUT / f"{attr}_grade_overlap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {attr}_grade_overlap.png")

# ── 5. Natural vs lab-grown per tier ─────────────────────────────────────────
sep("5. NATURAL vs LAB-GROWN BREAKDOWN PER TIER")

for name, df in [("JA", ja), ("BE", be)]:
    print(f"\n[{name}]")
    rows = []
    for tier in TIERS:
        sub = df[df["value_tier"] == tier]
        n_total = len(sub)
        n_nat = (sub["is_lab_diamond"] == False).sum()
        n_lab = (sub["is_lab_diamond"] == True).sum()
        rows.append({
            "tier":        TIER_LABELS[tier],
            "total":       n_total,
            "natural":     n_nat,
            "natural_%":   round(n_nat / n_total * 100, 1) if n_total else 0,
            "lab":         n_lab,
            "lab_%":       round(n_lab / n_total * 100, 1) if n_total else 0,
        })
    tbl = pd.DataFrame(rows).set_index("tier")
    print(tbl.to_string())

# Stacked bar: one chart per dataset
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (name, df) in zip(axes, [("JA", ja), ("BE", be)]):
    nat_pcts, lab_pcts = [], []
    for tier in TIERS:
        sub = df[df["value_tier"] == tier]
        total = len(sub)
        nat_pcts.append((sub["is_lab_diamond"] == False).sum() / total * 100 if total else 0)
        lab_pcts.append((sub["is_lab_diamond"] == True).sum()  / total * 100 if total else 0)

    x = np.arange(len(TIERS))
    bars_nat = ax.bar(x, nat_pcts, label="Natural", color="#5B9BD5", alpha=0.9)
    bars_lab = ax.bar(x, lab_pcts, bottom=nat_pcts, label="Lab-grown", color="#ED7D31", alpha=0.9)

    for bar, pct in zip(bars_nat, nat_pcts):
        if pct > 4:
            ax.text(bar.get_x() + bar.get_width()/2, pct/2,
                    f"{pct:.0f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    for bar, base, pct in zip(bars_lab, nat_pcts, lab_pcts):
        if pct > 4:
            ax.text(bar.get_x() + bar.get_width()/2, base + pct/2,
                    f"{pct:.0f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([TIER_LABELS[t] for t in TIERS], rotation=15, ha="right")
    ax.set_ylabel("% of tier")
    ax.set_ylim(0, 105)
    ax.set_title(f"{name} — Natural vs Lab per Tier", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

plt.tight_layout()
plt.savefig(OUT / "nat_vs_lab_per_tier.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[saved] nat_vs_lab_per_tier.png")

# ── 6. Summary table ─────────────────────────────────────────────────────────
sep("6. SUMMARY TABLE")

def img_count(img_root):
    return sum(1 for _ in img_root.rglob("*.jpg"))

ja_imgs = img_count(JA_IMGS)
be_imgs = img_count(BE_IMGS)

rows = []
for name, df, img_n in [("JA", ja, ja_imgs), ("BE", be, be_imgs)]:
    tiers_present = sorted(df["value_tier"].dropna().unique().tolist())
    shapes_present = sorted(df["shape"].dropna().str.lower().unique().tolist())
    rows.append({
        "dataset":       name,
        "total_rows":    len(df),
        "total_images":  img_n,
        "tiers_present": len(tiers_present),
        "shapes_present":len(shapes_present),
        "carat_min":     df["carat"].min(),
        "carat_max":     df["carat"].max(),
        "natural":       int((df["is_lab_diamond"] == False).sum()),
        "lab_grown":     int((df["is_lab_diamond"] == True).sum()),
    })

summary = pd.DataFrame(rows).set_index("dataset")
print(summary.to_string())

# ── Done ──────────────────────────────────────────────────────────────────────
print(f"\n[All outputs saved to {OUT}]\n")
