"""
EDA script for DLCGIPG diamond datasets.
Uses: ja_scraper/output/diamonds_natural_raw.csv
      ja_scraper/output/diamonds_lab_raw.csv
      ja_scraper/output/diamonds_labeled.csv
Outputs PNGs to: docs/eda/tabular/
"""

import sys
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
BASE = "/mnt/storage/projects/DLCGIPG"
NAT_PATH     = f"{BASE}/ja_scraper/output/diamonds_natural_raw.csv"
LAB_PATH     = f"{BASE}/ja_scraper/output/diamonds_lab_raw.csv"
LABELED_PATH = f"{BASE}/ja_scraper/output/diamonds_labeled.csv"
OUT_DIR      = f"{BASE}/docs/eda/tabular"

TIER_ORDER   = ["budget", "mid", "premium", "luxury"]
CUT_ORDER    = ["Fair", "Good", "Very Good", "Ideal", "Astor Ideal"]
COLOR_ORDER  = list("DEFGHIJ")
CLARITY_ORDER = ["FL", "IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1", "I2", "I3"]

sns.set_theme(style="whitegrid", palette="muted")

def sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ── load data ────────────────────────────────────────────────────────────────
nat     = pd.read_csv(NAT_PATH)
lab     = pd.read_csv(LAB_PATH)
labeled = pd.read_csv(LABELED_PATH)

nat_lbl = labeled[labeled["is_lab_diamond"] == False].copy()
lab_lbl = labeled[labeled["is_lab_diamond"] == True].copy()

# ── 1. Schema ────────────────────────────────────────────────────────────────
sep("1. SCHEMA — ROW COUNTS, COLUMNS, DTYPES")
for name, df in [("Natural raw", nat), ("Lab raw", lab), ("Labeled (combined)", labeled)]:
    print(f"\n[{name}]  rows={len(df):,}  cols={df.shape[1]}")
    dtype_df = df.dtypes.rename("dtype").to_frame()
    dtype_df.index.name = "column"
    print(dtype_df.to_string())

# ── 2. Missing values ────────────────────────────────────────────────────────
sep("2. MISSING VALUES PER COLUMN")
for name, df in [("Natural raw", nat), ("Lab raw", lab)]:
    mv = df.isnull().sum()
    mv = mv[mv > 0]
    if mv.empty:
        print(f"\n[{name}]  No missing values.")
    else:
        mv_df = mv.to_frame("missing")
        mv_df["pct"] = (mv_df["missing"] / len(df) * 100).round(2)
        print(f"\n[{name}]")
        print(mv_df.to_string())

# ── 3. Tier class distribution ────────────────────────────────────────────────
sep("3. TIER CLASS DISTRIBUTION")

def tier_table(df, name):
    counts = df["value_tier"].value_counts().reindex(TIER_ORDER).fillna(0).astype(int)
    pcts   = (counts / counts.sum() * 100).round(2)
    tbl    = pd.DataFrame({"count": counts, "pct%": pcts})
    print(f"\n[{name}]")
    print(tbl.to_string())
    return counts

nat_tier_counts = tier_table(nat_lbl, "Natural (labeled)")
lab_tier_counts = tier_table(lab_lbl, "Lab (labeled)")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, counts, title in zip(axes, [nat_tier_counts, lab_tier_counts],
                              ["Natural Diamonds — Tier Distribution",
                               "Lab-Grown Diamonds — Tier Distribution"]):
    bars = ax.bar(counts.index, counts.values, color=sns.color_palette("muted", 4))
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Value Tier")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{val:,}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/tier_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[saved] tier_distribution.png")

# ── 4. Class imbalance ratio ─────────────────────────────────────────────────
sep("4. CLASS IMBALANCE RATIO (smallest / largest)")
for name, df in [("Natural", nat_lbl), ("Lab", lab_lbl)]:
    counts = df["value_tier"].value_counts()
    ratio  = counts.min() / counts.max()
    print(f"  {name}: {counts.idxmin()}={counts.min():,} / {counts.idxmax()}={counts.max():,}  → ratio={ratio:.4f}")

# ── 5. Natural vs lab-grown ───────────────────────────────────────────────────
sep("5. NATURAL vs. LAB-GROWN COUNTS")
for name, df in [("Labeled dataset", labeled), ("Natural raw", nat), ("Lab raw", lab)]:
    if "is_lab_diamond" in df.columns:
        vc = df["is_lab_diamond"].value_counts()
        total = vc.sum()
        tbl = pd.DataFrame({
            "count": vc,
            "pct%": (vc / total * 100).round(2)
        })
        tbl.index = tbl.index.map({True: "lab", False: "natural"})
        print(f"\n[{name}]")
        print(tbl.to_string())

# ── 6. Shape distribution ────────────────────────────────────────────────────
sep("6. SHAPE DISTRIBUTION")

def shape_counts(df):
    return df["shape"].value_counts().sort_values(ascending=False)

nat_shapes = shape_counts(nat)
lab_shapes = shape_counts(lab)
all_shapes = sorted(set(nat_shapes.index) | set(lab_shapes.index))

shape_df = pd.DataFrame({
    "natural": nat_shapes.reindex(all_shapes).fillna(0).astype(int),
    "lab":     lab_shapes.reindex(all_shapes).fillna(0).astype(int),
}, index=all_shapes).sort_values("natural", ascending=False)

print("\n[Shape counts — natural | lab]")
print(shape_df.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col, title, color in zip(
        axes, ["natural", "lab"],
        ["Natural — Shape Distribution", "Lab-Grown — Shape Distribution"],
        ["steelblue", "coral"]):
    data = shape_df[col].sort_values(ascending=False)
    ax.bar(data.index, data.values, color=color)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Shape")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/shape_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] shape_distribution.png")

# ── 7. 4C distributions ──────────────────────────────────────────────────────
sep("7. 4C ATTRIBUTE DISTRIBUTIONS")

# Carat histogram
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, df, title, color in zip(
        axes, [nat, lab],
        ["Natural — Carat Distribution", "Lab-Grown — Carat Distribution"],
        ["steelblue", "coral"]):
    ax.hist(df["carat"].dropna(), bins=60, color=color, edgecolor="white", linewidth=0.4)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Carat")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    p99 = df["carat"].quantile(0.99)
    ax.axvline(p99, color="red", linestyle="--", linewidth=1, label=f"p99={p99:.2f}ct")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/carat_histogram.png", dpi=150, bbox_inches="tight")
plt.close()
print("[saved] carat_histogram.png")

# Cut value_counts
for attr, order in [("cut", CUT_ORDER), ("color", COLOR_ORDER), ("clarity", CLARITY_ORDER)]:
    print(f"\n[{attr.upper()} — natural | lab]")
    nat_vc = nat[attr].value_counts().reindex(order).dropna().astype(int)
    lab_vc = lab[attr].value_counts().reindex(order).dropna().astype(int)
    combo  = pd.DataFrame({"natural": nat_vc, "lab": lab_vc}).fillna(0).astype(int)
    print(combo.to_string())

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, title, color in zip(
            axes, ["natural", "lab"],
            [f"Natural — {attr.capitalize()} Distribution",
             f"Lab-Grown — {attr.capitalize()} Distribution"],
            ["steelblue", "coral"]):
        present = [v for v in order if v in combo.index and combo.loc[v, col] > 0]
        vals    = combo.loc[present, col]
        ax.bar(present, vals, color=color)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(attr.capitalize())
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    fname = f"{attr}_distribution.png"
    plt.savefig(f"{OUT_DIR}/{fname}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {fname}")

# ── 8. Carat outliers ────────────────────────────────────────────────────────
sep("8. CARAT OUTLIERS (> 99th percentile)")
for name, df in [("Natural", nat), ("Lab", lab)]:
    p99 = df["carat"].quantile(0.99)
    outliers = df[df["carat"] > p99][["diamond_id", "carat", "price_usd", "shape", "cut", "color", "clarity"]]
    print(f"\n[{name}]  p99={p99:.3f}ct  →  {len(outliers):,} rows above p99")
    if not outliers.empty:
        print(outliers.sort_values("carat", ascending=False).head(20).to_string(index=False))

# ── 9. Tier boundary sanity check ────────────────────────────────────────────
sep("9. TIER BOUNDARY SANITY CHECK — min/max carat & price per tier")
for name, df in [("Natural", nat_lbl), ("Lab", lab_lbl)]:
    print(f"\n[{name}]")
    tbl = (df.groupby("value_tier", observed=False)
             .agg(
                 n=("carat", "count"),
                 carat_min=("carat", "min"),
                 carat_max=("carat", "max"),
                 price_min=("price_usd", "min"),
                 price_max=("price_usd", "max"),
                 price_mean=("price_usd", "mean"),
             )
             .reindex(TIER_ORDER)
          )
    tbl["price_mean"] = tbl["price_mean"].round(0).astype("Int64")
    print(tbl.to_string())

print("\n[EDA complete — all plots saved to docs/eda/tabular/]\n")
