#!/usr/bin/env python3
"""
aggregate_results.py — DLCGIPG Stage 2 Results Aggregation
===========================================================
Reads all final_metrics.json files produced by train.py and evaluate.py,
assembles the complete 24-model experiment table, and writes structured
outputs for paper reporting and downstream analysis.

Usage
-----
    python3 scripts/aggregate_results.py

    # Custom results directory:
    python3 scripts/aggregate_results.py --results-dir /path/to/results/training

    # Skip CSV/JSON output (print only):
    python3 scripts/aggregate_results.py --no-save

Outputs (written to results/aggregated/)
-----------------------------------------
    summary_table.csv       — flat table of all 24 models, machine-readable
    within_site.csv         — within-site subset for paper Table 1
    cross_domain.csv        — cross-domain subset for paper Table 2
    full_results.json       — complete structured dump including per-class F1
    aggregate_results.log   — stdout mirror for reproducibility audit trail

Notes
-----
- Directories matching *attempt*_bad are excluded from the main tables
  but printed separately under ARCHIVED for audit purposes.
- Per-class F1 keys are normalized: both "per_class_f1" and "class_f1"
  are accepted for backwards compatibility with earlier evaluate.py versions.
- Metric keys accepted: test_accuracy / accuracy, test_macro_f1 / macro_f1,
  epochs_trained / best_epoch.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "training"
OUTPUT_DIR = PROJECT_ROOT / "results" / "aggregated"

ARCH_MAP = {
    "resnet50":       "ResNet50",
    "efficientnetv2": "EfficientNetV2-S",
    "vit":            "ViT-B/16",
}

SUBSET_MAP = {
    "ja_natural": ("JA", "natural"),
    "ja_lab":     ("JA", "lab"),
    "be_natural": ("BE", "natural"),
    "be_lab":     ("BE", "lab"),
}

# For sorting output rows
ARCH_ORDER   = {"ResNet50": 0, "EfficientNetV2-S": 1, "ViT-B/16": 2}
SUBSET_ORDER = {"JA natural": 0, "JA lab": 1, "BE natural": 2, "BE lab": 3}

RF_CEILING = {
    "ja_natural": {"accuracy": 0.8658, "macro_f1": 0.8567},
    "ja_lab":     {"accuracy": 0.9411, "macro_f1": 0.9217},
    "be_natural": {"accuracy": 0.8371, "macro_f1": 0.8226},
    "be_lab":     {"accuracy": 0.8961, "macro_f1": 0.8742},
}

CLASSES = ["budget", "mid_range", "premium", "investment_grade"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(d: dict, *keys, default=None):
    """Return first key found in dict, or default."""
    for k in keys:
        if k in d:
            return d[k]
    return default


def fmt(val, decimals=4):
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val) if val is not None else "?"


def load_metrics(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_folder(folder_name: str):
    """
    Parse a result folder name into (arch_raw, subset_raw, mode).
    Expected format: {arch}__{subset}__{mode}[__suffix]
    Returns None if the folder doesn't match the expected pattern.
    """
    parts = folder_name.split("__")
    if len(parts) < 3:
        return None
    arch_raw, subset_raw, mode = parts[0], parts[1], parts[2]
    if arch_raw not in ARCH_MAP or subset_raw not in SUBSET_MAP:
        return None
    if mode not in ("within", "cross"):
        return None
    return arch_raw, subset_raw, mode


def extract_row(folder_name: str, metrics: dict) -> dict:
    """Convert folder name + metrics dict into a unified row dict."""
    arch_raw, subset_raw, mode = parse_folder(folder_name)

    arch = ARCH_MAP[arch_raw]
    site, origin = SUBSET_MAP[subset_raw]
    train_subset = f"{site} {origin}"

    acc    = _get(metrics, "test_accuracy", "accuracy")
    mf1    = _get(metrics, "test_macro_f1", "macro_f1")
    epochs = _get(metrics, "epochs_trained", "best_epoch")
    per_class = _get(metrics, "per_class_f1", "class_f1", default={})

    direction = None
    test_site = None
    if mode == "cross":
        direction = "JA→BE" if site == "JA" else "BE→JA"
        test_site = "BE" if site == "JA" else "JA"

    return {
        "arch":          arch,
        "arch_raw":      arch_raw,
        "train_site":    site,
        "train_origin":  origin,
        "train_subset":  train_subset,
        "subset_raw":    subset_raw,
        "mode":          mode,
        "direction":     direction,
        "test_site":     test_site,
        "accuracy":      acc,
        "macro_f1":      mf1,
        "epochs":        epochs,
        "per_class_f1":  per_class,
        "folder":        folder_name,
    }


def sort_key_within(row):
    return (ARCH_ORDER[row["arch"]], SUBSET_ORDER[row["train_subset"]])


def sort_key_cross(row):
    dir_order = 0 if row["direction"] == "JA→BE" else 1
    return (ARCH_ORDER[row["arch"]], dir_order, SUBSET_ORDER[row["train_subset"]])


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_within(rows, log):
    log("\n" + "=" * 80)
    log("WITHIN-SITE RESULTS  (12 models)")
    log("=" * 80)
    log(f"{'Arch':<20} {'Subset':<14} {'Acc':>8} {'MacroF1':>9} {'Ep':>5}")
    log("-" * 62)
    for r in rows:
        log(f"{r['arch']:<20} {r['train_subset']:<14} "
            f"{fmt(r['accuracy']):>8} {fmt(r['macro_f1']):>9} {fmt(r['epochs'], 0):>5}")
        pc = r["per_class_f1"]
        if pc:
            for cls in CLASSES:
                v = pc.get(cls)
                log(f"    {cls:<22} F1={fmt(v)}")

    log("\n  RF tabular ceiling (reference):")
    log(f"  {'Subset':<14} {'Acc':>8} {'MacroF1':>9}")
    log(f"  {'-'*34}")
    for subset_raw, vals in RF_CEILING.items():
        site, origin = SUBSET_MAP[subset_raw]
        log(f"  {site+' '+origin:<14} {vals['accuracy']:>8.4f} {vals['macro_f1']:>9.4f}")


def print_cross(rows, log):
    log("\n" + "=" * 80)
    log("CROSS-DOMAIN RESULTS  (12 models)")
    log("=" * 80)
    log(f"{'Arch':<20} {'Train':<14} {'Test':<5} {'Dir':<8} {'Acc':>8} {'MacroF1':>9} {'Ep':>5}")
    log("-" * 74)
    for r in rows:
        log(f"{r['arch']:<20} {r['train_subset']:<14} {r['test_site']:<5} "
            f"{r['direction']:<8} {fmt(r['accuracy']):>8} {fmt(r['macro_f1']):>9} "
            f"{fmt(r['epochs'], 0):>5}")
        pc = r["per_class_f1"]
        if pc:
            for cls in CLASSES:
                v = pc.get(cls)
                log(f"    {cls:<22} F1={fmt(v)}")


def print_archived(bad_rows, log):
    if not bad_rows:
        return
    log("\n" + "=" * 80)
    log("ARCHIVED (excluded from main tables — methodology artifacts)")
    log("=" * 80)
    for folder, metrics in bad_rows:
        log(f"\n  Folder : {folder}")
        log(f"  Metrics: {json.dumps(metrics, indent=4)}")


def print_summary_stats(within_rows, cross_rows, log):
    log("\n" + "=" * 80)
    log("SUMMARY STATISTICS")
    log("=" * 80)

    def stats(values, label):
        values = [v for v in values if isinstance(v, float)]
        if not values:
            log(f"  {label}: no data")
            return
        log(f"  {label}: min={min(values):.4f}  max={max(values):.4f}  "
            f"mean={sum(values)/len(values):.4f}")

    wf1 = [r["macro_f1"] for r in within_rows]
    stats(wf1, "Within-site macro F1 (all architectures)")

    for arch in ["ResNet50", "EfficientNetV2-S", "ViT-B/16"]:
        vals = [r["macro_f1"] for r in within_rows if r["arch"] == arch]
        stats(vals, f"  Within-site macro F1 — {arch}")

    cf1_ja_be = [r["macro_f1"] for r in cross_rows if r["direction"] == "JA→BE"]
    cf1_be_ja = [r["macro_f1"] for r in cross_rows if r["direction"] == "BE→JA"]
    stats(cf1_ja_be, "Cross-domain macro F1 — JA→BE")
    stats(cf1_be_ja, "Cross-domain macro F1 — BE→JA")

    # mid_range F1 across all within-site runs
    mid_f1s = []
    for r in within_rows:
        v = r["per_class_f1"].get("mid_range")
        if isinstance(v, float):
            mid_f1s.append(v)
    if mid_f1s:
        log(f"\n  mid_range class F1 (within-site, all archs): "
            f"min={min(mid_f1s):.4f}  max={max(mid_f1s):.4f}  "
            f"mean={sum(mid_f1s)/len(mid_f1s):.4f}")
        log(f"  (pre-registered prediction: mid_range is hardest class)")

    # Cross-domain collapse indicator: models where >50% predictions are one class
    log("\n  Cross-domain per-class collapse check (premium dominance):")
    for r in cross_rows:
        pc = r["per_class_f1"]
        prem = pc.get("premium")
        others = [pc.get(c) for c in CLASSES if c != "premium"]
        if isinstance(prem, float) and all(isinstance(v, float) for v in others):
            if prem > 0.5 and all(v < 0.2 for v in others if v is not None):
                log(f"    ⚠  {r['arch']} {r['train_subset']}→{r['test_site']}: "
                    f"premium F1={prem:.4f}, others suppressed — collapse detected")


# ---------------------------------------------------------------------------
# CSV / JSON output
# ---------------------------------------------------------------------------

def write_csv(rows, path: Path, mode: str):
    import csv
    fieldnames = [
        "arch", "train_site", "train_origin", "train_subset", "mode",
        "direction", "test_site", "accuracy", "macro_f1", "epochs",
        "budget_f1", "mid_range_f1", "premium_f1", "investment_grade_f1",
        "folder",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            pc = r.get("per_class_f1", {})
            row = {**r, **{f"{cls}_f1": pc.get(cls) for cls in CLASSES}}
            w.writerow(row)


def write_json(within_rows, cross_rows, bad_rows, path: Path):
    payload = {
        "generated_at": datetime.now().isoformat(),
        "within_site":  within_rows,
        "cross_domain": cross_rows,
        "archived":     [{"folder": f, "metrics": m} for f, m in bad_rows],
        "rf_ceiling":   RF_CEILING,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Aggregate DLCGIPG Stage 2 results")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                        help="Directory containing per-run result folders")
    parser.add_argument("--no-save", action="store_true",
                        help="Print only; do not write CSV/JSON/log files")
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Set up logging to file + stdout
    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    log(f"DLCGIPG Stage 2 — Results Aggregation")
    log(f"Run at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Source : {results_dir}")

    # Collect all final_metrics.json files
    within_rows = []
    cross_rows  = []
    bad_rows    = []
    skipped     = []

    for p in sorted(results_dir.glob("*/final_metrics.json")):
        folder = p.parent.name

        # Archive bad attempts separately
        if "bad" in folder or "attempt" in folder:
            bad_rows.append((folder, load_metrics(p)))
            continue

        parsed = parse_folder(folder)
        if parsed is None:
            skipped.append(folder)
            continue

        metrics = load_metrics(p)
        row = extract_row(folder, metrics)

        if row["mode"] == "within":
            within_rows.append(row)
        else:
            cross_rows.append(row)

    if skipped:
        log(f"\nSkipped (unrecognized folder names): {skipped}")

    log(f"\nLoaded: {len(within_rows)} within-site | "
        f"{len(cross_rows)} cross-domain | "
        f"{len(bad_rows)} archived")

    expected = 12
    if len(within_rows) != expected:
        log(f"  WARNING: expected {expected} within-site rows, got {len(within_rows)}")
    if len(cross_rows) != expected:
        log(f"  WARNING: expected {expected} cross-domain rows, got {len(cross_rows)}")

    # Sort
    within_rows.sort(key=sort_key_within)
    cross_rows.sort(key=sort_key_cross)

    # Print tables
    print_within(within_rows, log)
    print_cross(cross_rows, log)
    print_summary_stats(within_rows, cross_rows, log)
    print_archived(bad_rows, log)

    # Save outputs
    if not args.no_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        all_rows = within_rows + cross_rows
        write_csv(all_rows,    OUTPUT_DIR / "summary_table.csv",   mode="all")
        write_csv(within_rows, OUTPUT_DIR / "within_site.csv",     mode="within")
        write_csv(cross_rows,  OUTPUT_DIR / "cross_domain.csv",    mode="cross")
        write_json(within_rows, cross_rows, bad_rows,
                   OUTPUT_DIR / "full_results.json")

        log_path = OUTPUT_DIR / "aggregate_results.log"
        with open(log_path, "w") as f:
            f.write("\n".join(lines))

        log(f"\nOutputs written to: {OUTPUT_DIR}/")
        log(f"  summary_table.csv")
        log(f"  within_site.csv")
        log(f"  cross_domain.csv")
        log(f"  full_results.json")
        log(f"  aggregate_results.log")


if __name__ == "__main__":
    main()
