"""
aggregate_results.py — Collects all Stage 2 training results into a single table
==================================================================================

Scans results/training/**/final_metrics.json and produces:
  - results/summary/stage2_results.csv     ← all runs, one row per run
  - results/summary/stage2_results.md      ← Markdown table for the paper

Usage:
    python src/aggregate_results.py \\
        --results_dir /mnt/storage/projects/DLCGIPG/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TIER_LABELS = sorted(["budget", "investment_grade", "mid_range", "premium"])


def load_result(path: Path) -> dict | None:
    try:
        with open(path) as f:
            data = json.load(f)
        row = {
            "arch":           data.get("arch", "?"),
            "subset":         data.get("subset", "?"),
            "mode":           "cross" if data.get("cross_domain") else "within",
            "test_accuracy":  round(data.get("test_accuracy", 0.0), 4),
            "test_macro_f1":  round(data.get("test_macro_f1", 0.0), 4),
            "best_val_f1":    round(data.get("best_val_f1", 0.0), 4),
            "epochs_run":     data.get("hyperparams", {}).get("epochs_run", "?"),
        }
        for cls in TIER_LABELS:
            row[f"f1_{cls}"] = round(data.get("per_class_f1", {}).get(cls, 0.0), 4)
        return row
    except Exception as e:
        print(f"  Warning: could not load {path}: {e}")
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    summary_dir = results_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for metrics_file in sorted(results_dir.glob("training/**/final_metrics.json")):
        row = load_result(metrics_file)
        if row:
            rows.append(row)
            print(f"  Loaded: {metrics_file.parent.name}")

    if not rows:
        print("No results found under", results_dir / "training")
        return

    df = pd.DataFrame(rows).sort_values(["mode", "subset", "arch"])

    csv_path = summary_dir / "stage2_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV → {csv_path}")

    # ── Markdown table ────────────────────────────────────────────────────────
    display_cols = [
        "arch", "subset", "mode",
        "test_accuracy", "test_macro_f1",
        "f1_budget", "f1_investment_grade", "f1_mid_range", "f1_premium",
        "best_val_f1", "epochs_run",
    ]
    md = df[display_cols].to_markdown(index=False)
    md_path = summary_dir / "stage2_results.md"
    with open(md_path, "w") as f:
        f.write("# Stage 2 Results — DLCGIPG\n\n")
        f.write(md)
        f.write("\n\n*RF baseline (tabular ceiling):*\n")
        f.write("| subset | test_accuracy | test_macro_f1 |\n")
        f.write("|---|---|---|\n")
        rf_baselines = {
            "ja_natural": (0.8658, 0.8567),
            "ja_lab":     (0.9411, 0.9217),
            "be_natural": (0.8371, 0.8226),
            "be_lab":     (0.8961, 0.8742),
        }
        for k, (acc, f1) in rf_baselines.items():
            f.write(f"| {k} | {acc} | {f1} |\n")
    print(f"MD  → {md_path}")

    # ── console summary ───────────────────────────────────────────────────────
    print("\n── Within-site summary ──────────────────────────────")
    within = df[df["mode"] == "within"][["arch", "subset", "test_accuracy", "test_macro_f1"]]
    print(within.to_string(index=False))

    print("\n── Cross-domain summary ─────────────────────────────")
    cross = df[df["mode"] == "cross"][["arch", "subset", "test_accuracy", "test_macro_f1"]]
    print(cross.to_string(index=False))


if __name__ == "__main__":
    main()
