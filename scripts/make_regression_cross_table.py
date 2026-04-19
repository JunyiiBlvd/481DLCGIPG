"""
make_regression_cross_table.py
================================
Reads the 4 regression cross-domain final_metrics.json files and the
existing results/aggregated/cross_domain.csv (classification results),
prints a comparison table, and saves:
    results/aggregated/regression_cross_comparison.csv

Clf F1 column uses EfficientNetV2-S rows from cross_domain.csv so the
backbone is consistent with the regression model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

RUNS = [
    ("ja_natural", "be_natural", "JA→BE natural"),
    ("ja_lab",     "be_lab",     "JA→BE lab"),
    ("be_natural", "ja_natural", "BE→JA natural"),
    ("be_lab",     "ja_lab",     "BE→JA lab"),
]

# Maps (train_site, origin) → direction key matching cross_domain.csv columns
# cross_domain.csv: arch, train_site, train_origin, train_subset, mode, direction, test_site, macro_f1, ...
CLF_ARCH = "EfficientNetV2-S"


def load_clf_f1(cross_domain_csv: Path) -> dict[str, float]:
    """Return {direction_label: macro_f1} for EfficientNetV2-S rows."""
    df = pd.read_csv(cross_domain_csv)
    df = df[df["arch"] == CLF_ARCH]

    mapping: dict[str, float] = {}
    for _, row in df.iterrows():
        src = row["train_subset"].lower().replace(" ", "_")   # e.g. "ja_natural"
        # derive target subset name
        if src.startswith("ja"):
            tgt = src.replace("ja_", "be_")
        else:
            tgt = src.replace("be_", "ja_")
        key = f"{src}→{tgt}"
        mapping[key] = float(row["macro_f1"])
    return mapping


def main() -> None:
    cross_domain_csv = ROOT / "results" / "aggregated" / "cross_domain.csv"
    if not cross_domain_csv.exists():
        print(f"ERROR: {cross_domain_csv} not found", file=sys.stderr)
        sys.exit(1)

    clf_f1_map = load_clf_f1(cross_domain_csv)

    rows = []
    missing = []

    for src, tgt, label in RUNS:
        metrics_path = (
            ROOT / "results" / "training" / "regression_cross"
            / f"efficientnetv2__{src}__{tgt}" / "final_metrics.json"
        )
        if not metrics_path.exists():
            missing.append(metrics_path)
            print(f"  MISSING: {metrics_path}")
            continue

        with open(metrics_path) as f:
            m = json.load(f)

        direction_key = f"{src}→{tgt}"
        reg_f1 = m.get("test_tier_macro_f1", float("nan"))
        clf_f1 = clf_f1_map.get(direction_key, float("nan"))
        delta  = reg_f1 - clf_f1 if (reg_f1 == reg_f1 and clf_f1 == clf_f1) else float("nan")

        rows.append({
            "direction":      label,
            "train_subset":   src,
            "test_subset":    tgt,
            "reg_cross_f1":   reg_f1,
            "clf_cross_f1":   clf_f1,
            "delta":          delta,
        })

    if missing:
        print(f"\n{len(missing)} run(s) not yet complete — partial table shown.\n")

    if not rows:
        print("No completed runs found.")
        sys.exit(0)

    df = pd.DataFrame(rows)

    # ── Print formatted table ─────────────────────────────────────────────────
    sep = "─" * 85
    print(f"\n{sep}")
    print(f"{'Direction':<18} {'Train':<12} {'Test':<12} "
          f"{'Reg Cross F1':>14} {'Clf Cross F1':>13} {'Delta':>8}")
    print(sep)
    for _, r in df.iterrows():
        print(f"{r['direction']:<18} {r['train_subset']:<12} {r['test_subset']:<12} "
              f"{r['reg_cross_f1']:>14.4f} {r['clf_cross_f1']:>13.4f} {r['delta']:>+8.4f}")
    print(sep)
    print(f"Note: Clf F1 = {CLF_ARCH} macro F1 from cross_domain.csv")
    print(f"      Reg F1 = regression→tier bucketing (target-domain log-price thresholds)")
    print(f"      Delta  = Reg F1 − Clf F1  (positive = regression generalises better)\n")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_path = ROOT / "results" / "aggregated" / "regression_cross_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
