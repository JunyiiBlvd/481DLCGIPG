"""
rf_baseline.py — Random Forest tabular baseline for diamond tier classification.

Features: carat (continuous), cut / color / clarity (ordinal-encoded)
Target:   tier_label  {0: budget, 1: mid_range, 2: premium, 3: investment_grade}

Run:
    python rf_baseline.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# ── Paths ─────────────────────────────────────────────────────────────────────
SPLITS  = Path("/mnt/storage/projects/DLCGIPG/data/splits")
OUT_DIR = Path("/mnt/storage/projects/DLCGIPG/results/rf_baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["carat", "cut", "color", "clarity"]
TARGET   = "tier_label"

# ── Ordinal mappings (higher = better quality) ────────────────────────────────
# Encoding: worst grade → 0, best grade → max index
CUT_ORDER     = ["Fair", "Good", "Very Good", "Ideal", "Ideal+"]        # 0–4
COLOR_ORDER   = ["K", "J", "I", "H", "G", "F", "E", "D"]              # 0–7
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF", "FL"]  # 0–8

CUT_MAP     = {v: i for i, v in enumerate(CUT_ORDER)}
COLOR_MAP   = {v: i for i, v in enumerate(COLOR_ORDER)}
CLARITY_MAP = {v: i for i, v in enumerate(CLARITY_ORDER)}

ORDINAL_MAPS = {"cut": CUT_MAP, "color": COLOR_MAP, "clarity": CLARITY_MAP}

TIER_NAMES = {0: "budget", 1: "mid_range", 2: "premium", 3: "investment_grade"}
DATASETS   = ["ja_natural", "ja_lab", "be_natural", "be_lab"]


def sep(t): print(f"\n{'='*58}\n  {t}\n{'='*58}")


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply fixed ordinal maps; unknown values become NaN → filled with -1."""
    out = df[FEATURES].copy()
    for col, mapping in ORDINAL_MAPS.items():
        out[col] = out[col].map(mapping).fillna(-1).astype(int)
    return out


def load_split(name: str, split: str) -> tuple[pd.DataFrame, pd.Series]:
    path = SPLITS / f"{name}_{split}.csv"
    df   = pd.read_csv(path, usecols=FEATURES + [TARGET], low_memory=False)
    return df[FEATURES], df[TARGET]


def run_experiment(dataset_name: str) -> dict:
    sep(dataset_name)

    # ── Load ──────────────────────────────────────────────────
    X_train_raw, y_train = load_split(dataset_name, "train")
    X_val_raw,   y_val   = load_split(dataset_name, "val")
    X_test_raw,  y_test  = load_split(dataset_name, "test")

    # Encoding is a fixed map (fit implicitly on domain knowledge, not data)
    X_train = encode_features(X_train_raw)
    X_val   = encode_features(X_val_raw)
    X_test  = encode_features(X_test_raw)

    # Report any OOV grades in val/test
    for split_name, raw in [("val", X_val_raw), ("test", X_test_raw)]:
        for col, mapping in ORDINAL_MAPS.items():
            oov = set(raw[col].dropna().unique()) - set(mapping.keys())
            if oov:
                print(f"  WARNING [{split_name}] OOV {col} values → -1: {oov}")

    print(f"\n  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # ── Train ─────────────────────────────────────────────────
    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────
    results = {}
    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        preds   = clf.predict(X)
        acc     = accuracy_score(y, preds)
        macro_f1 = f1_score(y, preds, average="macro")
        results[split_name] = {"accuracy": round(acc, 6), "macro_f1": round(macro_f1, 6)}

        print(f"\n  [{split_name}]  accuracy={acc:.4f}  macro_f1={macro_f1:.4f}")

        # Per-class F1
        per_class = f1_score(y, preds, average=None, labels=[0, 1, 2, 3])
        for cls, f1 in enumerate(per_class):
            print(f"    class {cls} ({TIER_NAMES[cls]:20s})  F1={f1:.4f}")

    # ── Feature importances ───────────────────────────────────
    importances = dict(zip(FEATURES, clf.feature_importances_.round(6).tolist()))
    print(f"\n  Feature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        flag = "  ← ⚠ labeling artifact?" if feat == "carat" and imp > 0.80 else ""
        print(f"    {feat:8s}  {imp:.4f}  {bar}{flag}")

    if importances["carat"] > 0.80:
        print(f"\n  WARNING: carat importance {importances['carat']:.4f} > 0.80 — "
              f"tier labels may be carat-driven rather than value-driven.")

    results["feature_importances"] = importances
    results["dataset"] = dataset_name
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    all_results = {}
    summary_rows = []

    for name in DATASETS:
        res = run_experiment(name)
        all_results[name] = res

        test = res["test"]
        summary_rows.append({
            "split":            name,
            "test_accuracy":    f"{test['accuracy']:.4f}",
            "macro_f1":         f"{test['macro_f1']:.4f}",
            "carat_importance": f"{res['feature_importances']['carat']:.4f}",
        })

        out_path = OUT_DIR / f"{name}_results.json"
        out_path.write_text(json.dumps(res, indent=2))

    # ── Summary table ─────────────────────────────────────────
    sep("Summary — Test Set Performance")
    df = pd.DataFrame(summary_rows)
    print(df.to_string(index=False))

    # Save combined results
    (OUT_DIR / "all_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
