"""
One-off script: backfill best_epoch and epochs_trained into classification
final_metrics.json files from their companion train_log.json files.

Skips: *_bad directories, results/regression/, results/adaptation/
"""

import json
import pathlib

TRAINING_ROOT = pathlib.Path(__file__).parent.parent / "results" / "training"
SKIP_SUFFIXES = ("_bad",)
SKIP_SUBDIRS = {"regression", "adaptation"}


def find_pairs():
    pairs = []
    for metrics_path in sorted(TRAINING_ROOT.rglob("final_metrics.json")):
        folder = metrics_path.parent
        # Skip bad-attempt dirs
        if any(part.endswith(s) for part in folder.parts for s in SKIP_SUFFIXES):
            continue
        # Skip regression and adaptation subtrees
        rel = folder.relative_to(TRAINING_ROOT)
        if rel.parts and rel.parts[0] in SKIP_SUBDIRS:
            continue
        log_path = folder / "train_log.json"
        if log_path.exists():
            pairs.append((folder, metrics_path, log_path))
    return pairs


def best_from_log(log_path):
    with open(log_path) as f:
        log = json.load(f)
    if not isinstance(log, list) or not log:
        raise ValueError(f"Unexpected train_log format in {log_path}")
    best = max(log, key=lambda e: e["val"]["macro_f1"])
    return best["epoch"], len(log)


def backfill(metrics_path, best_epoch, epochs_trained):
    with open(metrics_path) as f:
        metrics = json.load(f)
    metrics["best_epoch"] = best_epoch
    metrics["epochs_trained"] = epochs_trained
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")


def main():
    pairs = find_pairs()
    if not pairs:
        print("No matching pairs found.")
        return

    rows = []
    for folder, metrics_path, log_path in pairs:
        best_epoch, epochs_trained = best_from_log(log_path)
        backfill(metrics_path, best_epoch, epochs_trained)
        rows.append((folder.name, best_epoch, epochs_trained))

    # Print confirmation table
    col1 = max(len(r[0]) for r in rows)
    header = f"{'folder':<{col1}}  {'best_epoch':>10}  {'epochs_trained':>14}"
    print(header)
    print("-" * len(header))
    for name, be, et in rows:
        print(f"{name:<{col1}}  {be:>10}  {et:>14}")
    print(f"\nUpdated {len(rows)} files.")


if __name__ == "__main__":
    main()
