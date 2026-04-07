"""
scripts/run_adaptation.py — Domain Adaptation Experiment
=========================================================

Fine-tunes the EfficientNetV2 JA-natural cross-domain checkpoint on small
BE samples (N=500, 1000, 2000) to demonstrate the JA→BE domain gap is
bridgeable with minimal target-domain data.

Protocol per N
--------------
  Pool  : data/splits/be_natural_test.csv  (16 043 rows, never used in training)
  Sample: stratified-by-value_tier, without replacement, seed=42
  Split : sampled N → 80% finetune-train / 20% finetune-val (stratified)
  Eval  : remaining pool rows (not in the N sample) — same images the
          baseline cross-domain run was tested on (minus the N used here)

Fine-tuning settings
--------------------
  - Frozen: features[0]-features[4]  (EfficientNetV2-S backbone stem + blocks 1-5)
  - Unfrozen: features[5], features[6], features[7] (last two MBConv stages +
              head conv) + classifier
  - LR = 1e-5 (all unfrozen params), no differential LR, no scheduler
  - Loss: CrossEntropyLoss(label_smoothing=0.1), class weights from finetune-train
  - float32, no AMP
  - max 10 epochs, early stopping patience=3 (on finetune-val macro F1)

Outputs per N  (results/adaptation/efficientnetv2__ja_natural__N{n}/)
----------------------------------------------------------------------
  best_model.pth            — adapted checkpoint
  train_log.json            — per-epoch metrics
  final_metrics.json        — eval on held-out remainder
  classification_report.txt
  sampled_ids.json          — {diamond_id: value_tier} for the N sampled rows
  finetune_train.csv        — 80% of the N sample (for reproducibility)
  finetune_val.csv          — 20% of the N sample (for reproducibility)
  eval.csv                  — held-out remainder rows (for reproducibility)

Usage
-----
  cd /mnt/storage/projects/DLCGIPG
  source ja_scraper/venv/bin/activate
  python scripts/run_adaptation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader

# ── project imports ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR      = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from diamond_dataset import DiamondDataset, train_transform, val_test_transform
from models import get_model, TIER_LABELS

# ── constants ─────────────────────────────────────────────────────────────────
POOL_CSV    = PROJECT_ROOT / "data" / "splits" / "be_natural_test.csv"
IMAGE_DIR   = PROJECT_ROOT / "be_scraper" / "output" / "images"
CKPT_PATH   = PROJECT_ROOT / "results" / "training" / \
              "efficientnetv2__ja_natural__cross" / "best_model.pth"
RESULTS_DIR = PROJECT_ROOT / "results" / "adaptation"

SAMPLE_SIZES  = [500, 1000, 2000]
MAX_EPOCHS    = 10
PATIENCE      = 3
LR            = 1e-5
BATCH_SIZE    = 32
NUM_WORKERS   = 4
SEED          = 42

LABEL_NAMES  = sorted(TIER_LABELS)   # alphabetical: budget, investment_grade, mid_range, premium
BASELINE_F1  = 0.0708                 # from final_metrics.json of the cross-domain run

# ── helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze the EfficientNetV2 backbone except the last two MBConv stages
    (features[5], features[6]), the head conv (features[7]), and the classifier.

    EfficientNetV2-S features layout:
      [0] Conv2dNormActivation  — stem           → FROZEN
      [1] Sequential            — FusedMBConv s1 → FROZEN
      [2] Sequential            — FusedMBConv s2 → FROZEN
      [3] Sequential            — FusedMBConv s3 → FROZEN
      [4] Sequential            — MBConv s4      → FROZEN
      [5] Sequential            — MBConv s5      → unfrozen
      [6] Sequential            — MBConv s6      → unfrozen
      [7] Conv2dNormActivation  — head conv      → unfrozen
    classifier                                   → unfrozen
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last two MBConv stages + head conv
    for block in list(model.features)[5:]:
        for param in block.parameters():
            param.requires_grad = True

    # Unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True


def count_trainable(model: nn.Module) -> tuple[int, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def compute_class_weights(labels: list[int], num_classes: int = 4) -> torch.Tensor:
    """Inverse-frequency class weights, normalised to sum to num_classes."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)   # avoid /0
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def make_dataloader(
    df: pd.DataFrame,
    split: str,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    """
    Build a DataLoader directly from a DataFrame (no CSV needed).
    Applies train_transform for 'train', val_test_transform otherwise.
    """
    tfm = train_transform if split == "train" else val_test_transform

    class _DFDataset(DiamondDataset):
        """DiamondDataset variant that accepts a pre-filtered DataFrame."""
        def __init__(self, records_df: pd.DataFrame, image_dir: Path, transform):
            self.image_dir = image_dir
            self.transform = transform
            df2 = records_df.copy()
            df2["_img_path"] = df2.apply(
                lambda r: image_dir / str(r["value_tier"]) / f"{r['diamond_id']}.jpg",
                axis=1,
            )
            missing = ~df2["_img_path"].apply(lambda p: p.exists())
            if missing.sum():
                print(f"  [warn] skipping {missing.sum()} missing images")
                df2 = df2[~missing].reset_index(drop=True)
            self.records = df2[["diamond_id", "tier_label", "_img_path"]].reset_index(drop=True)

    dataset = _DFDataset(df, IMAGE_DIR, transform=tfm)
    shuffle  = (split == "train")
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = NUM_WORKERS,
        pin_memory  = True,
        drop_last   = (split == "train"),
    )


# ── stratified pool split ─────────────────────────────────────────────────────

def sample_pool(pool_df: pd.DataFrame, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified sample of n rows from pool_df (by value_tier).
    Returns (sampled_df, remainder_df).
    """
    sampled, remainder = train_test_split(
        pool_df,
        train_size   = n,
        stratify     = pool_df["value_tier"],
        random_state = SEED,
    )
    return sampled.reset_index(drop=True), remainder.reset_index(drop=True)


def split_finetune(sampled_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 stratified split of sampled rows into finetune-train and finetune-val."""
    ft_train, ft_val = train_test_split(
        sampled_df,
        test_size    = 0.2,
        stratify     = sampled_df["value_tier"],
        random_state = SEED,
    )
    return ft_train.reset_index(drop=True), ft_val.reset_index(drop=True)


# ── training / eval loops ────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW | None,
    device: torch.device,
    is_train: bool,
) -> dict:
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss   = criterion(logits, labels)
            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    n = len(all_labels)
    return {
        "loss":     total_loss / n,
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
    }


def evaluate_full(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss   = criterion(logits, labels)
            total_loss  += loss.item() * images.size(0)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    n          = len(all_labels)
    macro_f1   = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    per_class  = f1_score(all_labels, all_preds, average=None, zero_division=0).tolist()
    report     = classification_report(
        all_labels, all_preds, target_names=LABEL_NAMES, zero_division=0
    )
    return {
        "test_loss":            total_loss / n,
        "test_accuracy":        float(accuracy_score(all_labels, all_preds)),
        "test_macro_f1":        macro_f1,
        "per_class_f1":         {LABEL_NAMES[i]: per_class[i] for i in range(len(LABEL_NAMES))},
        "confusion_matrix":     confusion_matrix(all_labels, all_preds).tolist(),
        "label_order":          LABEL_NAMES,
        "classification_report": report,
    }


# ── main adaptation loop ──────────────────────────────────────────────────────

def adapt_one(
    n: int,
    pool_df: pd.DataFrame,
    device: torch.device,
) -> dict:
    """Run one adaptation experiment for sample size n. Returns final eval metrics."""
    print(f"\n{'='*70}")
    print(f"  ADAPTATION  N={n:,}")
    print(f"{'='*70}")

    out_dir = RESULTS_DIR / f"efficientnetv2__ja_natural__N{n}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── split pool ────────────────────────────────────────────────────────────
    sampled_df, eval_df = sample_pool(pool_df, n)
    ft_train_df, ft_val_df = split_finetune(sampled_df)

    print(f"  Pool      : {len(pool_df):,} rows")
    print(f"  Sampled   : {len(sampled_df):,}  "
          f"(ft-train={len(ft_train_df):,}  ft-val={len(ft_val_df):,})")
    print(f"  Eval (held-out): {len(eval_df):,} rows")

    # Class distribution in finetune-train
    dist = ft_train_df["value_tier"].value_counts().to_dict()
    print(f"  FT-train class dist: {dist}")

    # ── save artefacts for reproducibility ───────────────────────────────────
    sampled_ids = {
        str(row["diamond_id"]): row["value_tier"]
        for _, row in sampled_df.iterrows()
    }
    (out_dir / "sampled_ids.json").write_text(json.dumps(sampled_ids, indent=2))
    ft_train_df.to_csv(out_dir / "finetune_train.csv", index=False)
    ft_val_df.to_csv(  out_dir / "finetune_val.csv",   index=False)
    eval_df.to_csv(    out_dir / "eval.csv",            index=False)

    # ── data loaders ─────────────────────────────────────────────────────────
    ft_train_loader = make_dataloader(ft_train_df, "train")
    ft_val_loader   = make_dataloader(ft_val_df,   "val")
    eval_loader     = make_dataloader(eval_df,     "test")

    # Guard: DataLoader may filter missing images
    if len(ft_train_loader.dataset) == 0:
        raise RuntimeError(f"No valid training images found for N={n}.")

    # ── model ─────────────────────────────────────────────────────────────────
    model, _ = get_model("efficientnetv2", dropout=0.3)
    state    = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state)
    freeze_backbone(model)
    model = model.to(device)

    trainable, total = count_trainable(model)
    print(f"  Params: total={total:,}  trainable={trainable:,}  "
          f"frozen={total-trainable:,}")

    # ── loss & optimizer ──────────────────────────────────────────────────────
    # Class weights from the actual finetune-train labels (post image-filter)
    ft_labels = ft_train_loader.dataset.labels
    cw = compute_class_weights(ft_labels).to(device)
    print(f"  Class weights: { {LABEL_NAMES[i]: f'{cw[i].item():.3f}' for i in range(4)} }")

    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr           = LR,
        weight_decay = 1e-4,
    )

    # ── fine-tune ─────────────────────────────────────────────────────────────
    ckpt_path     = out_dir / "best_model.pth"
    best_val_f1   = -1.0
    patience_ctr  = 0
    epoch_log     = []

    print(f"\n  {'Ep':>3}  {'FT-Train Loss':>13}  {'FT-Train F1':>11}  "
          f"{'FT-Val Loss':>11}  {'FT-Val F1':>9}  {'Time':>6}")
    print("  " + "─" * 62)

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        tr = run_epoch(model, ft_train_loader, criterion, optimizer, device, is_train=True)
        vl = run_epoch(model, ft_val_loader,   criterion, None,      device, is_train=False)
        elapsed = time.time() - t0

        row = {"epoch": epoch, "train": tr, "val": vl, "elapsed_s": round(elapsed, 1)}
        epoch_log.append(row)

        print(f"  {epoch:>3}  {tr['loss']:>13.4f}  {tr['macro_f1']:>11.4f}  "
              f"{vl['loss']:>11.4f}  {vl['macro_f1']:>9.4f}  {elapsed:>5.1f}s")

        if vl["macro_f1"] > best_val_f1:
            best_val_f1  = vl["macro_f1"]
            patience_ctr = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"       ✓ best FT-val F1={best_val_f1:.4f} — checkpoint saved")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (patience={PATIENCE} exhausted).")
                break

    (out_dir / "train_log.json").write_text(json.dumps(epoch_log, indent=2))

    # ── eval on held-out remainder ────────────────────────────────────────────
    print(f"\n  Loading best checkpoint ({ckpt_path.name}) for held-out eval …")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Criterion without class weights for eval (unbiased loss estimate)
    eval_criterion = nn.CrossEntropyLoss()
    results = evaluate_full(model, eval_loader, eval_criterion, device)

    print(f"  Held-out eval  ({len(eval_loader.dataset):,} samples)")
    print(f"    Accuracy : {results['test_accuracy']:.4f}")
    print(f"    Macro F1 : {results['test_macro_f1']:.4f}  "
          f"(baseline: {BASELINE_F1:.4f}  Δ={results['test_macro_f1']-BASELINE_F1:+.4f})")
    for cls, f1v in results["per_class_f1"].items():
        print(f"    {cls:<20s}: {f1v:.4f}")

    # ── save final metrics ────────────────────────────────────────────────────
    report_text = results.pop("classification_report")
    final = {
        "experiment":         f"efficientnetv2__ja_natural__N{n}",
        "arch":               "efficientnetv2",
        "source_checkpoint":  str(CKPT_PATH),
        "adaptation_n":       n,
        "ft_train_size":      len(ft_train_loader.dataset),
        "ft_val_size":        len(ft_val_loader.dataset),
        "eval_size":          len(eval_loader.dataset),
        "best_ft_val_f1":     best_val_f1,
        "hyperparams": {
            "epochs_run":    len(epoch_log),
            "epochs_max":    MAX_EPOCHS,
            "lr":            LR,
            "batch_size":    BATCH_SIZE,
            "patience":      PATIENCE,
            "frozen_blocks": "features[0:5]",
            "unfrozen":      "features[5:8] + classifier",
        },
        **results,
    }
    (out_dir / "final_metrics.json").write_text(json.dumps(final, indent=2))
    (out_dir / "classification_report.txt").write_text(report_text)
    print(f"  Results saved → {out_dir}/")

    return final


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: No CUDA GPU detected — running on CPU (will be slow).")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Verify paths up front so failures are fast
    for p, label in [(POOL_CSV, "Pool CSV"), (CKPT_PATH, "Checkpoint"), (IMAGE_DIR, "Image dir")]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{label} not found: {p}")
    print(f"Pool CSV  : {POOL_CSV}  ({POOL_CSV.stat().st_size // 1024} KB)")
    print(f"Checkpoint: {CKPT_PATH}  ({CKPT_PATH.stat().st_size // (1024*1024)} MB)")

    # Load pool — keep only columns DiamondDataset needs + value_tier for stratification
    pool_df = pd.read_csv(
        POOL_CSV,
        usecols     = ["diamond_id", "value_tier", "tier_label"],
        low_memory  = False,
    )
    print(f"Pool size : {len(pool_df):,} rows")
    print(f"Class dist: {pool_df['value_tier'].value_counts().to_dict()}")

    summary_rows = []
    for n in SAMPLE_SIZES:
        metrics = adapt_one(n, pool_df, device)
        summary_rows.append({
            "N":          n,
            "eval_size":  metrics["eval_size"],
            "macro_f1":   metrics["test_macro_f1"],
            "accuracy":   metrics["test_accuracy"],
        })

    # ── before / after summary table ─────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  DOMAIN ADAPTATION SUMMARY")
    print(f"  Baseline (JA→BE, no adaptation, full test): macro F1 = {BASELINE_F1:.4f}")
    print(f"{'='*70}")
    print(f"  {'N':>6}  {'Eval size':>10}  {'Macro F1':>9}  {'Delta vs baseline':>18}")
    print("  " + "─" * 50)
    for row in summary_rows:
        delta = row["macro_f1"] - BASELINE_F1
        print(f"  {row['N']:>6,}  {row['eval_size']:>10,}  "
              f"{row['macro_f1']:>9.4f}  {delta:>+18.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
