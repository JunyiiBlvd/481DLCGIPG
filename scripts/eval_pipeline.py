"""
eval_pipeline.py — End-to-end pipeline evaluation: Stage 1 (species) → Stage 2 (tier)

Stage 1: ResNet50, 68-class gemstone species classifier (Sebastian)
Stage 2: EfficientNetV2-S, 4-class diamond value tier classifier

Flow:
  All images → Stage 1 → predicted "Diamond"? → Stage 2 → tier prediction

Usage:
    python scripts/eval_pipeline.py \
        --stage1_ckpt results/training/stage1/best_model.pth \
        --stage2_ckpt results/training/efficientnetv2__ja_natural__within/best_model.pth \
        --manifest    data/combined_pipeline_manifest.csv \
        --out_dir     results/pipeline_eval

Outputs (under --out_dir):
  - pipeline_metrics.json
  - pipeline_report.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import EfficientNet_V2_S_Weights, ResNet50_Weights

ROOT    = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"

# ── ImageNet transforms (same as train/eval throughout project) ───────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])

TIER_LABELS = ["budget", "investment_grade", "mid_range", "premium"]  # alphabetical
TIER_TO_IDX = {t: i for i, t in enumerate(TIER_LABELS)}


# ── Dataset ───────────────────────────────────────────────────────────────────

class ManifestDataset(Dataset):
    """Loads images from a path column in the combined manifest CSV."""

    def __init__(self, df: pd.DataFrame):
        self.records = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path = self.records.at[idx, "image_path"]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        return TRANSFORM(img), idx


# ── Model builders ────────────────────────────────────────────────────────────

def load_stage1(ckpt_path: str | Path, num_classes: int, device: torch.device) -> nn.Module:
    """ResNet50 with plain Linear head (Sebastian's architecture)."""
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def load_stage2(ckpt_path: str | Path, arch: str, device: torch.device) -> nn.Module:
    """Load a Stage 2 model. arch must be 'efficientnetv2' or 'resnet50'."""
    if arch == "efficientnetv2":
        model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, len(TIER_LABELS)),
        )
    elif arch == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, len(TIER_LABELS)),
        )
    else:
        raise ValueError(f"Unsupported Stage 2 arch: {arch}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model: nn.Module, df: pd.DataFrame,
                  device: torch.device, batch_size: int,
                  num_workers: int) -> np.ndarray:
    """Run model on all rows of df, return predicted class indices."""
    dataset = ManifestDataset(df)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    preds = np.empty(len(df), dtype=np.int64)
    with torch.no_grad():
        for imgs, idxs in loader:
            imgs   = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            batch_preds = logits.argmax(dim=1).cpu().numpy()
            preds[idxs.numpy()] = batch_preds
    return preds


# ── Metrics helpers ───────────────────────────────────────────────────────────

def stage1_metrics(df: pd.DataFrame, preds: np.ndarray,
                   classes: list[str]) -> dict:
    """Compute Stage 1 accuracy on Kaggle images and diamond routing rate."""
    kaggle = df[df["subset"] == "kaggle_stage1"].copy()
    kaggle["pred_species"] = [classes[p] for p in preds[kaggle.index]]

    acc = (kaggle["pred_species"] == kaggle["species"]).mean()

    # Diamond-specific: of all ground-truth Diamond images, how many predicted Diamond?
    gt_diamond = kaggle[kaggle["species"] == "Diamond"]
    diamond_recall = (
        (gt_diamond["pred_species"] == "Diamond").mean()
        if len(gt_diamond) > 0 else float("nan")
    )

    # Of all Stage 2 images (known diamonds), how many does Stage 1 route correctly?
    stage2_imgs = df[df["subset"] != "kaggle_stage1"].copy()
    stage2_imgs["pred_species"] = [classes[p] for p in preds[stage2_imgs.index]]
    routing_rate = (stage2_imgs["pred_species"] == "Diamond").mean()

    return {
        "kaggle_test_accuracy":          float(acc),
        "kaggle_diamond_recall":         float(diamond_recall),
        "stage2_image_routing_rate":     float(routing_rate),
        "kaggle_n":                      int(len(kaggle)),
        "stage2_n":                      int(len(stage2_imgs)),
        "correctly_routed_stage2":       int((stage2_imgs["pred_species"] == "Diamond").sum()),
    }


def stage2_metrics(df: pd.DataFrame, s1_preds: np.ndarray,
                   s2_preds: np.ndarray, classes: list[str]) -> dict:
    """
    Compute Stage 2 tier accuracy on images that Stage 1 routed to Stage 2.
    Also computes end-to-end accuracy on the full Stage 2 ground-truth set.
    """
    stage2_gt = df[df["value_tier"].isin(TIER_LABELS)].copy()
    stage2_gt["s1_pred_species"] = [classes[p] for p in s1_preds[stage2_gt.index]]
    stage2_gt["s2_pred_tier"]    = [TIER_LABELS[p] for p in s2_preds[stage2_gt.index]]

    # End-to-end: Stage 1 must route correctly AND Stage 2 must predict correctly
    routed   = stage2_gt[stage2_gt["s1_pred_species"] == "Diamond"]
    e2e_correct = (routed["s2_pred_tier"] == routed["value_tier"]).sum()
    e2e_acc     = e2e_correct / len(stage2_gt)  # denominator = full ground truth set

    # Stage 2 isolated accuracy (on routed images only)
    s2_acc = (routed["s2_pred_tier"] == routed["value_tier"]).mean() if len(routed) > 0 else float("nan")

    # Per-tier F1 (on routed images)
    if len(routed) > 0:
        gt_idx   = [TIER_TO_IDX[t] for t in routed["value_tier"]]
        pred_idx = [TIER_TO_IDX[t] for t in routed["s2_pred_tier"]]
        macro_f1 = float(f1_score(gt_idx, pred_idx, average="macro", zero_division=0))
        per_tier = f1_score(gt_idx, pred_idx, average=None,
                            labels=list(range(len(TIER_LABELS))),
                            zero_division=0).tolist()
        per_tier_f1 = {TIER_LABELS[i]: per_tier[i] for i in range(len(TIER_LABELS))}
    else:
        macro_f1    = float("nan")
        per_tier_f1 = {}

    # Per-subset breakdown
    subset_breakdown = {}
    for subset in stage2_gt["subset"].unique():
        sub = stage2_gt[stage2_gt["subset"] == subset]
        sub_routed = sub[sub["s1_pred_species"] == "Diamond"]
        sub_e2e = (sub_routed["s2_pred_tier"] == sub_routed["value_tier"]).sum() / len(sub)
        subset_breakdown[subset] = {
            "n_total":         int(len(sub)),
            "n_routed":        int(len(sub_routed)),
            "routing_rate":    float(len(sub_routed) / len(sub)),
            "e2e_accuracy":    float(sub_e2e),
        }

    return {
        "stage2_accuracy_on_routed":  float(s2_acc),
        "stage2_macro_f1_on_routed":  macro_f1,
        "stage2_per_tier_f1":         per_tier_f1,
        "end_to_end_accuracy":        float(e2e_acc),
        "n_routed_to_stage2":         int(len(routed)),
        "n_stage2_ground_truth":      int(len(stage2_gt)),
        "per_subset":                 subset_breakdown,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_ckpt", default=str(ROOT / "results/training/stage1/best_model.pth"))
    p.add_argument("--stage1_meta", default=str(ROOT / "results/training/stage1/final_metrics.json"),
                   help="final_metrics.json from Stage 1 (contains class list)")
    p.add_argument("--manifest",    default=str(ROOT / "data/combined_pipeline_manifest.csv"))
    p.add_argument("--out_dir",     default=str(ROOT / "results/pipeline_eval"))
    p.add_argument("--batch_size",  type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Load class list from Stage 1 metadata ─────────────────────────────────
    with open(args.stage1_meta) as f:
        meta = json.load(f)
    classes = meta["classes"]          # list of 68 class names, order = model output index
    diamond_idx = classes.index("Diamond")
    print(f"Stage 1: {len(classes)} classes  Diamond index: {diamond_idx}")

    # ── Best Stage 2 model per subset (by within-site macro F1) ─────────────
    training_dir = ROOT / "results" / "training"
    stage2_configs = {
        "ja_natural": {"arch": "efficientnetv2", "f1": 0.6724},
        "ja_lab":     {"arch": "efficientnetv2", "f1": 0.6589},
        "be_natural": {"arch": "efficientnetv2", "f1": 0.6093},
        "be_lab":     {"arch": "resnet50",        "f1": 0.5557},
    }
    for subset, cfg in stage2_configs.items():
        cfg["ckpt"] = training_dir / f"{cfg['arch']}__{subset}__within" / "best_model.pth"
        assert cfg["ckpt"].exists(), f"Missing Stage 2 checkpoint: {cfg['ckpt']}"

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading Stage 1 (ResNet50)...")
    stage1 = load_stage1(args.stage1_ckpt, len(classes), device)

    # ── Load manifest ─────────────────────────────────────────────────────────
    df = pd.read_csv(args.manifest)
    print(f"Manifest: {len(df):,} images")

    # ── Stage 1 inference (all images) ───────────────────────────────────────
    print("\nRunning Stage 1 inference...")
    s1_preds = run_inference(stage1, df, device, args.batch_size, args.num_workers)
    del stage1
    if device.type == "cuda":
        torch.cuda.empty_cache()

    routed_mask = s1_preds == diamond_idx
    print(f"Stage 1 routed {routed_mask.sum():,} / {len(df):,} images to Stage 2")

    # ── Stage 2 inference — each subset uses its own within-site model ────────
    s2_preds_routed = np.full(len(df), -1, dtype=np.int64)
    for subset, cfg in stage2_configs.items():
        subset_mask      = (df["subset"] == subset).values & routed_mask
        subset_routed_df = df[subset_mask].reset_index(drop=True)
        if len(subset_routed_df) == 0:
            continue
        print(f"  [{subset}] {cfg['arch']} (val F1={cfg['f1']}) — {len(subset_routed_df):,} routed images")
        stage2 = load_stage2(cfg["ckpt"], cfg["arch"], device)
        preds  = run_inference(stage2, subset_routed_df, device, args.batch_size, args.num_workers)
        s2_preds_routed[subset_mask] = preds
        del stage2
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Compute metrics ───────────────────────────────────────────────────────
    print("\nComputing metrics...")
    s1_m = stage1_metrics(df, s1_preds, classes)
    s2_m = stage2_metrics(df, s1_preds, s2_preds_routed, classes)

    results = {
        "stage1_ckpt":  args.stage1_ckpt,
        "stage2_ckpts": {k: {"arch": v["arch"], "f1": v["f1"], "ckpt": str(v["ckpt"])}
                         for k, v in stage2_configs.items()},
        "manifest":     args.manifest,
        "stage1":       s1_m,
        "stage2":       s2_m,
    }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    metrics_path = out_dir / "pipeline_metrics_per_subset_stage2.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Print report ──────────────────────────────────────────────────────────
    report = []
    report.append("=" * 60)
    report.append("PIPELINE EVALUATION — BEST MODEL PER SUBSET")
    report.append("=" * 60)
    report.append("\nNOTE: Each subset uses its best within-site Stage 2 model.")
    report.append(f"\nStage 1 checkpoint : {args.stage1_ckpt}")
    report.append("Stage 2 models (best per subset):")
    for subset, cfg in stage2_configs.items():
        report.append(f"  {subset:<12s}: {cfg['arch']:<16s} within-site F1={cfg['f1']:.4f}")
    report.append(f"Manifest           : {args.manifest}  ({len(df):,} images)")

    report.append(f"\n{'─'*60}")
    report.append("STAGE 1 — Gemstone Species (ResNet50, 68 classes)")
    report.append(f"{'─'*60}")
    report.append(f"  Kaggle test accuracy       : {s1_m['kaggle_test_accuracy']:.4f}")
    report.append(f"  Kaggle Diamond recall      : {s1_m['kaggle_diamond_recall']:.4f}")
    report.append(f"  JA/BE routing rate         : {s1_m['stage2_image_routing_rate']:.4f}")
    report.append(f"  (correctly routed          : {s1_m['correctly_routed_stage2']:,} / {s1_m['stage2_n']:,})")

    report.append(f"\n{'─'*60}")
    report.append("STAGE 2 — Value Tier (EfficientNetV2-S, 4 classes)")
    report.append(f"{'─'*60}")
    report.append(f"  Accuracy (on routed imgs)  : {s2_m['stage2_accuracy_on_routed']:.4f}")
    report.append(f"  Macro F1 (on routed imgs)  : {s2_m['stage2_macro_f1_on_routed']:.4f}")
    report.append("  Per-tier F1:")
    for tier, f1 in s2_m["stage2_per_tier_f1"].items():
        report.append(f"    {tier:<20s}: {f1:.4f}")

    report.append(f"\n{'─'*60}")
    report.append("END-TO-END")
    report.append(f"{'─'*60}")
    report.append(f"  End-to-end accuracy        : {s2_m['end_to_end_accuracy']:.4f}")
    report.append(f"  (correct / total GT        : — / {s2_m['n_stage2_ground_truth']:,})")

    report.append(f"\n{'─'*60}")
    report.append("PER-SUBSET BREAKDOWN")
    report.append(f"{'─'*60}")
    for subset, d in s2_m["per_subset"].items():
        report.append(f"  {subset}:")
        report.append(f"    routing rate  : {d['routing_rate']:.4f}  ({d['n_routed']:,}/{d['n_total']:,})")
        report.append(f"    e2e accuracy  : {d['e2e_accuracy']:.4f}")

    report_str = "\n".join(report)
    print(report_str)

    report_path = out_dir / "pipeline_report_per_subset_stage2.txt"
    with open(report_path, "w") as f:
        f.write(report_str + "\n")

    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
