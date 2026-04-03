"""
models.py — Model factory for DLCGIPG Stage 2

Supported architectures:
  - resnet50      : ResNet-50 (He et al., 2016), ImageNet1K_V2
  - efficientnetv2: EfficientNetV2-S (Tan & Le, 2021), ImageNet1K_V1
  - vit            : ViT-B/16 (Dosovitskiy et al., 2021), ImageNet1K_V1

All models:
  - Pretrained on ImageNet-1K
  - Final classification head replaced for NUM_CLASSES=4
  - Returns (model, param_groups) where param_groups carries differential LRs:
      backbone: base_lr * backbone_lr_scale  (default 0.1)
      head    : base_lr
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet50_Weights,
    EfficientNet_V2_S_Weights,
    ViT_B_16_Weights,
)

NUM_CLASSES = 4
TIER_LABELS = ["budget", "investment_grade", "mid_range", "premium"]  # alphabetical (sklearn default)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _freeze_backbone(model: nn.Module, freeze: bool) -> None:
    """Freeze or unfreeze all parameters except the head."""
    for name, param in model.named_parameters():
        if "head" not in name and "classifier" not in name and "fc" not in name:
            param.requires_grad = not freeze


def _build_param_groups(
    model: nn.Module,
    head_names: list[str],
    base_lr: float,
    backbone_lr_scale: float,
) -> list[dict]:
    """
    Split parameters into two groups:
      - backbone: lr = base_lr * backbone_lr_scale
      - head    : lr = base_lr
    head_names: list of substrings that identify head module names.
    """
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if any(h in name for h in head_names):
            head_params.append(param)
        else:
            backbone_params.append(param)
    return [
        {"params": backbone_params, "lr": base_lr * backbone_lr_scale},
        {"params": head_params,     "lr": base_lr},
    ]


# ---------------------------------------------------------------------------
# ResNet-50
# ---------------------------------------------------------------------------

def build_resnet50(
    base_lr: float = 3e-4,
    backbone_lr_scale: float = 0.1,
    dropout: float = 0.3,
) -> tuple[nn.Module, list[dict]]:
    """
    ResNet-50 with a two-layer classification head.
    Original 1000-class FC replaced by:
      Linear(2048 → 512) → ReLU → Dropout(p) → Linear(512 → 4)
    """
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(512, NUM_CLASSES),
    )
    param_groups = _build_param_groups(model, ["fc"], base_lr, backbone_lr_scale)
    return model, param_groups


# ---------------------------------------------------------------------------
# EfficientNetV2-S
# ---------------------------------------------------------------------------

def build_efficientnetv2(
    base_lr: float = 3e-4,
    backbone_lr_scale: float = 0.1,
    dropout: float = 0.3,
) -> tuple[nn.Module, list[dict]]:
    """
    EfficientNetV2-S. Final classifier replaced by:
      Dropout(p) → Linear(1280 → 4)
    Matches torchvision's native EfficientNet head structure.
    """
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, NUM_CLASSES),
    )
    param_groups = _build_param_groups(model, ["classifier"], base_lr, backbone_lr_scale)
    return model, param_groups


# ---------------------------------------------------------------------------
# ViT-B/16
# ---------------------------------------------------------------------------

def build_vit(
    base_lr: float = 3e-4,
    backbone_lr_scale: float = 0.1,
    dropout: float = 0.3,
) -> tuple[nn.Module, list[dict]]:
    """
    ViT-B/16 at 224×224 (ImageNet1K_V1 weights).
    Head (model.heads.head) replaced by:
      Dropout(p) → Linear(768 → 4)
    """
    model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    in_features = model.heads.head.in_features  # 768
    model.heads.head = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, NUM_CLASSES),
    )
    param_groups = _build_param_groups(model, ["heads"], base_lr, backbone_lr_scale)
    return model, param_groups


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

ARCH_REGISTRY = {
    "resnet50":       build_resnet50,
    "efficientnetv2": build_efficientnetv2,
    "vit":            build_vit,
}


def get_model(
    arch: str,
    base_lr: float = 3e-4,
    backbone_lr_scale: float = 0.1,
    dropout: float = 0.3,
) -> tuple[nn.Module, list[dict]]:
    """
    Returns (model, param_groups).

    Args:
        arch             : one of 'resnet50', 'efficientnetv2', 'vit'
        base_lr          : learning rate for the classification head
        backbone_lr_scale: backbone LR multiplier (head LR × scale)
        dropout          : dropout probability in classification head
    """
    if arch not in ARCH_REGISTRY:
        raise ValueError(f"Unknown arch '{arch}'. Choose from: {list(ARCH_REGISTRY)}")
    return ARCH_REGISTRY[arch](base_lr=base_lr, backbone_lr_scale=backbone_lr_scale, dropout=dropout)


def count_params(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


if __name__ == "__main__":
    for arch in ARCH_REGISTRY:
        m, pgs = get_model(arch)
        stats = count_params(m)
        print(f"{arch:20s}  total={stats['total']:,}  trainable={stats['trainable']:,}")
