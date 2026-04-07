"""
diamond_dataset.py — PyTorch Dataset for diamond image classification.

Usage:
    from diamond_dataset import DiamondDataset, get_dataloader, train_transform, val_test_transform

Image path convention:
    image_dir / value_tier / {diamond_id}.jpg
"""

import json
import logging
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

log = logging.getLogger(__name__)

# ── ImageNet normalisation constants ─────────────────────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Transform pipelines ───────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


# ── Dataset ───────────────────────────────────────────────────────────────────
class DiamondDataset(Dataset):
    """
    Dataset for diamond image classification.

    Args:
        csv_path:   Path to a split CSV (must contain diamond_id, value_tier, tier_label).
        image_dir:  Root image directory. Images are at image_dir/value_tier/{diamond_id}.jpg
        transform:  torchvision transform pipeline. Defaults to val_test_transform.
    """

    def __init__(self, csv_path: str | Path, image_dir: str | Path, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform if transform is not None else val_test_transform

        df = pd.read_csv(csv_path, usecols=["diamond_id", "value_tier", "tier_label"],
                         low_memory=False)

        # Resolve image paths and filter out missing files at init time
        # so __getitem__ never needs to handle absent files.
        df["_img_path"] = df.apply(
            lambda r: self.image_dir / str(r["value_tier"]) / f"{r['diamond_id']}.jpg",
            axis=1,
        )
        missing_mask = ~df["_img_path"].apply(lambda p: p.exists())
        n_missing = missing_mask.sum()
        if n_missing:
            log.warning(
                "%d image(s) missing from disk and will be skipped. "
                "First 5: %s",
                n_missing,
                df.loc[missing_mask, "diamond_id"].head(5).tolist(),
            )
            df = df[~missing_mask].reset_index(drop=True)

        self.records = df[["diamond_id", "tier_label", "_img_path"]].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row   = self.records.iloc[idx]
        label = int(row["tier_label"])

        try:
            image = Image.open(row["_img_path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (128, 128, 128))
        if self.transform:
            image = self.transform(image)

        return image, label

    @property
    def labels(self) -> list[int]:
        """All tier labels in dataset order — used by WeightedRandomSampler."""
        return self.records["tier_label"].tolist()


# ── DataLoader helper ─────────────────────────────────────────────────────────
def get_dataloader(
    csv_path: str | Path,
    image_dir: str | Path,
    split: str,
    batch_size: int,
    class_weights_path: str | Path | None = None,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Build a DataLoader for a given split.

    Args:
        csv_path:            Path to the split CSV.
        image_dir:           Root image directory.
        split:               One of "train", "val", "test".
        batch_size:          Batch size.
        class_weights_path:  Path to class_weights.json.
                             Required (and used) only when split == "train".
        num_workers:         DataLoader worker processes.
        pin_memory:          Pin CPU memory for faster GPU transfer.

    Returns:
        DataLoader with WeightedRandomSampler for train, sequential sampler otherwise.
    """
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

    tfm     = train_transform if split == "train" else val_test_transform
    dataset = DiamondDataset(csv_path, image_dir, transform=tfm)

    sampler = None
    shuffle = False

    if split == "train":
        shuffle = False  # WeightedRandomSampler handles ordering
        if class_weights_path is not None:
            weights_dict = json.loads(Path(class_weights_path).read_text())
            # Key in JSON is the split CSV stem (e.g. "ja_natural_train")
            split_key    = Path(csv_path).stem
            class_w      = weights_dict.get(split_key)
            if class_w is None:
                raise KeyError(
                    f"'{split_key}' not found in {class_weights_path}. "
                    f"Available keys: {list(weights_dict.keys())}"
                )
            # Map per-sample weights
            sample_weights = torch.tensor(
                [float(class_w[str(lbl)]) for lbl in dataset.labels],
                dtype=torch.float,
            )
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(dataset),
                replacement=True,
            )
        else:
            shuffle = True  # no weights — plain shuffle

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),  # avoid partial batches only during training
    )


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    SPLITS   = Path("/mnt/storage/projects/DLCGIPG/data/splits")
    JA_IMGS  = Path("/mnt/storage/projects/DLCGIPG/ja_scraper/output/images")
    WEIGHTS  = SPLITS / "class_weights.json"

    csv_path = SPLITS / "ja_natural_train.csv"

    print("── DiamondDataset smoke test ──")
    print(f"CSV:       {csv_path}")
    print(f"Image dir: {JA_IMGS}")

    dataset = DiamondDataset(csv_path, JA_IMGS, transform=train_transform)
    print(f"\nDataset length: {len(dataset):,}")

    img, label = dataset[0]
    print(f"Sample [0] — image shape: {tuple(img.shape)}  label: {label}")
    print(f"  pixel min={img.min():.3f}  max={img.max():.3f}  "
          f"mean={img.mean():.3f}")

    print("\n── DataLoader with WeightedRandomSampler ──")
    loader = get_dataloader(
        csv_path, JA_IMGS,
        split="train",
        batch_size=32,
        class_weights_path=WEIGHTS,
        num_workers=0,   # 0 for smoke test (avoids fork overhead)
        pin_memory=False,
    )
    print(f"Batches per epoch: {len(loader):,}")

    batch_imgs, batch_labels = next(iter(loader))
    print(f"Batch shape:  {tuple(batch_imgs.shape)}")
    print(f"Labels:       {batch_labels.tolist()}")
    print(f"Label counts: { {i: (batch_labels == i).sum().item() for i in range(4)} }")

    print("\n✓ Smoke test passed")
    sys.exit(0)
