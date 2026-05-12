#!/usr/bin/env python3
"""
map_to_hybchow_format.py

Normalize gemstone class folder names into the hybchow/gems class naming format.

What it does
------------
- Scans an input dataset directory.
- Detects either:
    1) split folders like train/test/valid containing class folders, or
    2) a single root containing class folders directly.
- Maps source class names to canonical hybchow display names.
- Copies or moves images into the output directory using the canonical class names.
- Writes reports:
    - mapping_report.csv
    - skipped_classes.txt
    - class_mapping.txt

Example
-------
python map_to_hybchow_format.py /path/to/source /path/to/output

python map_to_hybchow_format.py /path/to/source /path/to/output --move
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# hybchow canonical display names
CANONICAL_DISPLAY_NAMES = [
    "Alexandrite",
    "Almandine",
    "Amazonite",
    "Amber",
    "Amethyst",
    "Ametrine",
    "Andradite",
    "Aquamarine",
    "Aventurine Green",
    "Aventurine Yellow",
    "Benitoite",
    "Beryl Golden",
    "Bixbite",
    "Bloodstone",
    "Blue Lace Agate",
    "Carnelian",
    "Chalcedony",
    "Chalcedony Blue",
    "Chrome Diopside",
    "Chrysoberyl",
    "Chrysocolla",
    "Chrysoprase",
    "Citrine",
    "Coral",
    "Diamond",
    "Diaspore",
    "Dumortierite",
    "Emerald",
    "Fluorite",
    "Hessonite",
    "Iolite",
    "Jasper",
    "Kunzite",
    "Kyanite",
    "Lapis Lazuli",
    "Malachite",
    "Onyx Black",
    "Onyx Green",
    "Onyx Red",
    "Peridot",
    "Prehnite",
    "Pyrite",
    "Pyrope",
    "Quartz Beer",
    "Quartz Lemon",
    "Quartz Rutilated",
    "Quartz Smoky",
    "Rhodochrosite",
    "Rhodolite",
    "Rhodonite",
    "Ruby",
    "Sapphire Blue",
    "Sapphire Pink",
    "Sapphire Purple",
    "Sapphire Yellow",
    "Serpentine",
    "Sodalite",
    "Spessartite",
    "Sphene",
    "Sunstone",
    "Tanzanite",
    "Tigers Eye",
    "Topaz",
    "Tourmaline",
    "Tsavorite",
    "Turquoise",
    "Zircon",
    "Zoisite",
]

# Classes seen in your other list that are NOT part of hybchow target set.
# They are intentionally skipped unless you manually remap them.
EXPLICIT_SKIP = {
    "andalusite",
    "aug",
    "cats_eye",
    "danburite",
    "garnet_red",
    "goshenite",
    "grossular",
    "hiddenite",
    "jade",
    "labradorite",
    "larimar",
    "moonstone",
    "morganite",
    "opal",
    "quartz_rose",
    "scapolite",
    "spinel",
    "spodumene",
    "variscite",
}

# Alias fixes for same-class naming differences
ALIASES = {
    "tiger_eye": "tigers_eye",
    "tigers_eye": "tigers_eye",
    "tigerseye": "tigers_eye",
    "tiger_s_eye": "tigers_eye",
    "spessartine": "spessartite",
    "blue_lace_agate": "blue_lace_agate",
    "aventurine_green": "aventurine_green",
    "aventurine_yellow": "aventurine_yellow",
    "chalcedony_blue": "chalcedony_blue",
    "chrome_diopside": "chrome_diopside",
    "lapis_lazuli": "lapis_lazuli",
    "onyx_black": "onyx_black",
    "onyx_green": "onyx_green",
    "onyx_red": "onyx_red",
    "quartz_beer": "quartz_beer",
    "quartz_lemon": "quartz_lemon",
    "quartz_rutilated": "quartz_rutilated",
    "quartz_smoky": "quartz_smoky",
    "sapphire_blue": "sapphire_blue",
    "sapphire_pink": "sapphire_pink",
    "sapphire_purple": "sapphire_purple",
    "sapphire_yellow": "sapphire_yellow",
}

SPLIT_NAMES = ("train", "test", "valid", "val")


def normalize_key(name: str) -> str:
    s = name.strip().lower()
    s = s.replace("&", "and")
    s = s.replace("’", "'")
    s = s.replace("'", "")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


CANONICAL_KEY_TO_DISPLAY: Dict[str, str] = {
    normalize_key(name): name for name in CANONICAL_DISPLAY_NAMES
}
CANONICAL_KEYS = set(CANONICAL_KEY_TO_DISPLAY.keys())


def resolve_class_name(raw_name: str) -> Optional[str]:
    key = normalize_key(raw_name)
    key = ALIASES.get(key, key)

    if key in EXPLICIT_SKIP:
        return None

    if key in CANONICAL_KEYS:
        return CANONICAL_KEY_TO_DISPLAY[key]

    return None


def iter_images(folder: Path) -> Iterable[Path]:
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            yield p


def unique_dest_path(dest: Path) -> Path:
    if not dest.exists():
        return dest

    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    i = 2
    while True:
        candidate = parent / f"{stem}__dup{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def detect_layout(input_root: Path) -> Dict[str, Path]:
    split_dirs = {
        child.name.lower(): child
        for child in input_root.iterdir()
        if child.is_dir() and child.name.lower() in SPLIT_NAMES
    }

    if split_dirs:
        normalized = {}
        for name, path in split_dirs.items():
            normalized["valid" if name == "val" else name] = path
        return normalized

    return {"root": input_root}


def process_class_dir(
    class_dir: Path,
    split_name: str,
    output_root: Path,
    do_move: bool,
    mapping_rows: list[dict],
    skipped_rows: list[dict],
) -> tuple[int, int]:
    raw_name = class_dir.name
    target_name = resolve_class_name(raw_name)

    if target_name is None:
        skipped_rows.append({
            "split": split_name,
            "source_class": raw_name,
            "normalized_source_class": normalize_key(raw_name),
            "reason": "not_in_hybchow_target_set",
        })
        return 0, 0

    if split_name == "root":
        target_dir = output_root / target_name
    else:
        target_dir = output_root / split_name / target_name

    target_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img in iter_images(class_dir):
        dest = unique_dest_path(target_dir / img.name)
        if do_move:
            shutil.move(str(img), str(dest))
        else:
            shutil.copy2(str(img), str(dest))
        copied += 1

    mapping_rows.append({
        "split": split_name,
        "source_class": raw_name,
        "normalized_source_class": normalize_key(raw_name),
        "target_class": target_name,
        "images_copied": copied,
    })
    return copied, 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Map gemstone class folders into hybchow canonical folder names."
    )
    parser.add_argument("input_root", type=Path, help="Input dataset root.")
    parser.add_argument("output_root", type=Path, help="Output dataset root.")
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them.",
    )
    args = parser.parse_args()

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not input_root.exists() or not input_root.is_dir():
        print(f"Error: input directory does not exist or is not a directory: {input_root}", file=sys.stderr)
        return 1

    output_root.mkdir(parents=True, exist_ok=True)

    layout = detect_layout(input_root)
    mapping_rows: list[dict] = []
    skipped_rows: list[dict] = []

    total_images = 0
    total_class_dirs = 0

    for split_name, split_path in layout.items():
        class_dirs = [p for p in split_path.iterdir() if p.is_dir()]
        for class_dir in sorted(class_dirs, key=lambda p: p.name.lower()):
            total_class_dirs += 1
            copied, _ = process_class_dir(
                class_dir=class_dir,
                split_name=split_name,
                output_root=output_root,
                do_move=args.move,
                mapping_rows=mapping_rows,
                skipped_rows=skipped_rows,
            )
            total_images += copied

    # Write reports
    report_csv = output_root / "mapping_report.csv"
    with report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "source_class",
                "normalized_source_class",
                "target_class",
                "images_copied",
            ],
        )
        writer.writeheader()
        writer.writerows(mapping_rows)

    skipped_txt = output_root / "skipped_classes.txt"
    with skipped_txt.open("w", encoding="utf-8") as f:
        for row in skipped_rows:
            f.write(
                f"{row['split']}: {row['source_class']} "
                f"({row['normalized_source_class']}) -> {row['reason']}\n"
            )

    class_mapping_txt = output_root / "class_mapping.txt"
    with class_mapping_txt.open("w", encoding="utf-8") as f:
        for row in sorted(mapping_rows, key=lambda r: (r["split"], r["source_class"].lower())):
            f.write(f"{row['source_class']} -> {row['target_class']}\n")

    print(f"Done.")
    print(f"Scanned class folders: {total_class_dirs}")
    print(f"Copied/moved images: {total_images}")
    print(f"Output: {output_root}")
    print(f"Mapping report: {report_csv}")
    print(f"Skipped classes: {skipped_txt}")
    print(f"Class mapping: {class_mapping_txt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
