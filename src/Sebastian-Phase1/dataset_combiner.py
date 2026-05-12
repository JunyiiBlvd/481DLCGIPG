from pathlib import Path
import shutil
import hashlib

SPLITS = ["train", "valid", "test"]

def file_hash(path, chunk_size=8192):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()

def merge_datasets(dataset_a, dataset_b, output_dir):
    dataset_a = Path(dataset_a)
    dataset_b = Path(dataset_b)
    output_dir = Path(output_dir)

    seen_hashes = set()

    for split in SPLITS:
        for source_root in [dataset_a / split, dataset_b / split]:
            if not source_root.exists():
                continue

            for class_dir in source_root.iterdir():
                if not class_dir.is_dir():
                    continue

                dest_class_dir = output_dir / split / class_dir.name
                dest_class_dir.mkdir(parents=True, exist_ok=True)

                for img_path in class_dir.iterdir():
                    if not img_path.is_file():
                        continue

                    try:
                        h = file_hash(img_path)
                    except Exception as e:
                        print(f"Skipping {img_path}: {e}")
                        continue

                    if h in seen_hashes:
                        print(f"Duplicate skipped: {img_path}")
                        continue

                    seen_hashes.add(h)

                    dest_path = dest_class_dir / img_path.name

                    if dest_path.exists():
                        stem = img_path.stem
                        suffix = img_path.suffix
                        i = 1
                        while True:
                            new_name = f"{stem}_{i}{suffix}"
                            new_dest = dest_class_dir / new_name
                            if not new_dest.exists():
                                dest_path = new_dest
                                break
                            i += 1

                    shutil.copy2(img_path, dest_path)
                    print(f"Copied: {img_path} -> {dest_path}")

if __name__ == "__main__":
    merge_datasets(
        "./Hybchow-Presorted",
        "./Kamath-Converted",
        "./Combined-P1-Dataset"
    )
