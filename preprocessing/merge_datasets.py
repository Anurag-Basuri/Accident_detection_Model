"""Merge a source dataset into a target dataset directory.

Usage:
    python preprocessing/merge_datasets.py --source video-datasets/dataset-1 --target processed-datasets
"""

import os
import shutil
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

logging.basicConfig(
    filename="merge_datasets.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def copy_file(source_file: str, target_file: str, move: bool = False):
    """Copy or move a file, renaming to avoid overwrites."""
    if os.path.exists(target_file):
        base_name, ext = os.path.splitext(target_file)
        counter = 1
        while os.path.exists(target_file):
            target_file = f"{base_name}_{counter}{ext}"
            counter += 1
    try:
        if move:
            shutil.move(source_file, target_file)
        else:
            shutil.copy2(source_file, target_file)
        logging.info(f"Copied {source_file} -> {target_file}")
    except Exception as e:
        logging.error(f"Error: {source_file} -> {target_file}: {e}")


def process_files(source_label_path: str, target_label_path: str,
                  label: str, split: str, move: bool = False):
    """Process files in a single split/label directory in parallel."""
    os.makedirs(target_label_path, exist_ok=True)
    if not os.path.exists(source_label_path):
        return

    files = os.listdir(source_label_path)
    if not files:
        logging.warning(f"No files found in {source_label_path}")
        return

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(
            executor.map(
                lambda f: copy_file(
                    os.path.join(source_label_path, f),
                    os.path.join(target_label_path, f),
                    move,
                ),
                files,
            ),
            total=len(files),
            desc=f"Merging {split}/{label}",
        ))


def merge_datasets(source_path: str, target_path: str, move: bool = False):
    """Merge a source dataset into the target dataset."""
    for split in ["train", "val", "test"]:
        for label in ["Accident", "Non_Accident"]:
            source_label = os.path.join(source_path, split, label)
            target_label = os.path.join(target_path, split, label)
            process_files(source_label, target_label, label, split, move)

    print("Merging complete!")
    logging.info("Merging complete.")


def main():
    ap = argparse.ArgumentParser(description="Merge a source dataset into a target dataset")
    ap.add_argument("--source", default="video-datasets/dataset-1",
                    help="Source dataset path (default: video-datasets/dataset-1)")
    ap.add_argument("--target", default="processed-datasets",
                    help="Target dataset path (default: processed-datasets)")
    ap.add_argument("--move", action="store_true",
                    help="Move files instead of copying")
    args = ap.parse_args()

    merge_datasets(args.source, args.target, args.move)


if __name__ == "__main__":
    main()
