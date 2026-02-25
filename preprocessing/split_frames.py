"""Split extracted frames from all_data/ into train/val/test directories.

Usage:
    python preprocessing/split_frames.py --source processed-datasets/all_data --target processed-datasets
    python preprocessing/split_frames.py --source processed-datasets/all_data --target processed-datasets --train-ratio 0.8 --val-ratio 0.1
"""

import os
import shutil
import random
import argparse


def split_dataset(source_dir: str, target_dir: str,
                  train_ratio: float = 0.7, val_ratio: float = 0.2):
    """Split dataset from source into train/val/test under target."""
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0, f"train_ratio + val_ratio must be < 1.0 (got {train_ratio + val_ratio})"

    for label in ["Accident", "Non_Accident"]:
        label_dir = os.path.join(source_dir, label)
        if not os.path.isdir(label_dir):
            print(f"Skipping {label_dir} (not found)")
            continue

        images = os.listdir(label_dir)
        random.shuffle(images)

        train_count = int(len(images) * train_ratio)
        val_count = int(len(images) * val_ratio)

        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(target_dir, split, label), exist_ok=True)

        for i, img in enumerate(images):
            src_path = os.path.join(label_dir, img)
            if i < train_count:
                dest_dir = os.path.join(target_dir, "train", label)
            elif i < train_count + val_count:
                dest_dir = os.path.join(target_dir, "val", label)
            else:
                dest_dir = os.path.join(target_dir, "test", label)
            shutil.move(src_path, os.path.join(dest_dir, img))

        print(f"{label}: {train_count} train / {val_count} val / {len(images) - train_count - val_count} test")

    print("Dataset split complete!")


def main():
    ap = argparse.ArgumentParser(description="Split extracted frames into train/val/test")
    ap.add_argument("--source", default="processed-datasets/all_data",
                    help="Source directory with Accident/ and Non_Accident/ folders")
    ap.add_argument("--target", default="processed-datasets",
                    help="Target directory to create train/val/test under")
    ap.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio (default: 0.7)")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Val split ratio (default: 0.2)")
    args = ap.parse_args()

    split_dataset(args.source, args.target, args.train_ratio, args.val_ratio)


if __name__ == "__main__":
    main()
