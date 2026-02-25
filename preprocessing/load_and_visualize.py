"""Visualize a processed dataset: label/split distributions, missing frame checks, and random samples.

Usage:
    python preprocessing/load_and_visualize.py --dataset-path processed-datasets
"""

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from glob import glob

try:
    from natsort import natsorted
except ImportError:
    natsorted = sorted


def load_dataset(dataset_path: str):
    """Load dataset info as a list of (image_path, category, split) tuples."""
    dataset_info = []
    for split in ["train", "val", "test"]:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            continue
        for category in os.listdir(split_path):
            category_path = os.path.join(split_path, category)
            if os.path.isdir(category_path):
                images = natsorted(glob(os.path.join(category_path, "*.jpg")))
                for img_path in images:
                    dataset_info.append((img_path, category, split))
    return dataset_info


def check_missing_frames(dataset_path: str):
    """Check for missing frame indices within video groups."""
    missing_data = {}
    for split in ["train", "val", "test"]:
        split_path = os.path.join(dataset_path, split)
        if not os.path.isdir(split_path):
            continue
        for category in os.listdir(split_path):
            category_path = os.path.join(split_path, category)
            if not os.path.isdir(category_path):
                continue
            video_groups = {}
            for img_path in natsorted(glob(os.path.join(category_path, "*.jpg"))):
                basename = os.path.basename(img_path)
                video_name = "_".join(basename.split("_")[:-1])
                try:
                    frame_number = int(basename.split("_")[-1].split(".")[0])
                except ValueError:
                    continue
                video_groups.setdefault(video_name, []).append(frame_number)

            for video, frames in video_groups.items():
                expected = set(range(min(frames), max(frames) + 1))
                missing = sorted(expected - set(frames))
                if missing:
                    missing_data[video] = missing

    if missing_data:
        print("Missing frames found:")
        for video, frames in missing_data.items():
            print(f"  {video}: {len(frames)} missing -> {frames[:10]}{'...' if len(frames) > 10 else ''}")
    else:
        print("No missing frames detected.")


def display_statistics(dataset_info):
    """Print and plot label/split distributions."""
    print(f"Total images: {len(dataset_info)}")
    label_counts = Counter([label for _, label, _ in dataset_info])
    split_counts = Counter([split for _, _, split in dataset_info])

    print("\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")

    print("\nSplit distribution:")
    for split, count in split_counts.items():
        print(f"  {split}: {count}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x=list(label_counts.keys()), y=list(label_counts.values()), palette="coolwarm", ax=axes[0])
    axes[0].set_title("Label Distribution")
    axes[0].set_ylabel("Count")

    sns.barplot(x=list(split_counts.keys()), y=list(split_counts.values()), palette="viridis", ax=axes[1])
    axes[1].set_title("Split Distribution")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.show()


def show_random_images(dataset_info, num_samples: int = 5):
    """Display random sample images."""
    if len(dataset_info) == 0:
        print("No images to display.")
        return
    indices = np.random.choice(len(dataset_info), min(num_samples, len(dataset_info)), replace=False)
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(indices):
        img_path, label, split = dataset_info[idx]
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, len(indices), i + 1)
        plt.imshow(img)
        plt.title(f"{split.upper()} | {label}\n{os.path.basename(img_path)}", fontsize=9)
        plt.axis("off")
    plt.suptitle("Random Image Samples", fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="Visualize a processed dataset")
    ap.add_argument("--dataset-path", default="processed-datasets",
                    help="Root of the processed dataset (default: processed-datasets)")
    ap.add_argument("--samples", type=int, default=5, help="Number of random images to show")
    args = ap.parse_args()

    dataset_info = load_dataset(args.dataset_path)
    display_statistics(dataset_info)
    check_missing_frames(args.dataset_path)
    show_random_images(dataset_info, args.samples)


if __name__ == "__main__":
    main()
