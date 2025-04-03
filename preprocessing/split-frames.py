import os
import shutil
import random

# Paths
SOURCE_DIR = "processed-datasets/all_data"
TARGET_DIR = "processed-datasets"

# Train-Val-Test Split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Ensure output directories exist
for split in ["train", "val", "test"]:
    for label in ["Accident", "Non Accident"]:
        os.makedirs(os.path.join(TARGET_DIR, split, label), exist_ok=True)


def split_dataset():
    """Split dataset into train, val, test"""
    for label in ["Accident", "Non Accident"]:
        images = os.listdir(os.path.join(SOURCE_DIR, label))
        random.shuffle(images)

        train_count = int(len(images) * TRAIN_RATIO)
        val_count = int(len(images) * VAL_RATIO)

        for i, img in enumerate(images):
            src_path = os.path.join(SOURCE_DIR, label, img)

            if i < train_count:
                dest_dir = os.path.join(TARGET_DIR, "train", label)
            elif i < train_count + val_count:
                dest_dir = os.path.join(TARGET_DIR, "val", label)
            else:
                dest_dir = os.path.join(TARGET_DIR, "test", label)

            shutil.move(src_path, os.path.join(dest_dir, img))

    print("✅ Dataset successfully split into Train/Val/Test!")


split_dataset()
