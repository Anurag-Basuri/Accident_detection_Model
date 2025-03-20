import os
import shutil
import random

# Base directory
BASE_DIR = "extracted-frames"

# Define datasets
DATASETS = {
    "dataset-2": {
        "accident": "accident",
        "non-accident": "non-accident"
    },
    "dataset-3": {
        "accident": [
            "collision_with_motorcycle", "collision_with_stationary_object",
            "drifting_or_skidding", "fire_or_explosions", "head_on_collision",
            "objects_falling", "other_crash", "pedestrian_hit",
            "rear_collision", "rollover", "side_collision"
        ],
        "non-accident": ["negative_samples"]  # Move negative samples here
    }
}

# Split ratio: 70% train, 15% val, 15% test
SPLIT_RATIO = [0.7, 0.15, 0.15]
SPLITS = ["train", "val", "test"]

def split_data(dataset_name, categories):
    dataset_path = os.path.join(BASE_DIR, dataset_name)
    
    for label, category_folders in categories.items():
        if isinstance(category_folders, str):
            category_folders = [category_folders]  # Convert to list if single folder
        
        all_videos = []
        for category in category_folders:
            category_path = os.path.join(dataset_path, category)
            if os.path.exists(category_path):
                videos = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]
                all_videos.extend([(category, v) for v in videos])

        # Shuffle data before splitting
        random.shuffle(all_videos)

        # Calculate split sizes
        train_split = int(SPLIT_RATIO[0] * len(all_videos))
        val_split = train_split + int(SPLIT_RATIO[1] * len(all_videos))

        # Assign videos to sets
        data_splits = {
            "train": all_videos[:train_split],
            "val": all_videos[train_split:val_split],
            "test": all_videos[val_split:]
        }

        # Move data
        for split in SPLITS:
            for category, video_folder in data_splits[split]:
                src = os.path.join(dataset_path, category, video_folder)
                dst = os.path.join(dataset_path, split, label, video_folder)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)
        
        print(f"✅ {dataset_name}: Moved {len(all_videos)} videos to train/val/test")

# Run split for each dataset
for dataset, categories in DATASETS.items():
    split_data(dataset, categories)

print("🎯 Dataset successfully split into train, val, and test!")
