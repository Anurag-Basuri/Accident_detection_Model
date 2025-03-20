import os
import shutil
import random

# Set seed for reproducibility
random.seed(42)

# Define paths
BASE_DIR = "extracted-frames"
DATASET_2_DIR = os.path.join(BASE_DIR, "dataset-2")
DATASET_3_DIR = os.path.join(BASE_DIR, "dataset-3")

OUTPUT_DIR = "processed-datasets"

# Train-Val-Test Split Ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def split_data(dataset_path, accident_folders, non_accident_folders, output_path):
    """
    Splits and organizes frames from extracted video datasets into train, val, and test.
    """
    print(f"\nProcessing dataset: {dataset_path}")

    # Ensure output directories exist
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_path, split, "Accident"), exist_ok=True)
        os.makedirs(os.path.join(output_path, split, "Non Accident"), exist_ok=True)

    # Function to move data while keeping full videos together
    def move_files(category, folder_list, split):
        for folder in folder_list:
            folder_path = os.path.join(dataset_path, folder)
            if not os.path.exists(folder_path):
                continue  # Skip if folder doesn't exist

            videos = sorted(os.listdir(folder_path))  # Ensure order is maintained
            random.shuffle(videos)  # Shuffle for randomness

            # Determine split sizes
            total_videos = len(videos)
            train_cutoff = int(total_videos * TRAIN_RATIO)
            val_cutoff = train_cutoff + int(total_videos * VAL_RATIO)

            # Assign videos to splits
            for i, video in enumerate(videos):
                if i < train_cutoff:
                    target_split = "train"
                elif i < val_cutoff:
                    target_split = "val"
                else:
                    target_split = "test"

                src_path = os.path.join(folder_path, video)
                dest_path = os.path.join(output_path, target_split, category, video)

                shutil.move(src_path, dest_path)

    # Move files for both Accident and Non-Accident categories
    move_files("Accident", accident_folders, output_path)
    move_files("Non Accident", non_accident_folders, output_path)

# Processing Dataset-2
accident_folders_2 = ["Accident"]
non_accident_folders_2 = ["Non-Accident"]
split_data(DATASET_2_DIR, accident_folders_2, non_accident_folders_2, OUTPUT_DIR)

# Processing Dataset-3
accident_folders_3 = [
    "collision_with_motorcycle",
    "collision_with_stationary_object",
    "drifting_or_skidding",
    "fire_or_explosions",
    "head on collision",
    "objects_falling",
    "other_crash",
    "pedestrian_hit",
    "rear_collision",
    "rollover",
    "side_collision"
]
non_accident_folders_3 = ["negative_samples"]
split_data(DATASET_3_DIR, accident_folders_3, non_accident_folders_3, OUTPUT_DIR)

print("\n✅ Dataset-2 and Dataset-3 successfully processed and split into train, val, and test sets.")
