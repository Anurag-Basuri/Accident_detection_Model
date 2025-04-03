import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from glob import glob
from natsort import natsorted  # Ensure files are loaded in correct order
from tqdm import tqdm  # For progress bars

# Set the dataset path
DATASET_PATH = "processed-datasets"

# Function to load dataset in correct order
def load_dataset():
    dataset_info = []
    for split in ["train", "val", "test"]:  # Ensure correct dataset structure
        split_path = os.path.join(DATASET_PATH, split)
        if os.path.exists(split_path):
            for category in os.listdir(split_path):  # Accident, Non-Accident
                category_path = os.path.join(split_path, category)
                if os.path.isdir(category_path):
                    images = natsorted(glob(os.path.join(category_path, "*.jpg")))  # Ensure sequential order
                    for img_path in images:
                        dataset_info.append((img_path, category, split))
    return dataset_info

# Function to check missing frames
def check_missing_frames():
    missing_data = {}
    for split in ["train", "val", "test"]:
        split_path = os.path.join(DATASET_PATH, split)
        for category in os.listdir(split_path):
            category_path = os.path.join(split_path, category)
            if os.path.isdir(category_path):
                video_groups = {}
                for img_path in natsorted(glob(os.path.join(category_path, "*.jpg"))):
                    video_name = "_".join(os.path.basename(img_path).split("_")[:-1])  # Extract video name
                    frame_number = int(os.path.basename(img_path).split("_")[-1].split(".")[0])  # Extract frame number
                    
                    if video_name not in video_groups:
                        video_groups[video_name] = []
                    video_groups[video_name].append(frame_number)

                # Check if frame numbers are sequential
                for video, frames in video_groups.items():
                    expected_frames = list(range(min(frames), max(frames) + 1))
                    if frames != expected_frames:
                        missing_frames = list(set(expected_frames) - set(frames))
                        missing_data[video] = missing_frames

    if missing_data:
        print("⚠ Missing Frames Found:")
        for video, frames in missing_data.items():
            print(f"🚨 {video}: Missing {len(frames)} frames → {frames}")
    else:
        print("✅ No missing frames detected.")

# Function to display dataset statistics
def display_dataset_statistics(dataset_info):
    print(f"✅ Total Images Loaded: {len(dataset_info)}")
    label_counts = Counter([label for _, label, _ in dataset_info])
    split_counts = Counter([split for _, _, split in dataset_info])

    print("\n📊 Label Distribution:")
    for label, count in label_counts.items():
        print(f"   - {label}: {count} images")

    print("\n📊 Split Distribution:")
    for split, count in split_counts.items():
        print(f"   - {split}: {count} images")

    # Plot label distribution
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(label_counts.keys()), y=list(label_counts.values()), palette="coolwarm")
    plt.xlabel("Categories")
    plt.ylabel("Number of Images")
    plt.title("Label Distribution in Processed Dataset")
    plt.xticks(rotation=45)
    plt.show()

    # Plot split distribution
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(split_counts.keys()), y=list(split_counts.values()), palette="viridis")
    plt.xlabel("Splits")
    plt.ylabel("Number of Images")
    plt.title("Split Distribution in Processed Dataset")
    plt.xticks(rotation=45)
    plt.show()

# Function to display random images with metadata
def show_random_images(dataset_info, num_samples=5):
    plt.figure(figsize=(15, 10))
    random_samples = np.random.choice(len(dataset_info), num_samples, replace=False)

    for i, idx in enumerate(random_samples):
        img_path, label, split = dataset_info[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.title(f"{split.upper()} | {label}\n{os.path.basename(img_path)}", fontsize=10)
        plt.axis("off")

    plt.suptitle("Random Image Samples with Metadata", fontsize=16)
    plt.tight_layout()
    plt.show()

# Function to analyze frame counts per video
def analyze_frame_counts(dataset_info):
    frame_counts = {}
    for img_path, label, split in dataset_info:
        video_name = "_".join(os.path.basename(img_path).split("_")[:-1])  # Extract video name
        if video_name not in frame_counts:
            frame_counts[video_name] = 0
        frame_counts[video_name] += 1

    # Plot frame count distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(list(frame_counts.values()), bins=20, kde=True, color="blue")
    plt.xlabel("Number of Frames per Video")
    plt.ylabel("Frequency")
    plt.title("Distribution of Frame Counts per Video")
    plt.show()

    print(f"📊 Total Videos: {len(frame_counts)}")
    print(f"   - Average Frames per Video: {np.mean(list(frame_counts.values())):.2f}")
    print(f"   - Minimum Frames in a Video: {min(frame_counts.values())}")
    print(f"   - Maximum Frames in a Video: {max(frame_counts.values())}")

# Load dataset
dataset_info = load_dataset()

# Display dataset statistics
display_dataset_statistics(dataset_info)

# Check for missing frames
check_missing_frames()

# Analyze frame counts
analyze_frame_counts(dataset_info)

# Show random images
show_random_images(dataset_info)