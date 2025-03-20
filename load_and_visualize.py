import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from glob import glob
from natsort import natsorted  # Ensure files are loaded in correct order

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

# Load dataset
dataset_info = load_dataset()

# Check total images and labels
print(f"✅ Total Images Loaded: {len(dataset_info)}")
label_counts = Counter([label for _, label, _ in dataset_info])
print("📊 Label Distribution:", label_counts)

# Plot label distribution
plt.figure(figsize=(10, 5))
sns.barplot(x=list(label_counts.keys()), y=list(label_counts.values()), palette="coolwarm")
plt.xlabel("Categories")
plt.ylabel("Number of Images")
plt.title("Label Distribution in Processed Dataset")
plt.xticks(rotation=45)
plt.show()

# Function to display random images
def show_random_images(dataset_info, num_samples=5):
    plt.figure(figsize=(15, 10))
    random_samples = np.random.choice(len(dataset_info), num_samples, replace=False)

    for i, idx in enumerate(random_samples):
        img_path, label, split = dataset_info[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.title(f"{split.upper()} | {label}", fontsize=12)
        plt.axis("off")

    plt.show()

# Check missing frames
check_missing_frames()

# Show random images
show_random_images(dataset_info)
