import os
import cv2
import random
import shutil
from tqdm import tqdm

# Paths
DATASET_2_PATH = "datasets/video-datasets/dataset-2"
DATASET_3_PATH = "datasets/video-datasets/dataset-3"
OUTPUT_DIR = "processed-datasets"

# Frame extraction settings
FRAME_INTERVAL = 10  # Extract every 10th frame
IMAGE_SIZE = (224, 224)  # Resize frames

# Train-Val-Test Split Ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Ensure output directories exist
for split in ["train", "val", "test"]:
    for label in ["Accident", "Non Accident"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, label), exist_ok=True)


def extract_frames(video_path, output_folder, label):
    """Extract frames from video and save as images"""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    vid_name = os.path.splitext(os.path.basename(video_path))[0]

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        if frame_count % FRAME_INTERVAL == 0:
            frame = cv2.resize(frame, IMAGE_SIZE)
            save_path = os.path.join(output_folder, label, f"{vid_name}_{frame_count}.jpg")
            cv2.imwrite(save_path, frame)

        frame_count += 1
    cap.release()


def process_dataset(dataset_path, dataset_type):
    """Process videos and extract frames"""
    for folder in tqdm(os.listdir(dataset_path), desc=f"Processing {dataset_type}"):
        folder_path = os.path.join(dataset_path, folder)

        if not os.path.isdir(folder_path):
            continue  # Skip non-folder items
        
        # Determine label
        if dataset_type == "dataset-3":
            label = "Non Accident" if folder == "negative_samples" else "Accident"
        else:  # dataset-2
            label = "Accident" if folder == "accident" else "Non Accident"

        output_folder = os.path.join(OUTPUT_DIR, "all_data")

        # Ensure label directories exist
        os.makedirs(os.path.join(output_folder, label), exist_ok=True)

        # Process each video in the folder
        for video_file in os.listdir(folder_path):
            if video_file.endswith((".mp4", ".avi", ".mov")):
                video_path = os.path.join(folder_path, video_file)
                extract_frames(video_path, output_folder, label)


def split_dataset():
    """Split dataset into train, val, test"""
    all_data_path = os.path.join(OUTPUT_DIR, "all_data")
    for label in ["Accident", "Non Accident"]:
        images = os.listdir(os.path.join(all_data_path, label))
        random.shuffle(images)

        train_count = int(len(images) * TRAIN_RATIO)
        val_count = int(len(images) * VAL_RATIO)

        for i, img in enumerate(images):
            src_path = os.path.join(all_data_path, label, img)

            if i < train_count:
                dest_dir = os.path.join(OUTPUT_DIR, "train", label)
            elif i < train_count + val_count:
                dest_dir = os.path.join(OUTPUT_DIR, "val", label)
            else:
                dest_dir = os.path.join(OUTPUT_DIR, "test", label)

            shutil.move(src_path, os.path.join(dest_dir, img))


# Process datasets
print("🚀 Extracting frames from Dataset-2...")
process_dataset(DATASET_2_PATH, "dataset-2")
print("🚀 Extracting frames from Dataset-3...")
process_dataset(DATASET_3_PATH, "dataset-3")

print("✅ Frame extraction complete! Organizing into Train/Val/Test splits...")
split_dataset()
print("✅ Dataset processing complete! Ready for training.")