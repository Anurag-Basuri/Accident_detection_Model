"""Extract frames from video datasets and save as images.

Usage:
    python preprocessing/extract_frames.py --dataset-2 datasets/video-datasets/dataset-2 --dataset-3 datasets/video-datasets/dataset-3 --output processed-datasets
"""

import os
import argparse
import cv2
from tqdm import tqdm


def extract_frames(video_path: str, output_folder: str, label: str,
                   frame_interval: int = 10, image_size: tuple = (224, 224)):
    """Extract frames from a single video and save as images."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    vid_name = os.path.splitext(os.path.basename(video_path))[0]

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if frame_count % frame_interval == 0:
            frame = cv2.resize(frame, image_size)
            save_path = os.path.join(output_folder, label, f"{vid_name}_{frame_count}.jpg")
            cv2.imwrite(save_path, frame)
        frame_count += 1
    cap.release()


def process_dataset(dataset_path: str, dataset_type: str, output_dir: str,
                    frame_interval: int = 10, image_size: tuple = (224, 224)):
    """Process all videos in a dataset folder and extract frames."""
    all_data_dir = os.path.join(output_dir, "all_data")

    for folder in tqdm(os.listdir(dataset_path), desc=f"Processing {dataset_type}"):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue

        # Map folder names to labels based on dataset convention
        if dataset_type == "dataset-3":
            label = "Non_Accident" if folder == "negative_samples" else "Accident"
        else:
            label = "Accident" if folder == "accident" else "Non_Accident"

        os.makedirs(os.path.join(all_data_dir, label), exist_ok=True)

        for video_file in os.listdir(folder_path):
            if video_file.lower().endswith((".mp4", ".avi", ".mov")):
                video_path = os.path.join(folder_path, video_file)
                extract_frames(video_path, all_data_dir, label, frame_interval, image_size)


def main():
    ap = argparse.ArgumentParser(description="Extract frames from video datasets")
    ap.add_argument("--dataset-2", default="datasets/video-datasets/dataset-2",
                    help="Path to dataset-2 (default: datasets/video-datasets/dataset-2)")
    ap.add_argument("--dataset-3", default="datasets/video-datasets/dataset-3",
                    help="Path to dataset-3 (default: datasets/video-datasets/dataset-3)")
    ap.add_argument("--output", default="processed-datasets",
                    help="Output directory (default: processed-datasets)")
    ap.add_argument("--frame-interval", type=int, default=10,
                    help="Extract every Nth frame (default: 10)")
    ap.add_argument("--image-size", type=int, nargs=2, default=[224, 224],
                    help="Resize frames to this size (default: 224 224)")
    args = ap.parse_args()

    image_size = tuple(args.image_size)

    if os.path.isdir(args.dataset_2):
        print(f"Extracting frames from dataset-2: {args.dataset_2}")
        process_dataset(args.dataset_2, "dataset-2", args.output, args.frame_interval, image_size)

    if os.path.isdir(args.dataset_3):
        print(f"Extracting frames from dataset-3: {args.dataset_3}")
        process_dataset(args.dataset_3, "dataset-3", args.output, args.frame_interval, image_size)

    print("Frame extraction complete!")


if __name__ == "__main__":
    main()
