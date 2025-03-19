import cv2
import os
from tqdm import tqdm

# Set up paths
VIDEO_DATASET_PATH = "video-datasets"
EXTRACTED_FRAMES_PATH = "extracted-frames"

# Dataset directories to process
datasets = ["dataset-2", "dataset-3"]

# Create output directories if they don't exist
os.makedirs(EXTRACTED_FRAMES_PATH, exist_ok=True)

# Frame extraction function
def extract_frames(video_path, output_folder, video_name):
    """Extracts frames from a video and saves them in the given folder."""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get FPS of the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # Adaptive frame rate extraction
    if duration < 5:  
        frame_interval = 1  # Extract every frame
    elif duration > 60:  
        frame_interval = fps * 2  # Extract 1 frame every 2 seconds
    else:  
        frame_interval = fps  # Extract 1 frame per second

    frame_count = 0
    os.makedirs(output_folder, exist_ok=True)

    with tqdm(total=total_frames, desc=f"Processing {video_name}", unit="frames") as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(output_folder, f"{video_name}_frame{frame_count}.jpg")
                frame = cv2.resize(frame, (224, 224))  # Resize to 224x224
                cv2.imwrite(frame_filename, frame)

            frame_count += 1
            pbar.update(1)

    cap.release()

# Process each dataset
for dataset in datasets:
    dataset_path = os.path.join(VIDEO_DATASET_PATH, dataset)
    output_dataset_path = os.path.join(EXTRACTED_FRAMES_PATH, dataset)

    for category in os.listdir(dataset_path):  # Loop through accident types
        category_path = os.path.join(dataset_path, category)
        output_category_path = os.path.join(output_dataset_path, category)

        os.makedirs(output_category_path, exist_ok=True)

        for video_file in os.listdir(category_path):
            video_path = os.path.join(category_path, video_file)
            video_name, _ = os.path.splitext(video_file)

            extract_frames(video_path, output_category_path, video_name)

print("✅ Frame extraction complete. Frames saved in 'extracted-frames/'")
