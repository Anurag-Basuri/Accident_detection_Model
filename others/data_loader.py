import os
import re
import cv2
import numpy as np
from glob import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from sklearn.utils import shuffle

# ========================
# CONFIGURATION
# ========================
IMAGE_SIZE = (224, 224)
SEQUENCE_LENGTH = 30
STEP_SIZE = 5  # You can tweak this. Lower means more overlapping sequences.

class AccidentDataLoader(Sequence):
    """
    Custom Data Generator for loading video sequences from a flat directory.
    Assumes frames are named like: 000001_10.jpg (video_id_frameIndex.jpg).
    """

    def __init__(self, directory, batch_size=8, shuffle=True, augment=False):
        self.directory = directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.class_names = ["Non Accident", "Accident"]
        print(f"[DEBUG] Initializing AccidentDataLoader with directory: {directory}")
        self.sequence_data = self._build_sequence_data()
        self.indexes = np.arange(len(self.sequence_data))
        print(f"[DEBUG] Total samples loaded: {len(self.sequence_data)}")

        if self.augment:
            self.augmenter = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                fill_mode="nearest"
            )

        if self.shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        print("[DEBUG] Shuffling data...")
        self.sequence_data = shuffle(self.sequence_data, random_state=42)
        self.indexes = np.arange(len(self.sequence_data))

    def _build_sequence_data(self):
        sequence_data = []
        print("[DEBUG] Building sequence data...")

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.directory, class_name)

            if not os.path.exists(class_dir):
                print(f"⚠️ Directory not found: {class_dir}")
                continue

            all_frames = glob(os.path.join(class_dir, "*.jpg"))
            print(f"[DEBUG] Found {len(all_frames)} frames in {class_dir}")

            grouped = {}
            for path in all_frames:
                basename = os.path.basename(path)
                match = re.match(r"(\d+)_\d+\.jpg", basename)
                if match:
                    video_id = match.group(1)
                    grouped.setdefault(video_id, []).append(path)

            for video_id, frames in grouped.items():
                sorted_frames = sorted(
                    frames, key=lambda x: int(re.search(r"_(\d+)\.jpg", x).group(1))
                )
                # Add all, even if shorter than SEQUENCE_LENGTH
                sequence_data.append({
                    "frames": sorted_frames,
                    "label": class_idx
                })

        print(f"[DEBUG] Total sequences built: {len(sequence_data)}")
        return sequence_data

    def __len__(self):
        return (len(self.sequence_data) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        batch_data = self.sequence_data[index * self.batch_size:(index + 1) * self.batch_size]
        print(f"[DEBUG] Loading batch {index + 1}/{len(self)} with {len(batch_data)} samples")

        X = np.zeros((len(batch_data), SEQUENCE_LENGTH, *IMAGE_SIZE, 3), dtype=np.float32)
        y = np.zeros((len(batch_data), 1), dtype=np.float32)

        for i, item in enumerate(batch_data):
            frames = []
            for frame_path in item["frames"]:
                frame = cv2.imread(frame_path)
                if frame is None:
                    print(f"⚠️ Skipping corrupted frame: {frame_path}")
                    frame = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)

                frame = cv2.resize(frame, IMAGE_SIZE)
                frame = frame / 255.0

                if self.augment:
                    frame = self.augmenter.random_transform(frame)

                frames.append(frame)

            X[i] = self._pad_or_trim(frames)
            y[i] = item["label"]

        print(f"[DEBUG] Batch loaded with shape: X={X.shape}, y={y.shape}")
        return X, y

    def _pad_or_trim(self, frames):
        if len(frames) < SEQUENCE_LENGTH:
            last = frames[-1] if frames else np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
            while len(frames) < SEQUENCE_LENGTH:
                frames.append(last)
        return np.array(frames[:SEQUENCE_LENGTH])

    def on_epoch_end(self):
        if self.shuffle:
            self._shuffle_data()
