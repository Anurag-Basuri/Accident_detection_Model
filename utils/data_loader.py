import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
import random

# ========================
# CONFIGURATION
# ========================
IMAGE_SIZE = (224, 224)
SEQUENCE_LENGTH = 30  # Number of frames per video sequence
BATCH_SIZE = 16
DATA_AUGMENTATION = True  # Enable for training

# ========================
# VIDEO SEQUENCE LOADER
# ========================
class AccidentDataLoader(Sequence):
    """
    Custom Data Generator for loading video sequences.
    """

    def __init__(self, directory, batch_size=BATCH_SIZE, shuffle=True, augment=False):
        self.directory = directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.class_names = ["Non Accident", "Accident"]
        self.file_paths, self.labels = self._load_data()
        self.indexes = np.arange(len(self.file_paths))

        if self.shuffle:
            np.random.shuffle(self.indexes)

        if self.augment:
            self.augmenter = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                fill_mode="nearest"
            )

    def _load_data(self):
        file_paths = []
        labels = []

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.directory, class_name)

            if not os.path.exists(class_dir):
                print(f"⚠️ Warning: Directory {class_dir} not found. Skipping.")
                continue

            for video_folder in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video_folder)
                file_paths.append(video_path)
                labels.append(class_idx)

        return np.array(file_paths), np.array(labels)

    def __len__(self):
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        """
        Returns a batch of video sequences and their labels.
        """
        batch_indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_file_paths = [self.file_paths[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]

        batch_sequences = [self._load_video_sequence(fp) for fp in batch_file_paths]
        batch_sequences = np.array(batch_sequences)
        batch_labels = np.array(batch_labels)

        return batch_sequences, batch_labels

    def _load_video_sequence(self, video_folder):
        """
        Loads frames from a video sequence, resizes them, and pads if necessary.
        """
        frames = []
        frame_files = sorted(os.listdir(video_folder))  # Sort to maintain correct order

        for frame_file in frame_files:
            frame_path = os.path.join(video_folder, frame_file)
            frame = cv2.imread(frame_path)

            if frame is None:
                print(f"⚠️ Skipping corrupted frame: {frame_path}")
                continue

            frame = cv2.resize(frame, IMAGE_SIZE)
            frame = frame / 255.0  # Normalize
            frames.append(frame)

        if self.augment:
            frames = [self.augmenter.random_transform(frame) for frame in frames]

        return self._pad_sequence(frames)

    def _pad_sequence(self, frames):
        """
        Pads a sequence to the required length using the last frame.
        """
        if len(frames) < SEQUENCE_LENGTH:
            last_frame = frames[-1] if frames else np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
            frames.extend([last_frame] * (SEQUENCE_LENGTH - len(frames)))

        return np.array(frames[:SEQUENCE_LENGTH])

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# ========================
# USAGE EXAMPLE
# ========================
if __name__ == "__main__":
    train_loader = AccidentDataLoader("processed-datasets/train", batch_size=8, shuffle=True, augment=True)
    val_loader = AccidentDataLoader("processed-datasets/val", batch_size=8, shuffle=False, augment=False)

    # Load a sample batch
    X_batch, y_batch = train_loader[0]

    print(f"Sample batch shape: {X_batch.shape}")  # (batch_size, sequence_length, 224, 224, 3)
    print(f"Sample labels: {y_batch}")  # Array of 0s and 1s
