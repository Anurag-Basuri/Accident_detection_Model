import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence

# ========================
# CONFIGURATION
# ========================
IMAGE_SIZE = (224, 224)
SEQUENCE_LENGTH = 30

class AccidentDataLoader(Sequence):
    """
    Custom Data Generator for video sequence loading.
    """

    def __init__(self, directory, batch_size=8, shuffle=True, augment=False):
        self.directory = directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.class_names = ["Non Accident", "Accident"]
        self.file_paths, self.labels = self._load_data()
        self.indexes = np.arange(len(self.file_paths))

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
            np.random.shuffle(self.indexes)

    def _load_data(self):
        file_paths = []
        labels = []

        for idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.directory, class_name)

            if not os.path.exists(class_dir):
                print(f"⚠️ Directory not found: {class_dir}")
                continue

            for video_folder in os.listdir(class_dir):
                full_path = os.path.join(class_dir, video_folder)
                if os.path.isdir(full_path):
                    file_paths.append(full_path)
                    labels.append(idx)

        return np.array(file_paths), np.array(labels)

    def __len__(self):
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = [self.file_paths[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]

        batch_sequences = [self._load_video_sequence(path) for path in batch_paths]
        return np.array(batch_sequences), np.array(batch_labels)

    def _load_video_sequence(self, folder_path):
        frames = []
        frame_files = sorted(os.listdir(folder_path))

        for fname in frame_files:
            frame_path = os.path.join(folder_path, fname)
            frame = cv2.imread(frame_path)

            if frame is None:
                print(f"⚠️ Corrupted frame skipped: {frame_path}")
                continue

            frame = cv2.resize(frame, IMAGE_SIZE)
            frame = frame / 255.0  # Normalize

            if self.augment:
                frame = self.augmenter.random_transform(frame)

            frames.append(frame)

        return self._pad_or_trim(frames)

    def _pad_or_trim(self, frames):
        """
        Ensures fixed-length sequences by trimming or padding.
        """
        if len(frames) < SEQUENCE_LENGTH:
            last_frame = frames[-1] if frames else np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
            while len(frames) < SEQUENCE_LENGTH:
                frames.append(last_frame)
        return np.array(frames[:SEQUENCE_LENGTH])

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
