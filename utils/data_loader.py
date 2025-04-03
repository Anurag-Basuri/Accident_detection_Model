import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence
import random

class AccidentDataLoader(Sequence):
    def __init__(self, dataset_path, batch_size=32, img_size=(224, 224), sequence_length=10, shuffle=True):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.sequence_length = sequence_length
        self.shuffle = shuffle

        # Load data
        self.samples = self._load_data()
        if self.shuffle:
            random.shuffle(self.samples)

    def _load_data(self):
        """Load video sequences and labels"""
        samples = []
        for label in ["Accident", "Non Accident"]:
            class_path = os.path.join(self.dataset_path, label)
            for video_folder in os.listdir(class_path):
                video_path = os.path.join(class_path, video_folder)
                frames = sorted(os.listdir(video_path))[:self.sequence_length]  # Trim/pad
                if len(frames) < self.sequence_length:
                    frames += [frames[-1]] * (self.sequence_length - len(frames))  # Padding

                frame_paths = [os.path.join(video_path, f) for f in frames]
                samples.append((frame_paths, 1 if label == "Accident" else 0))
        
        return samples

    def __len__(self):
        return len(self.samples) // self.batch_size

    def __getitem__(self, index):
        batch_samples = self.samples[index * self.batch_size: (index + 1) * self.batch_size]
        X, y = self._load_batch(batch_samples)
        return np.array(X), np.array(y)

    def _load_batch(self, batch_samples):
        X, y = [], []
        for frame_paths, label in batch_samples:
            frames = [img_to_array(load_img(fp, target_size=self.img_size)) / 255.0 for fp in frame_paths]
            X.append(frames)
            y.append(label)
        return X, y
