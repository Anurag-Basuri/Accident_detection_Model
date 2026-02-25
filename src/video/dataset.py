import os
import cv2
import numpy as np
from typing import List, Tuple

CLASS_NAME_MAP = {
    "non accident": "Non_Accident",
    "non-accident": "Non_Accident",
    "non_accident": "Non_Accident",
}


class VideoDataset:
    def __init__(self, root_dir: str, classes: List[str], frame_count: int = 16, frame_size: Tuple[int, int] = (112, 112)):
        self.root_dir = root_dir
        self.classes = classes
        self.frame_count = frame_count
        self.frame_size = frame_size
        self.samples = []
        for i, c in enumerate(classes):
            # normalize possible variants
            c_norm = CLASS_NAME_MAP.get(c.lower(), c)
            class_dir = os.path.join(root_dir, c_norm)
            if not os.path.isdir(class_dir):
                continue
            for name in os.listdir(class_dir):
                if name.lower().endswith((".mp4", ".avi", ".mov")):
                    self.samples.append((os.path.join(class_dir, name), i))

    def __len__(self):
        return len(self.samples)

    def _sample_frames(self, path: str) -> np.ndarray:
        cap = cv2.VideoCapture(path)
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, max(frame_total - 1, 0), num=self.frame_count).astype(int)
        frames = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.frame_size)
            frames.append(frame)
        cap.release()
        if not frames:
            return np.zeros((self.frame_count, self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
        arr = np.stack(frames, axis=0)
        return arr

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        frames = self._sample_frames(path)
        # shape: T, H, W, C -> normalize
        frames = frames.astype(np.float32) / 255.0
        return frames, label, path
