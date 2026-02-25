import os
import numpy as np
import torch
from src.common.config import get_paths, get_model_map
from src.video.dataset import VideoDataset
from src.video.model import VideoClassifier


def predict_video(path: str, frame_count: int = 16, frame_size=(112, 112)) -> dict:
    classes = ["Accident", "Non_Accident"]
    ds = VideoDataset(root_dir="", classes=classes, frame_count=frame_count, frame_size=frame_size)
    ds.samples = [(path, 0)]
    frames, _, _ = ds[0]
    x = torch.tensor(np.expand_dims(frames, axis=0), dtype=torch.float32)

    paths = get_paths()
    model_map = get_model_map(paths["models_root"]) or {}
    model_path = os.path.join(paths["models_root"], model_map.get("video", {}).get("path", "video_model.pth"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoClassifier(num_classes=len(classes)).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(x.to(device))
        prob = torch.softmax(logits, dim=1)[0]
        cls = int(torch.argmax(prob).item())
        score = float(torch.max(prob).item())
    return {"class_index": cls, "score": score}
