import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from src.common.config import get_paths
from src.video.dataset import VideoDataset
from src.video.model import VideoClassifier


def collate(batch):
    xs, ys, _ = zip(*batch)
    xs = torch.tensor(np.stack(xs, axis=0), dtype=torch.float32)
    ys = torch.tensor(ys, dtype=torch.long)
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--frame-count", type=int, default=16)
    ap.add_argument("--frame-size", type=int, nargs=2, default=(112, 112))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    paths = get_paths()
    data_root = args.data_dir or os.path.join(paths["data_root"], paths.get("video_data_dir", "datasets/video-datasets/dataset-1"))
    classes = ["Accident", "Non_Accident"]

    ds = VideoDataset(os.path.join(data_root, args.split), classes, args.frame_count, tuple(args.frame_size))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model or os.path.join(paths["models_root"], "video_model.pth")
    model = VideoClassifier(num_classes=len(classes)).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(pred)
            y_true.extend(y.numpy())

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    report = classification_report(y_true, y_pred, target_names=classes)
    print("Classification Report:\n", report)


if __name__ == "__main__":
    main()
