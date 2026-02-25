import argparse
import numpy as np
import torch
from src.video.model import VideoClassifier
from src.video.dataset import VideoDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to a video file")
    ap.add_argument("--model", required=True, help="Path to model .pth")
    ap.add_argument("--frame-count", type=int, default=16)
    ap.add_argument("--frame-size", type=int, nargs=2, default=(112, 112))
    args = ap.parse_args()

    classes = ["Accident", "Non_Accident"]
    ds = VideoDataset(root_dir="", classes=classes, frame_count=args.frame_count, frame_size=tuple(args.frame_size))
    # Hack: directly use dataset sampler for the single file
    ds.samples = [(args.input, 0)]
    frames, _, _ = ds[0]
    x = torch.tensor(np.expand_dims(frames, axis=0), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoClassifier(num_classes=len(classes)).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(x.to(device))
        prob = torch.softmax(logits, dim=1)[0]
        cls = int(torch.argmax(prob).item())
        score = float(torch.max(prob).item())
    print({"class_index": cls, "score": score})


if __name__ == "__main__":
    main()
