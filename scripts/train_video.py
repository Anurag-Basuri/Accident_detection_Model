import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
    ap.add_argument("--train-subdir", default="train")
    ap.add_argument("--val-subdir", default="val")
    ap.add_argument("--frame-count", type=int, default=16)
    ap.add_argument("--frame-size", type=int, nargs=2, default=(112, 112))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model-out", default=None)
    args = ap.parse_args()

    paths = get_paths()
    data_root = args.data_dir or os.path.join(paths["data_root"], paths.get("video_data_dir", "datasets/video-datasets/dataset-1"))
    classes = ["Accident", "Non_Accident"]

    train_ds = VideoDataset(os.path.join(data_root, args.train_subdir), classes, args.frame_count, tuple(args.frame_size))
    val_ds = VideoDataset(os.path.join(data_root, args.val_subdir), classes, args.frame_count, tuple(args.frame_size))

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoClassifier(num_classes=len(classes)).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
            pred = torch.argmax(logits, dim=1)
            total += y.size(0)
            correct += int((pred == y).sum().item())
        print(f"Epoch {epoch}: train loss {loss_sum/len(train_dl):.4f}, acc {correct/total:.4f}")

        model.eval()
        with torch.no_grad():
            total, correct = 0, 0
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = torch.argmax(logits, dim=1)
                total += y.size(0)
                correct += int((pred == y).sum().item())
            print(f"Epoch {epoch}: val acc {correct/total:.4f}")

    out_path = args.model_out or os.path.join(paths["models_root"], "video_model.pth")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved video model to {out_path}")


if __name__ == "__main__":
    main()
