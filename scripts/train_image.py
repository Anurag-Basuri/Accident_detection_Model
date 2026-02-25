import os
import argparse
import tensorflow as tf
from src.common.config import get_paths
from src.image.dataset import load_image_datasets
from src.image.model import build_mobilenet_v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None, help="Root dir with train/val/test")
    ap.add_argument("--img-size", type=int, nargs=2, default=(224, 224))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--model-out", default=None, help="Output .h5 path")
    args = ap.parse_args()

    paths = get_paths()
    data_dir = args.data_dir or os.path.join(paths["data_root"], paths.get("image_data_dir", "traffic_binary"))
    train_ds, val_ds, test_ds = load_image_datasets(data_dir, tuple(args.img_size), args.batch_size)

    # infer classes from dataset
    class_names = train_ds.class_names
    model = build_mobilenet_v2(num_classes=len(class_names), img_size=tuple(args.img_size))
    model.summary()

    log_dir = os.path.join(paths.get("logs_root", "logs"), "image")
    os.makedirs(log_dir, exist_ok=True)
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir)]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    out_path = args.model_out or os.path.join(paths["models_root"], "image_model.h5")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.save(out_path)
    print(f"Saved image model to {out_path}")


if __name__ == "__main__":
    main()
