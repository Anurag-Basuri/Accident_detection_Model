import os
import argparse
import tensorflow as tf
import numpy as np
from PIL import Image
from src.common.config import get_paths


def load_image(path, img_size=(224, 224)):
    img = Image.open(path).convert("RGB")
    img = img.resize(img_size)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--img-size", type=int, nargs=2, default=(224, 224))
    args = ap.parse_args()

    paths = get_paths()
    model_path = args.model or os.path.join(paths["models_root"], "image_model.h5")
    model = tf.keras.models.load_model(model_path)
    x = load_image(args.input, tuple(args.img_size))
    preds = model.predict(x)
    cls_idx = int(np.argmax(preds, axis=1)[0])
    score = float(np.max(preds))
    print({"class_index": cls_idx, "score": score})


if __name__ == "__main__":
    main()
