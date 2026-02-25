import os
import numpy as np
import tensorflow as tf
from PIL import Image
from src.common.config import get_paths, get_model_map


def _load_image(path, img_size=(224, 224)):
    img = Image.open(path).convert("RGB")
    img = img.resize(img_size)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_image(path: str) -> dict:
    paths = get_paths()
    model_map = get_model_map(paths["models_root"]) or {}
    model_path = os.path.join(paths["models_root"], model_map.get("image", {}).get("path", "image_model.h5"))
    model = tf.keras.models.load_model(model_path)
    x = _load_image(path)
    preds = model.predict(x)
    cls_idx = int(np.argmax(preds, axis=1)[0])
    score = float(np.max(preds))
    return {"class_index": cls_idx, "score": score}
