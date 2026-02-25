from typing import Union
import os

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def infer_severity(input_path: str, weights_path: str, img_size: int = 640) -> Union[dict, None]:
    if YOLO is None:
        raise RuntimeError("Ultralytics YOLO not installed. Add 'ultralytics' to requirements.")
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)
    model = YOLO(weights_path)
    results = model(input_path, imgsz=img_size)
    # Aggregate highest-confidence class
    best = None
    for r in results:
        for p in r.probs.top5:  # if classification; for detection use r.boxes
            pass  # placeholder
    # For simplicity, return first result class with max prob
    r = results[0]
    if hasattr(r, 'probs') and r.probs is not None:
        cls = int(r.probs.top1)
        score = float(r.probs.top1conf)
        return {"class_index": cls, "score": score}
    # If detection model, return number of boxes
    if hasattr(r, 'boxes') and r.boxes is not None:
        n = len(r.boxes)
        return {"detections": n}
    return {}
