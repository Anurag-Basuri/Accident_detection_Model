import argparse
from src.common.config import get_paths

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="Path to trained YOLO weights")
    ap.add_argument("--data", default=None, help="YOLO data.yaml path")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    if YOLO is None:
        raise RuntimeError("Ultralytics YOLO not installed.")

    paths = get_paths()
    data_yaml = args.data or "configs/severity.yaml"
    weights = args.weights or paths.get("models_root", "models") + "/severity_model.pt"

    model = YOLO(weights)
    metrics = model.val(data=data_yaml, imgsz=args.imgsz, split=args.split)
    print(metrics)


if __name__ == "__main__":
    main()
