import argparse
from src.common.config import get_paths

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="YOLO data.yaml path")
    ap.add_argument("--model", default="yolov8n.pt", help="Base YOLO model")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    if YOLO is None:
        raise RuntimeError("Ultralytics YOLO not installed. Add 'ultralytics' to requirements.")

    paths = get_paths()
    data_yaml = args.data or "configs/severity.yaml"

    model = YOLO(args.model)
    model.train(data=data_yaml, epochs=args.epochs, imgsz=args.imgsz, project=paths.get("logs_root", "logs"), name="severity")


if __name__ == "__main__":
    main()
