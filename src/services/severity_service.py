import os
from src.common.config import get_paths, get_model_map
from src.severity.infer import infer_severity


def predict_severity(path: str) -> dict:
    paths = get_paths()
    model_map = get_model_map(paths["models_root"]) or {}
    weights = os.path.join(paths["models_root"], model_map.get("severity", {}).get("path", "severity_model.pt"))
    return infer_severity(path, weights)
