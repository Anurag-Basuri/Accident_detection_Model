import os
import yaml

DEFAULT_PATHS = {
    "data_root": os.path.join(os.getcwd(), "data"),
    "models_root": os.path.join(os.getcwd(), "models"),
    "logs_root": os.path.join(os.getcwd(), "logs"),
}


def load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_paths(config_dir: str = "configs") -> dict:
    paths_cfg = load_yaml(os.path.join(config_dir, "paths.yaml"))
    return {**DEFAULT_PATHS, **paths_cfg}


def get_model_map(models_dir: str = "models") -> dict:
    latest_path = os.path.join(models_dir, "latest.yaml")
    return load_yaml(latest_path)
