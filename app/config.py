"""App-level configuration: model paths, label maps, UI settings."""

import os

# ---------------------------------------------------------------------------
# Paths — resolved relative to the project root (one level above app/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

IMAGE_MODEL_PATH = os.path.join(MODELS_DIR, "image_model.h5")
VIDEO_MODEL_PATH = os.path.join(MODELS_DIR, "video_model.pth")
SEVERITY_MODEL_PATH = os.path.join(MODELS_DIR, "severity_model.pt")

# ---------------------------------------------------------------------------
# Label maps  (class_index -> human-readable name)
# ---------------------------------------------------------------------------
IMAGE_LABELS = {0: "Accident", 1: "Non-Accident"}
VIDEO_LABELS = {0: "Accident", 1: "Non-Accident"}

SEVERITY_LABELS = {
    0: "Minor",
    1: "Moderate",
    2: "Severe",
}

# ---------------------------------------------------------------------------
# Supported file types
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "bmp"]
VIDEO_EXTENSIONS = ["mp4", "avi", "mov", "mkv"]

# ---------------------------------------------------------------------------
# UI colours / styling
# ---------------------------------------------------------------------------
ACCIDENT_COLOR = "#FF4B4B"       # red
NON_ACCIDENT_COLOR = "#21C354"   # green
SEVERITY_COLORS = {
    "Minor": "#FFC107",       # amber
    "Moderate": "#FF6F00",    # orange
    "Severe": "#D50000",      # dark-red
}

PAGE_ICON = "🚨"
PAGE_TITLE = "Accident Detection System"
