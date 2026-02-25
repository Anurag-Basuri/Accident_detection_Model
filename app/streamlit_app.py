"""
Accident Detection System — Streamlit Application
===================================================

Launch with:
    streamlit run app/streamlit_app.py

Features:
    1. Image-based accident detection  (MobileNetV2 / TensorFlow)
    2. Video-based accident detection   (R3D-18 / PyTorch)
    3. Severity assessment              (YOLOv8 / Ultralytics)
"""

import sys
import os
import tempfile

import streamlit as st

# Ensure project root is on sys.path so `src.*` imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import (
    PAGE_TITLE,
    PAGE_ICON,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    IMAGE_MODEL_PATH,
    VIDEO_MODEL_PATH,
    SEVERITY_MODEL_PATH,
)
from app.components import (
    render_image_result,
    render_video_result,
    render_severity_result,
)

# ── Page configuration ────────────────────────────────────────────────────
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")


# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.title(f"{PAGE_ICON} {PAGE_TITLE}")
mode = st.sidebar.radio(
    "Select Mode",
    ["Image Detection", "Video Detection", "Severity Assessment", "About"],
)

st.sidebar.markdown("---")
st.sidebar.caption("AI-Powered Real-Time Accident Information System")


# ── Helper: save uploaded file to temp path ───────────────────────────────
def _save_upload(uploaded_file, suffix: str) -> str:
    """Write an UploadedFile to a temp path and return that path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


# ═══════════════════════════════════════════════════════════════════════════
# MODE 1 — Image Detection
# ═══════════════════════════════════════════════════════════════════════════
if mode == "Image Detection":
    st.header("Image Accident Detection")
    st.write("Upload an image to classify it as **Accident** or **Non-Accident**.")

    uploaded = st.file_uploader(
        "Choose an image", type=IMAGE_EXTENSIONS, key="img_upload"
    )

    if uploaded is not None:
        tmp_path = _save_upload(uploaded, f".{uploaded.name.split('.')[-1]}")
        with st.spinner("Analysing image..."):
            try:
                from src.services.image_service import predict_image

                result = predict_image(tmp_path)
                render_image_result(tmp_path, result)
            except FileNotFoundError:
                st.error(
                    f"Image model weights not found at `{IMAGE_MODEL_PATH}`. "
                    "Please train the model first using `python scripts/train_image.py`."
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════
# MODE 2 — Video Detection
# ═══════════════════════════════════════════════════════════════════════════
elif mode == "Video Detection":
    st.header("Video Accident Detection")
    st.write("Upload a video clip to classify it as **Accident** or **Non-Accident**.")

    uploaded = st.file_uploader(
        "Choose a video", type=VIDEO_EXTENSIONS, key="vid_upload"
    )

    if uploaded is not None:
        tmp_path = _save_upload(uploaded, f".{uploaded.name.split('.')[-1]}")
        with st.spinner("Analysing video (sampling frames)..."):
            try:
                from src.services.video_service import predict_video

                result = predict_video(tmp_path)
                render_video_result(tmp_path, result)
            except FileNotFoundError:
                st.error(
                    f"Video model weights not found at `{VIDEO_MODEL_PATH}`. "
                    "Please train the model first using `python scripts/train_video.py`."
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════
# MODE 3 — Severity Assessment
# ═══════════════════════════════════════════════════════════════════════════
elif mode == "Severity Assessment":
    st.header("Accident Severity Assessment")
    st.write(
        "Upload an image of an accident scene to assess the severity "
        "using a YOLOv8-based model."
    )

    uploaded = st.file_uploader(
        "Choose an image", type=IMAGE_EXTENSIONS, key="sev_upload"
    )

    if uploaded is not None:
        tmp_path = _save_upload(uploaded, f".{uploaded.name.split('.')[-1]}")
        with st.spinner("Running severity analysis..."):
            try:
                from src.services.severity_service import predict_severity

                result = predict_severity(tmp_path)
                render_severity_result(tmp_path, result)
            except FileNotFoundError:
                st.error(
                    f"Severity model weights not found at `{SEVERITY_MODEL_PATH}`. "
                    "Please train the model first using `python scripts/train_severity.py`."
                )
            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════
# ABOUT
# ═══════════════════════════════════════════════════════════════════════════
else:
    st.header("About")
    st.markdown(
        """
        ## AI-Powered Real-Time Accident Information System

        This application demonstrates a supervised machine-learning system that
        classifies images and videos as **Accident** or **Non-Accident**, and
        optionally assesses the **severity** of detected accidents.

        ### Models Used

        | Pipeline | Model | Framework |
        |----------|-------|-----------|
        | Image Classification | MobileNetV2 (transfer learning) | TensorFlow / Keras |
        | Video Classification | R3D-18 (3D ResNet) | PyTorch |
        | Severity Assessment | YOLOv8 | Ultralytics |

        ### How It Works

        1. **Image Detection** — A single image is resized to 224×224 and passed
           through a MobileNetV2 backbone with a custom classification head.
        2. **Video Detection** — 16 evenly-spaced frames are sampled from the
           video, resized to 112×112, and fed into a 3D ResNet (R3D-18) that
           captures both spatial and temporal features.
        3. **Severity Assessment** — A YOLOv8 model analyses the accident image
           to classify the severity or detect relevant objects.

        ### Project Structure

        ```
        src/          — Core ML library (models, datasets, services)
        scripts/      — CLI scripts for training, evaluation, inference
        preprocessing/— Data preparation utilities
        configs/      — YAML configuration files
        models/       — Trained model weights
        app/          — This Streamlit application
        tests/        — Test suite
        ```

        ---
        *Developed as a college project for supervised machine learning.*
        """
    )
