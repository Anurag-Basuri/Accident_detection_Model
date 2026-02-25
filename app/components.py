"""Reusable Streamlit UI components for displaying predictions."""

import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from app.config import (
    ACCIDENT_COLOR,
    NON_ACCIDENT_COLOR,
    SEVERITY_COLORS,
    IMAGE_LABELS,
    VIDEO_LABELS,
    SEVERITY_LABELS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_from_path(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _label_color(label: str) -> str:
    """Return a colour hex string for the given label."""
    if "accident" in label.lower() and "non" not in label.lower():
        return ACCIDENT_COLOR
    return NON_ACCIDENT_COLOR


# ---------------------------------------------------------------------------
# Image result
# ---------------------------------------------------------------------------

def render_image_result(image_path: str, prediction: dict):
    """Display the uploaded image alongside its classification result.

    Parameters
    ----------
    image_path : str
        Path to the image file on disk.
    prediction : dict
        Must contain ``class_index`` (int) and ``score`` (float 0-1).
    """
    label = IMAGE_LABELS.get(prediction["class_index"], "Unknown")
    score = prediction["score"]
    color = _label_color(label)

    col_img, col_res = st.columns([2, 1])
    with col_img:
        st.image(_pil_from_path(image_path), use_container_width=True)
    with col_res:
        st.markdown(f"### Prediction")
        st.markdown(
            f"<h2 style='color:{color}'>{label}</h2>",
            unsafe_allow_html=True,
        )
        st.metric("Confidence", f"{score:.1%}")


# ---------------------------------------------------------------------------
# Video result
# ---------------------------------------------------------------------------

def _sample_video_frames(video_path: str, n: int = 6) -> list:
    """Return up to *n* evenly-spaced frames as PIL images."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, total - 1, num=n, dtype=int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    cap.release()
    return frames


def render_video_result(video_path: str, prediction: dict):
    """Display sampled frames from the video and the classification result.

    Parameters
    ----------
    video_path : str
        Path to the video file on disk.
    prediction : dict
        Must contain ``class_index`` (int) and ``score`` (float 0-1).
    """
    label = VIDEO_LABELS.get(prediction["class_index"], "Unknown")
    score = prediction["score"]
    color = _label_color(label)

    st.markdown(
        f"<h2 style='color:{color}'>Result: {label} ({score:.1%})</h2>",
        unsafe_allow_html=True,
    )

    frames = _sample_video_frames(video_path, n=6)
    if frames:
        st.markdown("**Sampled frames:**")
        cols = st.columns(len(frames))
        for col, frame in zip(cols, frames):
            col.image(frame, use_container_width=True)
    else:
        st.warning("Could not read frames from the video file.")


# ---------------------------------------------------------------------------
# Severity result
# ---------------------------------------------------------------------------

def render_severity_result(image_path: str, prediction: dict):
    """Display severity assessment result.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    prediction : dict
        Expected keys vary:
        - Classification model: ``class_index``, ``score``
        - Detection model: ``detections`` (int count)
    """
    col_img, col_res = st.columns([2, 1])
    with col_img:
        st.image(_pil_from_path(image_path), use_container_width=True)

    with col_res:
        st.markdown("### Severity Assessment")
        if "class_index" in prediction:
            label = SEVERITY_LABELS.get(prediction["class_index"], f"Class {prediction['class_index']}")
            score = prediction.get("score", 0)
            color = SEVERITY_COLORS.get(label, ACCIDENT_COLOR)
            st.markdown(
                f"<h2 style='color:{color}'>{label}</h2>",
                unsafe_allow_html=True,
            )
            st.metric("Confidence", f"{score:.1%}")
        elif "detections" in prediction:
            n = prediction["detections"]
            st.metric("Objects detected", n)
            if n > 0:
                st.info(f"YOLO detected **{n}** object(s) in the image.")
            else:
                st.success("No accident-related objects detected.")
        else:
            st.json(prediction)


# ---------------------------------------------------------------------------
# Confidence gauge (simple progress bar)
# ---------------------------------------------------------------------------

def confidence_gauge(score: float, label: str = "Confidence"):
    """Render a visual confidence bar."""
    color = ACCIDENT_COLOR if score > 0.7 else NON_ACCIDENT_COLOR
    st.markdown(f"**{label}:** {score:.1%}")
    st.progress(min(score, 1.0))
