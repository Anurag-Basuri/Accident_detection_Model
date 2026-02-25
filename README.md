# AI-Powered Real-Time Accident Information System

## Overview

Road accidents are a significant issue, causing loss of life and property. Delays in accident detection and reporting often lead to increased casualties and slower emergency response times. This project develops an AI-powered system that leverages computer vision and machine learning to detect accidents from images and videos, and assess their severity.

This is a **supervised machine-learning** project: models learn from labelled datasets (Accident / Non-Accident) to classify new inputs.

## Features

- **Image Classification** — Detect accidents in single images using a MobileNetV2 CNN
- **Video Classification** — Detect accidents in video clips using a 3D ResNet (R3D-18)
- **Severity Assessment** — Evaluate accident severity with a YOLOv8-based model
- **Streamlit App** — Interactive web UI for uploading images/videos and viewing predictions

## Models

| Pipeline             | Architecture                              | Framework          | Input               |
| -------------------- | ----------------------------------------- | ------------------ | ------------------- |
| Image Classification | MobileNetV2 (transfer learning, ImageNet) | TensorFlow / Keras | 224×224 image       |
| Video Classification | R3D-18 (3D ResNet)                        | PyTorch            | 16 frames @ 112×112 |
| Severity Assessment  | YOLOv8                                    | Ultralytics        | 640×640 image       |

## Project Structure

```
ML_project/
├── app/                           # Streamlit web application
│   ├── streamlit_app.py           #   Main entry point
│   ├── components.py              #   Reusable UI components
│   └── config.py                  #   App settings & label maps
│
├── src/                           # Core ML library
│   ├── common/
│   │   └── config.py              #   Shared config & path utilities
│   ├── image/
│   │   ├── model.py               #   MobileNetV2 model builder
│   │   └── dataset.py             #   TF image dataset loader
│   ├── video/
│   │   ├── model.py               #   R3D-18 video classifier
│   │   └── dataset.py             #   OpenCV video frame sampler
│   ├── severity/
│   │   └── infer.py               #   YOLOv8 severity inference
│   └── services/                  #   High-level prediction APIs
│       ├── image_service.py
│       ├── video_service.py
│       └── severity_service.py
│
├── scripts/                       # CLI: train / evaluate / infer
│   ├── train_image.py
│   ├── train_video.py
│   ├── train_severity.py
│   ├── eval_image.py
│   ├── eval_video.py
│   ├── eval_severity.py
│   ├── infer_image.py
│   ├── infer_video.py
│   └── normalize_dataset.py
│
├── preprocessing/                 # One-time data preparation
│   ├── extract_frames.py          #   Extract frames from videos
│   ├── split_frames.py            #   Split into train/val/test
│   ├── merge_datasets.py          #   Merge multiple datasets
│   └── load_and_visualize.py      #   Dataset statistics & viz
│
├── configs/                       # YAML configurations
│   ├── yolo_config.yaml
│   └── detection_config.yaml
│
├── models/                        # Trained model weights
│   ├── image_model.h5
│   ├── video_model.pth
│   ├── severity_model.pt
│   ├── yolov8n.pt
│   └── yolov8x.pt
│
├── data/                          # Sample data
│   ├── image/
│   └── videos/
│
├── tests/                         # Test suite
│   ├── test_services.py
│   └── test_severity.py
│
├── requirements.txt
├── setup.py
├── .gitignore
├── MIT License.md
└── README.md
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ML_project

# Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install the project in editable mode (optional, enables src.* imports everywhere)
pip install -e .
```

## Dataset Preparation

The dataset should be organized into `Accident/` and `Non_Accident/` folders under `train/`, `val/`, and `test/` splits:

```
data/
├── train/
│   ├── Accident/
│   └── Non_Accident/
├── val/
│   ├── Accident/
│   └── Non_Accident/
└── test/
    ├── Accident/
    └── Non_Accident/
```

### Preprocessing scripts

```bash
# 1. Extract frames from video datasets
python preprocessing/extract_frames.py --dataset-2 <path> --dataset-3 <path> --output processed-datasets

# 2. Split extracted frames into train/val/test
python preprocessing/split_frames.py --source processed-datasets/all_data --target processed-datasets

# 3. Merge additional datasets
python preprocessing/merge_datasets.py --source <path> --target processed-datasets

# 4. Normalize class folder names (fix "Non Accident" -> "Non_Accident")
python scripts/normalize_dataset.py --root processed-datasets

# 5. Visualize dataset statistics
python preprocessing/load_and_visualize.py --dataset-path processed-datasets
```

## Training

```bash
# Train image classifier (MobileNetV2)
python scripts/train_image.py --data-dir data --epochs 10 --batch-size 32

# Train video classifier (R3D-18)
python scripts/train_video.py --data-dir data --epochs 10 --batch-size 4

# Train severity model (YOLOv8)
python scripts/train_severity.py --data configs/severity.yaml --epochs 50
```

## Evaluation

```bash
# Evaluate image model
python scripts/eval_image.py --data-dir data --model models/image_model.h5

# Evaluate video model
python scripts/eval_video.py --data-dir data --model models/video_model.pth

# Evaluate severity model
python scripts/eval_severity.py --weights models/severity_model.pt --data configs/severity.yaml
```

## Inference (CLI)

```bash
# Single image
python scripts/infer_image.py --input path/to/image.jpg

# Single video
python scripts/infer_video.py --input path/to/video.mp4 --model models/video_model.pth
```

## Streamlit App

The interactive web application lets you upload images or videos and see predictions in real time.

```bash
streamlit run app/streamlit_app.py
```

The app provides three modes:

1. **Image Detection** — Upload an image, get Accident / Non-Accident classification
2. **Video Detection** — Upload a video, see sampled frames and classification
3. **Severity Assessment** — Upload an accident image, get severity analysis

## Testing

```bash
pytest tests/ -v
```

## License

This project is licensed under the MIT License — see [MIT License.md](MIT%20License.md) for details.
