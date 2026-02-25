# AI-Powered Real-Time Accident Information System

<p align="center">
  <b>Supervised Machine Learning | Computer Vision | Accident Detection & Severity Assessment</b>
</p>

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [System Architecture](#system-architecture)
- [Models in Detail](#models-in-detail)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation & Setup](#installation--setup)
- [Data — What You Need & How to Prepare It](#data--what-you-need--how-to-prepare-it)
- [Training the Models](#training-the-models)
- [Evaluating the Models](#evaluating-the-models)
- [Running Inference (CLI)](#running-inference-cli)
- [Streamlit Application](#streamlit-application)
- [How the Application Uses Trained Models](#how-the-application-uses-trained-models)
- [Configuration Files](#configuration-files)
- [Testing](#testing)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

Road accidents cause massive loss of life and property every year. Delays in detecting and reporting accidents lead to slower emergency response times and increased casualties. This project builds an **AI-powered system** that uses computer vision and deep learning to:

1. **Detect** whether an accident has occurred from a single **image** or a **video clip**
2. **Assess the severity** of the detected accident

This is a **supervised machine-learning** project — the models learn from human-labelled datasets where each sample is tagged as either **Accident** or **Non-Accident**.

### Problem Statement

- Manual accident reporting is slow, inconsistent, and error-prone
- Emergency services often learn about accidents late, increasing casualties
- Insurance claim processing is delayed due to lack of structured reports

### What This System Does

```
  Input (Image / Video)
         │
         ▼
  ┌──────────────┐     ┌────────────────┐
  │  Is there an  │ YES │  How severe is  │
  │  accident?    │────►│  the accident?  │
  │  (Binary)     │     │  (Severity)     │
  └──────┬───────┘     └────────┬───────┘
         │ NO                    │
         ▼                       ▼
   "Non-Accident"        "Minor / Moderate / Severe"
```

---

## How It Works

The system has **three ML pipelines** that work together:

### Pipeline 1 — Image Classification

> _"Given a single photo, is there an accident?"_

- Takes one image (e.g., a traffic camera screenshot)
- Resizes it to **224 × 224** pixels
- Passes it through a **MobileNetV2** CNN (pretrained on ImageNet)
- Outputs: **Accident** or **Non-Accident** with a confidence score

### Pipeline 2 — Video Classification

> _"Given a video clip, is there an accident?"_

- Takes a short video clip (a few seconds of footage)
- Samples **16 evenly-spaced frames** from the video
- Resizes each frame to **112 × 112** pixels
- Feeds all 16 frames into a **3D ResNet (R3D-18)** that understands both spatial content AND temporal motion
- Outputs: **Accident** or **Non-Accident** with a confidence score

### Pipeline 3 — Severity Assessment

> _"Given an accident image, how severe is it?"_

- Takes an image of a confirmed accident scene
- Passes it through a **YOLOv8** model at **640 × 640** resolution
- Outputs: Severity classification (e.g., Minor / Moderate / Severe) or detected objects with bounding boxes

### Why Three Pipelines?

| Scenario                                  | Best Pipeline            |
| ----------------------------------------- | ------------------------ |
| Traffic camera captures a single frame    | **Image Classification** |
| Dashboard camera records a video clip     | **Video Classification** |
| Accident confirmed, need to assess damage | **Severity Assessment**  |

---

## System Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA PREPARATION                         │
│                                                              │
│  Raw Videos ──► extract_frames.py ──► split_frames.py        │
│  Raw Images ──────────────────────► split into train/val/test│
│  Multiple Sources ──► merge_datasets.py                      │
│  Folder Name Fixes ──► normalize_dataset.py                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING                          │
│                                                              │
│  train_image.py ──► MobileNetV2 ──► image_model.h5          │
│  train_video.py ──► R3D-18      ──► video_model.pth         │
│  train_severity.py ──► YOLOv8   ──► severity_model.pt       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      APPLICATION                             │
│                                                              │
│  Streamlit App (app/streamlit_app.py)                        │
│    ├── Image Detection   ──► image_service.py                │
│    ├── Video Detection   ──► video_service.py                │
│    └── Severity Analysis ──► severity_service.py             │
│                                                              │
│  CLI Scripts (scripts/infer_*.py)                            │
└─────────────────────────────────────────────────────────────┘
```

### Service Layer Architecture

The trained models are NOT loaded directly in the Streamlit app. Instead, a **services layer** (`src/services/`) provides clean prediction APIs:

```
app/streamlit_app.py
      │
      │  calls
      ▼
src/services/
  ├── image_service.py    →  predict_image(path) → {"class_index": 0, "score": 0.95}
  ├── video_service.py    →  predict_video(path) → {"class_index": 1, "score": 0.87}
  └── severity_service.py →  predict_severity(path) → {"class_index": 2, "score": 0.91}
              │
              │  uses
              ▼
        src/image/model.py        ← model architecture
        src/image/dataset.py      ← data loading
        src/video/model.py        ← model architecture
        src/video/dataset.py      ← frame sampling
        src/severity/infer.py     ← YOLO inference
        src/common/config.py      ← path resolution
              │
              │  loads weights from
              ▼
        models/
          ├── image_model.h5
          ├── video_model.pth
          └── severity_model.pt
```

---

## Models in Detail

### 1. Image Classifier — MobileNetV2

```
Input Image (224 × 224 × 3)
       │
       ▼
┌─────────────────────┐
│  Rescaling Layer     │  Normalize pixels to [-1, 1]
│  (1/127.5, offset=-1)│
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  MobileNetV2 Base   │  Pretrained on ImageNet (1.4M images)
│  (FROZEN weights)   │  1280 feature channels output
│  include_top=False   │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  GlobalAveragePool2D │  Reduces spatial dims → 1280-d vector
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Dense(128, ReLU)    │  Fully-connected hidden layer
│  Dropout(0.3)        │  Regularisation to prevent overfitting
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Dense(2, Softmax)   │  Output: [P(Accident), P(Non-Accident)]
└─────────────────────┘
```

**Key details:**

- **Transfer learning**: MobileNetV2 backbone is frozen (pre-trained on ImageNet); only the top Dense layers are trained
- **Loss function**: Categorical cross-entropy
- **Optimiser**: Adam (default learning rate)
- **Output**: Softmax probabilities for 2 classes
- **Framework**: TensorFlow / Keras
- **Saved as**: `models/image_model.h5`

### 2. Video Classifier — R3D-18

```
Input Video → Sample 16 Frames → Each 112 × 112 × 3
       │
       ▼
  Tensor Shape: (Batch, 16, 112, 112, 3)
       │
       │  permute to PyTorch video format
       ▼
  Tensor Shape: (Batch, 3, 16, 112, 112)
       │               ↑
       │         (C, T, H, W)
       ▼
┌─────────────────────┐
│  R3D-18 Backbone     │  3D ResNet with 18 layers
│  (3D convolutions)   │  Learns spatial + temporal features
│                      │  Understands MOTION across frames
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  FC Layer (512 → 2)  │  Final classification
└─────────────────────┘
  Output: [logit_Accident, logit_Non-Accident]
```

**Key details:**

- **3D convolutions**: Unlike 2D CNNs that see one image at a time, R3D-18 uses 3D kernels that span across frames — this lets it detect motion patterns like sudden stops, collisions, and impact
- **Frame sampling**: 16 frames are sampled at equal intervals across the video (`np.linspace`), so a 5-second or 30-second video both get 16 representative frames
- **Loss function**: Cross-entropy loss
- **Optimiser**: Adam (lr=0.001)
- **Output**: Raw logits → softmax → class prediction
- **Framework**: PyTorch
- **Saved as**: `models/video_model.pth` (state dict only)

### 3. Severity Model — YOLOv8

```
Input Image (640 × 640)
       │
       ▼
┌─────────────────────┐
│  YOLOv8 Model        │  Can be classification or detection mode
│  (Ultralytics)       │
└──────────┬──────────┘
           ▼
  Classification Mode:  {"class_index": 2, "score": 0.91}  → "Severe"
         OR
  Detection Mode:       {"detections": 3}  → 3 accident-related objects found
```

**Key details:**

- **YOLOv8** is trained using the Ultralytics framework with a `data.yaml` config
- Can operate in **classification** mode (severity grading) or **detection** mode (bounding boxes)
- **Framework**: Ultralytics
- **Saved as**: `models/severity_model.pt`

---

## Project Structure

```
ML_project/
│
├── app/                               # ── STREAMLIT APPLICATION ──
│   ├── __init__.py
│   ├── streamlit_app.py               # Main UI: upload → predict → display
│   ├── components.py                  # Reusable UI: result cards, confidence gauge
│   └── config.py                      # Model paths, label maps, colours
│
├── src/                               # ── CORE ML LIBRARY ──
│   ├── __init__.py
│   ├── common/
│   │   ├── __init__.py
│   │   └── config.py                  # YAML loader, path resolver
│   ├── image/
│   │   ├── __init__.py
│   │   ├── model.py                   # build_mobilenet_v2() → tf.keras.Model
│   │   └── dataset.py                 # load_image_datasets() → train/val/test
│   ├── video/
│   │   ├── __init__.py
│   │   ├── model.py                   # VideoClassifier(nn.Module) → R3D-18
│   │   └── dataset.py                 # VideoDataset class → frame sampler
│   ├── severity/
│   │   ├── __init__.py
│   │   └── infer.py                   # infer_severity() → YOLOv8 prediction
│   └── services/                      # High-level prediction APIs
│       ├── __init__.py
│       ├── image_service.py           # predict_image(path) → dict
│       ├── video_service.py           # predict_video(path) → dict
│       └── severity_service.py        # predict_severity(path) → dict
│
├── scripts/                           # ── CLI SCRIPTS ──
│   ├── train_image.py                 # Train MobileNetV2
│   ├── train_video.py                 # Train R3D-18
│   ├── train_severity.py              # Train YOLOv8 severity
│   ├── eval_image.py                  # Evaluate image model (metrics + ROC)
│   ├── eval_video.py                  # Evaluate video model (conf matrix)
│   ├── eval_severity.py               # Evaluate severity model
│   ├── infer_image.py                 # Single-image CLI inference
│   ├── infer_video.py                 # Single-video CLI inference
│   └── normalize_dataset.py           # Fix class folder name variants
│
├── preprocessing/                     # ── DATA PREPARATION ──
│   ├── extract_frames.py              # Video → individual frame images
│   ├── split_frames.py                # Split all_data/ → train/val/test
│   ├── merge_datasets.py              # Combine multiple dataset sources
│   └── load_and_visualize.py          # Dataset statistics & sample viz
│
├── configs/                           # ── CONFIGURATION ──
│   ├── yolo_config.yaml               # YOLO model, classes, tracking params
│   └── detection_config.yaml          # Collision detection heuristic thresholds
│
├── models/                            # ── TRAINED WEIGHTS (git-ignored) ──
│   ├── image_model.h5                 # Keras image classifier
│   ├── video_model.pth                # PyTorch video classifier
│   ├── severity_model.pt              # YOLOv8 severity model
│   ├── yolov8n.pt                     # YOLOv8 nano (base)
│   └── yolov8x.pt                     # YOLOv8 extra-large (base)
│
├── data/                              # ── SAMPLE DATA (git-ignored) ──
│   ├── image/                         # Sample accident/non-accident images
│   └── videos/                        # Sample accident/non-accident videos
│
├── tests/                             # ── TEST SUITE ──
│   ├── test_services.py               # Tests for image & video services
│   └── test_severity.py               # Tests for severity service
│
├── .gitignore                         # Ignores data/, models/, venv/, etc.
├── MIT License.md
├── README.md                          # ← You are here
├── requirements.txt                   # Python dependencies
└── setup.py                           # Package configuration
```

---

## Requirements

### Hardware

| Component | Minimum                  | Recommended                                |
| --------- | ------------------------ | ------------------------------------------ |
| **RAM**   | 8 GB                     | 16 GB                                      |
| **GPU**   | Not required (CPU works) | NVIDIA GPU with CUDA (for faster training) |
| **Disk**  | 5 GB (code + models)     | 20 GB+ (with datasets)                     |

### Software

| Software   | Version                     |
| ---------- | --------------------------- |
| **Python** | 3.9 or higher               |
| **pip**    | Latest                      |
| **Git**    | Any                         |
| **OS**     | Windows 10/11, Linux, macOS |

### Python Dependencies

| Package         | Purpose                                                | Version  |
| --------------- | ------------------------------------------------------ | -------- |
| `tensorflow`    | Image model (MobileNetV2) training & inference         | ≥ 2.12.0 |
| `torch`         | Video model (R3D-18) training & inference              | ≥ 2.2.0  |
| `torchvision`   | Pretrained video models, transforms                    | ≥ 0.17.0 |
| `ultralytics`   | YOLOv8 severity model                                  | ≥ 8.1.0  |
| `opencv-python` | Video reading, frame extraction, image processing      | ≥ 4.9.0  |
| `numpy`         | Array operations                                       | ≥ 1.26.0 |
| `pillow`        | Image loading/saving                                   | ≥ 10.2.0 |
| `matplotlib`    | Plotting training curves, confusion matrices           | ≥ 3.8.0  |
| `seaborn`       | Statistical visualisation                              | ≥ 0.12.0 |
| `scikit-learn`  | Metrics (confusion matrix, classification report, ROC) | ≥ 1.2.0  |
| `pandas`        | Data manipulation                                      | ≥ 2.0.0  |
| `pyyaml`        | YAML config file parsing                               | ≥ 6.0.0  |
| `streamlit`     | Web application framework                              | ≥ 1.31.0 |
| `tqdm`          | Progress bars during preprocessing                     | ≥ 4.65.0 |
| `pytest`        | Running tests                                          | ≥ 7.0.0  |

All dependencies are listed in `requirements.txt`.

---

## Installation & Setup

### Step 1 — Clone the Repository

```bash
git clone <repository-url>
cd ML_project
```

### Step 2 — Create a Virtual Environment

```bash
python -m venv venv

# Activate it:
venv\Scripts\activate          # Windows (PowerShell/CMD)
# source venv/bin/activate     # Linux / macOS
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Install the Project (Editable Mode)

This enables `from src.xxx import yyy` imports from anywhere:

```bash
pip install -e .
```

### Quick Verification

```bash
# Check TensorFlow
python -c "import tensorflow; print(tensorflow.__version__)"

# Check PyTorch
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Check Ultralytics
python -c "from ultralytics import YOLO; print('OK')"

# Check Streamlit
python -c "import streamlit; print(streamlit.__version__)"
```

---

## Data — What You Need & How to Prepare It

### What Type of Data?

This project needs **labelled** image and/or video data. Each sample must belong to one of **two classes**:

| Class            | Folder Name     | Description                              | Examples                                             |
| ---------------- | --------------- | ---------------------------------------- | ---------------------------------------------------- |
| **Accident**     | `Accident/`     | Images or videos showing a road accident | Car crashes, motorcycle collisions, vehicle pile-ups |
| **Non-Accident** | `Non_Accident/` | Normal traffic scenes with no accident   | Regular traffic flow, parked cars, empty roads       |

### Data Format

| Pipeline             | Accepted Formats                | Notes                                          |
| -------------------- | ------------------------------- | ---------------------------------------------- |
| Image Classification | `.jpg`, `.jpeg`, `.png`, `.bmp` | Single frame images                            |
| Video Classification | `.mp4`, `.avi`, `.mov`          | Short video clips (2-30 seconds recommended)   |
| Severity Assessment  | `.jpg`, `.jpeg`, `.png`         | For YOLO: annotated in YOLO format (see below) |

### How to Label the Data

Labelling is done **by folder structure** — no separate annotation files needed for the image and video classifiers:

```
Simply place each image/video into the correct folder:

    Accident/car_crash_001.jpg       ← This IS an accident
    Non_Accident/normal_traffic.jpg  ← This is NOT an accident
```

For the **severity model** (YOLOv8), if using detection mode, you need YOLO-format annotation files:

```
# Each image gets a .txt file with the same name:
#   image_001.jpg  →  image_001.txt
#
# Each line in the .txt file:
#   <class_id> <x_center> <y_center> <width> <height>
#
# All values are normalised (0.0 – 1.0) relative to image dimensions
#
# Example (class 0 = Minor, class 1 = Moderate, class 2 = Severe):
0 0.45 0.32 0.12 0.08
2 0.71 0.55 0.25 0.30
```

### How to Divide the Data (Train / Validation / Test Split)

The data must be split into **three sets**:

```
┌──────────────────────────────────────────────────────────┐
│                    ALL YOUR DATA                          │
│                                                          │
│  ┌────────────────────────────────────┐  70% of data     │
│  │            TRAINING SET            │  Model learns    │
│  │                                    │  from this       │
│  └────────────────────────────────────┘                  │
│                                                          │
│  ┌──────────────────────┐  20% of data                   │
│  │    VALIDATION SET     │  Monitors learning during     │
│  │                       │  training (prevents overfit)  │
│  └──────────────────────┘                                │
│                                                          │
│  ┌───────────┐  10% of data                              │
│  │ TEST SET   │  Final evaluation — model never sees     │
│  │            │  this data during training                │
│  └───────────┘                                           │
└──────────────────────────────────────────────────────────┘
```

| Split          | Ratio | Purpose                                                                                        |
| -------------- | ----- | ---------------------------------------------------------------------------------------------- |
| **Train**      | 70%   | The model learns patterns from this data                                                       |
| **Validation** | 20%   | Used during training to check performance and prevent overfitting                              |
| **Test**       | 10%   | Used ONLY after training to measure final accuracy — the model never sees this during training |

### Required Folder Structure

After splitting, your dataset must look like this:

```
your_dataset/
├── train/                          # 70% of data
│   ├── Accident/
│   │   ├── crash_001.jpg
│   │   ├── crash_002.jpg
│   │   ├── collision_video_01.mp4  # (for video pipeline)
│   │   └── ...
│   └── Non_Accident/
│       ├── normal_001.jpg
│       ├── traffic_flow_01.mp4
│       └── ...
│
├── val/                            # 20% of data
│   ├── Accident/
│   │   └── ...
│   └── Non_Accident/
│       └── ...
│
└── test/                           # 10% of data
    ├── Accident/
    │   └── ...
    └── Non_Accident/
        └── ...
```

> **Important:** The folder must be named exactly `Accident` and `Non_Accident`. The system also auto-converts `Non Accident`, `Non-Accident`, and `non-accident` variants.

### How Many Samples Do You Need?

| Quality                        | Images per Class | Videos per Class |
| ------------------------------ | ---------------- | ---------------- |
| **Minimum** (proof of concept) | 200              | 50               |
| **Good** (decent accuracy)     | 1,000            | 200              |
| **Ideal** (production-ready)   | 5,000+           | 1,000+           |

> Tip: Balanced classes (equal Accident and Non-Accident counts) give the best results.

### Preprocessing Pipeline

If you have raw video datasets that need to be converted to frames:

```
Raw Videos (datasets/)
      │
      │  Step 1: Extract frames from each video (every 10th frame)
      ▼
python preprocessing/extract_frames.py \
    --dataset-2 datasets/video-datasets/dataset-2 \
    --dataset-3 datasets/video-datasets/dataset-3 \
    --output processed-datasets \
    --frame-interval 10 \
    --image-size 224 224
      │
      │  Step 2: Split extracted frames into train/val/test
      ▼
python preprocessing/split_frames.py \
    --source processed-datasets/all_data \
    --target processed-datasets \
    --train-ratio 0.7 \
    --val-ratio 0.2
      │
      │  Step 3 (optional): Merge another dataset source
      ▼
python preprocessing/merge_datasets.py \
    --source another-dataset \
    --target processed-datasets
      │
      │  Step 4: Fix folder name inconsistencies
      ▼
python scripts/normalize_dataset.py --root processed-datasets
      │
      │  Step 5: Visualise & verify
      ▼
python preprocessing/load_and_visualize.py --dataset-path processed-datasets
```

The visualisation script shows:

- Total image count per class (bar chart)
- Split distribution (bar chart)
- Missing frame detection
- Random sample image display

---

## Training the Models

### Training the Image Classifier

```bash
python scripts/train_image.py \
    --data-dir path/to/dataset \
    --img-size 224 224 \
    --batch-size 32 \
    --epochs 10 \
    --model-out models/image_model.h5
```

**What happens during training:**

```
Epoch 1/10
  ┌─────────────────────────────────┐
  │ 1. Load batch of 32 images       │
  │ 2. Resize to 224×224             │
  │ 3. Rescale pixels to [-1, 1]     │
  │ 4. Pass through frozen MobileNet │
  │ 5. Pass through Dense layers     │
  │ 6. Compute categorical cross-    │
  │    entropy loss                  │
  │ 7. Backpropagate (only top       │
  │    layers — base is frozen)      │
  │ 8. Update weights with Adam      │
  └─────────────────────────────────┘
  Train Loss: 0.45  |  Train Acc: 0.82
  Val Loss:   0.38  |  Val Acc:   0.87

  ... repeat for all epochs ...

Saved: models/image_model.h5
```

| Parameter      | Flag           | Default               | Description                      |
| -------------- | -------------- | --------------------- | -------------------------------- |
| Data directory | `--data-dir`   | Auto from config      | Root with train/val/test subdirs |
| Image size     | `--img-size`   | 224 224               | Width and height in pixels       |
| Batch size     | `--batch-size` | 32                    | Images per training step         |
| Epochs         | `--epochs`     | 10                    | Full passes over training data   |
| Output path    | `--model-out`  | models/image_model.h5 | Where to save trained weights    |

### Training the Video Classifier

```bash
python scripts/train_video.py \
    --data-dir path/to/video-dataset \
    --frame-count 16 \
    --frame-size 112 112 \
    --batch-size 4 \
    --epochs 10 \
    --lr 0.001 \
    --model-out models/video_model.pth
```

**What happens during training:**

```
Epoch 1/10
  ┌──────────────────────────────────┐
  │ 1. Load batch of 4 videos         │
  │ 2. Sample 16 frames per video     │
  │    (evenly spaced)                │
  │ 3. Resize each frame to 112×112   │
  │ 4. Normalise pixels to [0, 1]     │
  │ 5. Stack → tensor (4, 16, 112,    │
  │    112, 3)                        │
  │ 6. Permute → (4, 3, 16, 112, 112) │
  │ 7. Pass through R3D-18 with 3D    │
  │    convolutions                   │
  │ 8. Compute cross-entropy loss     │
  │ 9. Backpropagate through all      │
  │    layers                         │
  │10. Update weights with Adam       │
  └──────────────────────────────────┘
  Train Loss: 0.62  |  Train Acc: 0.74
  Val Acc: 0.78

  ... repeat for all epochs ...

Saved: models/video_model.pth
```

| Parameter      | Flag             | Default                | Description                    |
| -------------- | ---------------- | ---------------------- | ------------------------------ |
| Data directory | `--data-dir`     | Auto from config       | Root with train/val subdirs    |
| Train subdir   | `--train-subdir` | train                  | Training split folder name     |
| Val subdir     | `--val-subdir`   | val                    | Validation split folder name   |
| Frame count    | `--frame-count`  | 16                     | Frames sampled per video       |
| Frame size     | `--frame-size`   | 112 112                | Frame width and height         |
| Batch size     | `--batch-size`   | 4                      | Videos per training step       |
| Epochs         | `--epochs`       | 10                     | Full passes over training data |
| Learning rate  | `--lr`           | 0.001                  | Adam learning rate             |
| Output path    | `--model-out`    | models/video_model.pth | Where to save trained weights  |

### Training the Severity Model

```bash
python scripts/train_severity.py \
    --data configs/severity.yaml \
    --model yolov8n.pt \
    --epochs 50 \
    --imgsz 640
```

This uses the Ultralytics YOLO training pipeline. You need a `severity.yaml` config pointing to your annotated severity dataset.

| Parameter   | Flag       | Default               | Description                             |
| ----------- | ---------- | --------------------- | --------------------------------------- |
| Data config | `--data`   | configs/severity.yaml | YOLO data.yaml with class names & paths |
| Base model  | `--model`  | yolov8n.pt            | Starting weights (yolov8n/s/m/l/x.pt)   |
| Epochs      | `--epochs` | 50                    | Training epochs                         |
| Image size  | `--imgsz`  | 640                   | Input resolution                        |

---

## Evaluating the Models

### Evaluate Image Model

```bash
python scripts/eval_image.py --data-dir path/to/dataset --model models/image_model.h5
```

**Outputs:**

- Test loss and accuracy
- Confusion matrix
- Classification report (precision, recall, F1-score per class)
- ROC-AUC score (if binary)

### Evaluate Video Model

```bash
python scripts/eval_video.py \
    --data-dir path/to/video-dataset \
    --model models/video_model.pth \
    --split test
```

**Outputs:**

- Confusion matrix
- Classification report (precision, recall, F1-score)

### Evaluate Severity Model

```bash
python scripts/eval_severity.py \
    --weights models/severity_model.pt \
    --data configs/severity.yaml \
    --split val
```

**Outputs:**

- Standard YOLO metrics (mAP, precision, recall)

### Understanding Evaluation Metrics

```
                        Predicted
                   Accident    Non-Accident
                ┌───────────┬──────────────┐
  Actual  Acc.  │    TP     │      FN      │
                ├───────────┼──────────────┤
  Actual  Non   │    FP     │      TN      │
                └───────────┴──────────────┘

  Accuracy  = (TP + TN) / Total
  Precision = TP / (TP + FP)    → "Of all predicted accidents, how many were real?"
  Recall    = TP / (TP + FN)    → "Of all real accidents, how many did we catch?"
  F1-Score  = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## Running Inference (CLI)

### Single Image

```bash
python scripts/infer_image.py --input path/to/image.jpg --model models/image_model.h5
```

Output:

```
{'class_index': 0, 'score': 0.9523}
```

→ Class 0 = Accident, with 95.2% confidence

### Single Video

```bash
python scripts/infer_video.py \
    --input path/to/video.mp4 \
    --model models/video_model.pth \
    --frame-count 16 \
    --frame-size 112 112
```

Output:

```
{'class_index': 1, 'score': 0.8741}
```

→ Class 1 = Non-Accident, with 87.4% confidence

---

## Streamlit Application

The Streamlit app is a web interface that lets you upload files and get instant predictions without touching the command line.

### Launch

```bash
streamlit run app/streamlit_app.py
```

The app opens in your browser at `http://localhost:8501`.

### App Modes

```
┌──────────────────────────────────────────────────────────────┐
│                    STREAMLIT APP                              │
│                                                              │
│  Sidebar                    Main Area                        │
│  ┌──────────────┐          ┌──────────────────────────────┐  │
│  │ Select Mode:  │          │                              │  │
│  │               │          │  [Upload area]               │  │
│  │ ● Image Det.  │  ──►    │                              │  │
│  │ ○ Video Det.  │          │  ┌──────────┐ ┌──────────┐  │  │
│  │ ○ Severity   │          │  │  Image    │ │ Result:  │  │  │
│  │ ○ About      │          │  │  Preview  │ │ Accident │  │  │
│  │               │          │  │          │ │ 95.2%    │  │  │
│  └──────────────┘          │  └──────────┘ └──────────┘  │  │
│                             └──────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

| Mode                    | What It Does                                                 | Model Used       |
| ----------------------- | ------------------------------------------------------------ | ---------------- |
| **Image Detection**     | Upload `.jpg`/`.png` → Accident or Non-Accident + confidence | MobileNetV2 (TF) |
| **Video Detection**     | Upload `.mp4`/`.avi` → Shows sampled frames + prediction     | R3D-18 (PyTorch) |
| **Severity Assessment** | Upload accident image → Severity class or detection boxes    | YOLOv8           |
| **About**               | Project description, model details, architecture info        | —                |

### App Files

| File                   | Purpose                                                                                                                    |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `app/streamlit_app.py` | Main entry point — page layout, file upload, calls services                                                                |
| `app/components.py`    | Reusable UI components: `render_image_result()`, `render_video_result()`, `render_severity_result()`, `confidence_gauge()` |
| `app/config.py`        | App settings: model paths, label maps (`{0: "Accident", 1: "Non-Accident"}`), colours, supported file types                |

---

## How the Application Uses Trained Models

Here is the exact flow when a user uploads an image in the Streamlit app:

```
User uploads photo.jpg
       │
       ▼
app/streamlit_app.py
  1. Saves upload to a temp file
  2. Calls: predict_image(temp_path)
       │
       ▼
src/services/image_service.py
  1. Loads config paths (src/common/config.py)
  2. Resolves model path → models/image_model.h5
  3. Loads the Keras model: tf.keras.models.load_model(...)
  4. Opens the image with PIL, resizes to 224×224
  5. Normalises pixels to [0, 1]
  6. Runs model.predict(image_array)
  7. Returns: {"class_index": 0, "score": 0.95}
       │
       ▼
app/components.py → render_image_result()
  1. Maps class_index 0 → "Accident" (from app/config.py)
  2. Picks colour: red for Accident, green for Non-Accident
  3. Displays image alongside prediction with confidence metric
       │
       ▼
User sees the result in the browser
```

The video and severity flows are analogous — the services layer abstracts all model-loading complexity, while the app layer handles only UI rendering.

---

## Configuration Files

### `configs/yolo_config.yaml`

Controls the YOLO object detection pipeline:

```yaml
model:
  name: yolov8x.pt # YOLOv8 variant (n/s/m/l/x)
  confidence: 0.25 # Minimum detection confidence
  iou: 0.45 # NMS (Non-Max Suppression) IoU threshold

inference:
  image_size: [640, 640] # Input resolution
  device: cuda # cuda or cpu

classes: # COCO class indices for vehicles
  - 2 # car
  - 5 # bus
  - 7 # truck
  - 3 # motorcycle
  - 1 # bicycle

tracking:
  max_age: 30 # Frames to keep tracking a lost object
  min_hits: 3 # Min detections to start tracking
  iou_threshold: 0.3 # IOU threshold for track association
```

### `configs/detection_config.yaml`

Controls accident detection heuristics (when using real-time video analysis with object tracking):

```yaml
collision_detection:
  min_overlap_area: 0.3 # Bounding box overlap to flag collision
  sudden_overlap_threshold: 0.5 # Sudden overlap jump threshold
  max_normal_speed: 50 # Normal speed in pixels/frame
  sudden_speed_drop: 0.7 # Speed drop ratio to flag accident
  angle_threshold: 45 # Opposing direction angle (degrees)
  min_stationary_frames: 15 # Frames to flag vehicle as stopped
  analysis_window: 30 # Frames to analyse for patterns

alert_system:
  min_confidence: 0.8 # Minimum confidence to fire alert
  cooldown_period: 300 # Min frames between alerts
```

---

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

| Test File                | What It Tests                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------------- |
| `tests/test_services.py` | Creates dummy images/videos, runs predict_image() and predict_video(), checks output format |
| `tests/test_severity.py` | Tests severity service with a dummy image (skips if weights not present)                    |

---

## Future Improvements

- Real-time video stream analysis (webcam / RTSP / CCTV feeds)
- Object tracking with DeepSORT/ByteTrack for trajectory-based collision detection
- Multi-camera fusion for wider coverage
- Mobile application for generating real-time alerts
- Integration with emergency response API systems
- Enhanced severity grading with more fine-grained categories
- Dashboard for historical accident analytics

---

## License

This project is licensed under the MIT License — see [MIT License.md](MIT%20License.md) for details.
