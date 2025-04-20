# AI-Powered Real-Time Accident Information System

## Overview

Road accidents are a significant issue, causing loss of life and property. Delays in accident detection and reporting often lead to increased casualties and slower emergency response times. This project aims to develop an AI-powered system that leverages computer vision and machine learning to detect accidents in real-time from photos and videos.

## Features

- **Accident Detection**: Real-time detection of accidents using object detection and tracking
- **Severity Assessment**: Analysis of collision patterns and vehicle behavior
- **Automated Notifications**: Real-time alerts to emergency services upon accident detection
- **Insurance Reporting**: Automated generation of structured reports to assist in claim processing

## Technical Implementation

### Core Components

#### 1. Object Detection (YOLOv8)

- **Model**: YOLOv8 pretrained on COCO dataset
- **Capabilities**:
  - Detection of 80 common classes (cars, trucks, buses, etc.)
  - High-precision bounding box detection
  - Real-time processing capabilities
- **Output Format**:
  ```json
  [{"label": "car", "confidence": 0.9, "bbox": [x1, y1, x2, y2]}, ...]
  ```

#### 2. Object Tracking (DeepSORT/ByteTrack)

- **Purpose**: Track objects across video frames
- **Features**:
  - Object ID assignment and maintenance
  - Motion prediction and tracking
  - Speed and direction calculation
- **Output**: Tracked object information with IDs and trajectories

### Detection Pipeline

```
           Video
             ↓
      [1] Frame Extraction
             ↓
      [2] YOLOv8 Detection
             ↓
   [3] DeepSORT Tracking
             ↓
[4] Accident Heuristics
             ↓
     🚨 Accident Detection
```

### Accident Detection Logic

#### Heuristic Rules

1. **Bounding Box Overlap**

   - Sudden and significant overlap between vehicle bounding boxes
   - Threshold-based collision detection

2. **Speed Analysis**

   - Sudden deceleration detection
   - Abnormal speed patterns
   - Impact velocity calculation

3. **Direction Analysis**

   - Opposing direction collision detection
   - Abnormal trajectory changes
   - Post-impact movement patterns

4. **Post-Impact Behavior**
   - Vehicle immobilization detection
   - Multiple vehicle involvement
   - Secondary collision detection

### Technical Stack

| Component            | Technology           |
| -------------------- | -------------------- |
| Object Detection     | YOLOv8 (ultralytics) |
| Object Tracking      | DeepSORT/ByteTrack   |
| Video Processing     | OpenCV               |
| Programming Language | Python               |

## Project Structure

```
├── train.py                 # Main training script
├── evaluate.py              # Model evaluation
├── models/
│   └── cnn_lstm_efficientnet.py  # Model architecture
├── preprocessing/
│   ├── extract_frames.py    # Video frame extraction
│   ├── merge_datasets.py    # Dataset combination
│   ├── load_and_visualize.py # Data visualization
│   └── split-frames.py      # Data splitting
├── datasets/                # Raw datasets
├── processed-datasets/      # Processed data
└── notebooks/              # Jupyter notebooks
```

## Installation and Setup

1. Clone the repository:

```bash
git clone [repository-url]
cd [repository-name]
```

2. Install dependencies:

```bash
pip install ultralytics
pip install opencv-python
pip install deep_sort_realtime
pip install -r requirements.txt
```

3. Prepare your dataset:

- Place raw videos in the `datasets/` directory
- Run preprocessing scripts to extract and prepare frames
- Ensure proper train/validation split

## Usage

### Training

```bash
python train.py
```

### Evaluation

```bash
python evaluate.py
```

## Objectives

1. Faster and more accurate accident detection.
2. Automated emergency response to reduce casualties.
3. Streamlined insurance claim processing through automated reporting.
4. Scalable AI model for integration with smart traffic systems.

## Problem Statement

Road accidents, particularly in the Indian subcontinent, are exacerbated by traffic congestion, poor road conditions, and delays in detection and reporting. Current manual reporting methods are inefficient and inconsistent, leading to slower emergency responses and prolonged insurance claim processes.

### Challenges

- Limited availability of high-quality accident datasets specific to the Indian subcontinent.
- Accurate differentiation between genuine accidents and false alarms.
- Real-time processing of large-scale image/video data.
- Seamless integration with emergency response systems.

## Proposed Solution

The system will utilize YOLOv8 and DeepSORT to detect and track vehicles in real-time, applying heuristic rules to identify potential accidents. The system will be trained on real-world traffic scenarios to ensure high accuracy and relevance.

By incorporating AI into road safety, this project aims to:

- Drastically reduce accident response times.
- Improve efficiency in insurance claim processing.
- Enhance overall public safety through smarter traffic systems.

## Scope

This project focuses on image/video-based accident detection using object detection and tracking, and does not include factors like weather or driver behavior analysis.

## Future Improvements

1. Integration with real-time traffic monitoring systems
2. Enhanced severity assessment using object detection
3. Multi-camera fusion for better coverage
4. Mobile application for real-time alerts
5. Integration with emergency response systems

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- YOLOv8 for object detection
- DeepSORT/ByteTrack for object tracking
- OpenCV for video processing
- Various open-source datasets for training and validation
