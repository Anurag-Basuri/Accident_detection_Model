# AI-Powered Real-Time Accident Information System

## Overview

Road accidents are a significant issue, causing loss of life and property. Delays in accident detection and reporting often lead to increased casualties and slower emergency response times. This project aims to develop an AI-powered system that leverages computer vision and machine learning to detect accidents in real-time from photos and videos.

## Features

- **Accident Detection**: Machine learning-based classification to distinguish between accident and non-accident events.
- **Severity Assessment**: Use of object detection techniques (e.g., YOLO, Faster R-CNN) to evaluate accident severity.
- **Automated Notifications**: Real-time alerts to emergency services upon accident detection.
- **Insurance Reporting**: Automated generation of structured reports to assist in claim processing and dispute resolution.

## Technical Implementation

### Model Architecture

- **Hybrid CNN-LSTM Model**: Combines EfficientNetB0 for spatial feature extraction with LSTM for temporal sequence learning
- **Key Components**:
  - TimeDistributed layers for video sequence processing
  - LSTM layers (128 and 64 units) for temporal pattern recognition
  - Batch normalization and dropout for regularization
  - Binary classification output (accident vs non-accident)

### Training Configuration

- **Image Processing**:
  - Input size: 224x224 pixels
  - Sequence length: 30 frames
  - Batch size: 8 (training), 16 (evaluation)
- **Optimization**:
  - Optimizer: AdamW with learning rate 1e-4
  - Loss function: Binary cross-entropy
  - Early stopping with patience of 10 epochs
  - Learning rate reduction on plateau

### Data Pipeline

- **Preprocessing**:
  - Video frame extraction
  - Dataset merging and splitting
  - Data augmentation for training
  - Validation set preparation
- **Evaluation**:
  - Comprehensive performance metrics
  - Confusion matrix visualization
  - Training history plotting
  - Results export to CSV

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

The system will utilize machine learning and computer vision to detect accidents in real-time, classify their severity, and automate emergency notifications. The model will be trained on real-world accident datasets to ensure high accuracy and relevance.

By incorporating AI into road safety, this project aims to:

- Drastically reduce accident response times.
- Improve efficiency in insurance claim processing.
- Enhance overall public safety through smarter traffic systems.

## Scope

This project focuses on image/video-based accident detection and does not include factors like weather or driver behavior analysis.

## Future Improvements

1. Integration with real-time traffic monitoring systems
2. Enhanced severity assessment using object detection
3. Multi-camera fusion for better coverage
4. Mobile application for real-time alerts
5. Integration with emergency response systems

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- EfficientNetB0 for feature extraction
- TensorFlow and Keras for deep learning framework
- OpenCV for video processing
- Various open-source datasets for training and validation
