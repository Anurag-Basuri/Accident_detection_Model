import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import os
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time

class VideoDetector:
    def __init__(self):
        """Initialize the video detection system"""
        try:
            # Load YOLO model
            self.yolo_model = YOLO('yolov8n.pt')
            
            # Load custom accident detection model
            # Get the absolute path of the current file
            current_file = os.path.abspath(__file__)
            # Get the project root directory (two levels up from current file)
            project_root = os.path.dirname(os.path.dirname(current_file))
            # Construct the path to the model file
            model_path = os.path.join(project_root, 'models', 'image_model.h5')
            
            # Verify the model file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model file not found at {model_path}\n"
                    f"Current file: {current_file}\n"
                    f"Project root: {project_root}\n"
                    f"Please ensure the model file exists in the correct location."
                )
            
            self.logger.info(f"Loading model from: {model_path}")
            self.accident_model = tf.keras.models.load_model(model_path)
            
            # Define vehicle classes
            self.vehicle_classes = {
                2: 'car',
                3: 'motorcycle',
                5: 'bus',
                7: 'truck'
            }
            
            # Detection parameters
            self.min_confidence = 0.5
            self.min_vehicles = 2
            self.min_frames_for_accident = 3
            
            # Tracking parameters
            self.track_history = defaultdict(lambda: [])
            self.frame_buffer = []
            self.max_buffer_size = 30
            
            # Configure logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize VideoDetector: {str(e)}")
            raise
    
    def process_video(self, video_path: str) -> Dict:
        """Process a video file for accident detection"""
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"accident_detected": False, "error": "Failed to open video"}
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            # Initialize results
            results = []
            accident_frames = []
            vehicle_counts = []
            severity_scores = []
            
            # Process each frame
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                frame_result = self._process_frame(frame)
                results.append(frame_result)
                
                # Update statistics
                if frame_result['accident_detected']:
                    accident_frames.append(frame_idx)
                vehicle_counts.append(frame_result['vehicle_count'])
                severity_scores.append(frame_result['severity']['severity_score'])
                
                frame_idx += 1
            
            # Release video capture
            cap.release()
            
            # Analyze results
            is_accident = len(accident_frames) >= self.min_frames_for_accident
            avg_vehicle_count = np.mean(vehicle_counts) if vehicle_counts else 0
            avg_severity = np.mean(severity_scores) if severity_scores else 0
            
            # Determine severity level
            if avg_severity < 0.3:
                severity_level = 'Minor'
            elif avg_severity < 0.7:
                severity_level = 'Moderate'
            else:
                severity_level = 'Severe'
            
            return {
                'accident_detected': is_accident,
                'severity': {
                    'level': severity_level,
                    'severity_score': avg_severity
                },
                'vehicle_count': int(avg_vehicle_count),
                'duration': duration,
                'accident_frames': accident_frames,
                'frame_results': results
            }
            
        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")
            return {"accident_detected": False, "error": str(e)}
    
    def _process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame for accident detection"""
        try:
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run YOLO detection
            yolo_results = self.yolo_model.track(
                frame_rgb,
                conf=self.min_confidence,
                persist=True,
                classes=list(self.vehicle_classes.keys())
            )
            
            # Process YOLO results
            if not yolo_results or len(yolo_results) == 0:
                return {
                    'accident_detected': False,
                    'vehicle_count': 0,
                    'severity': {'level': 'None', 'severity_score': 0.0},
                    'tracks': {},
                    'detections': []
                }
            
            # Get the first result
            result = yolo_results[0]
            
            # Count vehicles and get tracks
            vehicle_count = 0
            tracks = {}
            for box in result.boxes:
                if box.id is not None:
                    track_id = int(box.id[0])
                    tracks[track_id] = {
                        'bbox': box.xyxy[0].tolist(),
                        'class': self.vehicle_classes.get(int(box.cls[0]), 'unknown'),
                        'confidence': float(box.conf[0])
                    }
                    vehicle_count += 1
            
            # Run accident detection on frame
            frame_resized = cv2.resize(frame_rgb, (224, 224))
            frame_normalized = frame_resized / 255.0
            frame_expanded = np.expand_dims(frame_normalized, axis=0)
            
            accident_prediction = self.accident_model.predict(frame_expanded)
            is_accident = accident_prediction[0][0] > 0.4
            
            # Calculate severity
            severity_score = self._calculate_severity(vehicle_count, tracks)
            
            return {
                'accident_detected': is_accident,
                'vehicle_count': vehicle_count,
                'severity': {
                    'level': 'Severe' if severity_score > 0.7 else 'Moderate' if severity_score > 0.3 else 'Minor',
                    'severity_score': severity_score
                },
                'tracks': tracks,
                'detections': result.boxes
            }
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            return {
                'accident_detected': False,
                'vehicle_count': 0,
                'severity': {'level': 'None', 'severity_score': 0.0},
                'tracks': {},
                'detections': []
            }
    
    def _calculate_severity(self, vehicle_count: int, tracks: Dict) -> float:
        """Calculate severity score based on vehicle count and types"""
        if vehicle_count < self.min_vehicles:
            return 0.0
        
        # Base severity on vehicle count
        count_severity = min(vehicle_count / 5, 1.0)
        
        # Adjust severity based on vehicle types
        type_multiplier = 1.0
        for track in tracks.values():
            vehicle_type = track['class']
            if vehicle_type in ['truck', 'bus']:
                type_multiplier = max(type_multiplier, 1.5)
            elif vehicle_type == 'motorcycle':
                type_multiplier = max(type_multiplier, 1.2)
        
        # Calculate final severity
        severity = count_severity * type_multiplier
        return min(severity, 1.0) 