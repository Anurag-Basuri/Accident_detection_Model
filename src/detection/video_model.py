import cv2
import numpy as np
from ultralytics import YOLO
import logging
from typing import Dict, List, Tuple, Optional
import os
from collections import deque

class VideoModel:
    def __init__(self, model_path: str = None):
        """Initialize the video detection model"""
        try:
            # Load YOLO model
            if model_path is None:
                # Use default YOLOv8 model
                self.model = YOLO('yolov8n.pt')
            else:
                self.model = YOLO(model_path)
            
            # Define vehicle classes (COCO dataset classes)
            self.vehicle_classes = {
                2: 'car',
                3: 'motorcycle',
                5: 'bus',
                7: 'truck',
                0: 'person'  # Include person for pedestrian accidents
            }
            
            # Detection parameters
            self.min_confidence = 0.5
            self.min_vehicles = 2
            self.min_overlap = 0.3
            self.min_frames_for_accident = 3  # Minimum consecutive frames to confirm accident
            
            # Tracking parameters
            self.frame_buffer = deque(maxlen=30)  # Store last 30 frames for analysis
            self.accident_frames = 0  # Count consecutive frames with accident
            self.vehicle_history = deque(maxlen=10)  # Store vehicle counts for trend analysis
            
            # Configure logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)
            
            # Test model
            self._test_model()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize VideoModel: {str(e)}")
            raise
    
    def _test_model(self):
        """Test if the model is working properly"""
        try:
            # Create a test image
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            results = self.model(test_img)
            self.logger.info("Model test successful")
        except Exception as e:
            self.logger.error(f"Model test failed: {str(e)}")
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
            
            # Process every nth frame to improve performance
            frame_skip = max(1, int(fps / 5))  # Process 5 frames per second
            
            # Process frames
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_skip == 0:
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
            results = self.model(
                frame_rgb,
                conf=self.min_confidence,
                classes=list(self.vehicle_classes.keys())
            )
            
            if not results or len(results) == 0:
                return {
                    'accident_detected': False,
                    'vehicle_count': 0,
                    'severity': {'level': 'None', 'severity_score': 0.0},
                    'tracks': {},
                    'detections': []
                }
            
            # Get the first result
            result = results[0]
            
            # Process detections
            vehicle_boxes = []
            tracks = {}
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id in self.vehicle_classes:
                    tracks[len(tracks)] = {
                        'bbox': box.xyxy[0].tolist(),
                        'class': self.vehicle_classes[class_id],
                        'confidence': float(box.conf[0])
                    }
                    vehicle_boxes.append(box)
            
            # Update frame buffer and vehicle history
            self.frame_buffer.append({
                'frame': frame,
                'boxes': vehicle_boxes,
                'tracks': tracks
            })
            self.vehicle_history.append(len(vehicle_boxes))
            
            # Check for potential accident
            is_accident, overlap, confidence = self._check_for_accident(vehicle_boxes)
            
            # Check for sudden changes in vehicle count
            if len(self.vehicle_history) >= 3:
                recent_counts = list(self.vehicle_history)[-3:]
                if max(recent_counts) - min(recent_counts) >= 2:
                    is_accident = True
            
            # Calculate severity
            severity = self._calculate_severity(len(vehicle_boxes), tracks)
            
            return {
                'accident_detected': is_accident,
                'vehicle_count': len(vehicle_boxes),
                'severity': {
                    'level': 'Severe' if severity > 0.7 else 'Moderate' if severity > 0.3 else 'Minor',
                    'severity_score': severity
                },
                'tracks': tracks,
                'detections': result.boxes,
                'overlap': overlap,
                'confidence': confidence
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
    
    def _check_for_accident(self, boxes: List) -> Tuple[bool, float, float]:
        """Check if an accident is detected based on vehicle positions and movements"""
        if len(boxes) < self.min_vehicles:
            return False, 0.0, 0.0
        
        try:
            # Calculate overlap between vehicles
            overlap = self._calculate_overlap(boxes)
            
            # Calculate average confidence
            confidences = [float(box.conf[0]) for box in boxes if hasattr(box, 'conf') and len(box.conf) > 0]
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Check for accident conditions
            is_accident = (overlap >= self.min_overlap and 
                         len(boxes) >= self.min_vehicles and 
                         avg_confidence >= self.min_confidence)
            
            return is_accident, overlap, avg_confidence
            
        except Exception as e:
            self.logger.error(f"Error checking for accident: {str(e)}")
            return False, 0.0, 0.0
    
    def _calculate_overlap(self, boxes: List) -> float:
        """Calculate the maximum overlap between any two vehicles"""
        if len(boxes) < 2:
            return 0.0
        
        max_overlap = 0.0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                try:
                    box1 = boxes[i].xyxy[0]
                    box2 = boxes[j].xyxy[0]
                    
                    # Calculate intersection area
                    x1 = max(box1[0], box2[0])
                    y1 = max(box1[1], box2[1])
                    x2 = min(box1[2], box2[2])
                    y2 = min(box1[3], box2[3])
                    
                    if x2 > x1 and y2 > y1:
                        intersection = (x2 - x1) * (y2 - y1)
                        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                        overlap = intersection / min(area1, area2)
                        max_overlap = max(max_overlap, overlap)
                except Exception:
                    continue
        
        return max_overlap
    
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
            elif vehicle_type == 'person':
                type_multiplier = max(type_multiplier, 1.3)
        
        # Calculate final severity
        severity = count_severity * type_multiplier
        return min(severity, 1.0) 