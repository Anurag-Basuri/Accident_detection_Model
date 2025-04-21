import cv2
import numpy as np
from ultralytics import YOLO
import os
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time

class YOLODetector:
    def __init__(self, model_path: str = 'yolov8n.pt'):
        """Initialize the YOLO-based detection system"""
        try:
            # Load YOLO model
            self.model = YOLO(model_path)
            
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
            self.min_overlap = 0.3
            self.min_speed_change = 0.3
            
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
            self.logger.error(f"Failed to initialize YOLODetector: {str(e)}")
            raise
    
    def process_image(self, image_path: str) -> Dict:
        """Process a single image for accident detection"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return {"accident_detected": False, "error": "Failed to read image"}
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Run YOLO detection with tracking
            results = self.model.track(
                img_rgb,
                conf=self.min_confidence,
                persist=True,
                classes=list(self.vehicle_classes.keys()),
                tracker="bytetrack.yaml"
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
                if box.id is not None:
                    track_id = int(box.id[0])
                    tracks[track_id] = {
                        'bbox': box.xyxy[0].tolist(),
                        'class': self.vehicle_classes.get(int(box.cls[0]), 'unknown'),
                        'confidence': float(box.conf[0])
                    }
                    vehicle_boxes.append(box)
            
            # Check for potential accident
            is_accident, overlap, confidence = self._check_for_accident(vehicle_boxes)
            
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
            self.logger.error(f"Error processing image: {str(e)}")
            return {"accident_detected": False, "error": str(e)}
    
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
            
            # Run YOLO detection with tracking
            results = self.model.track(
                frame_rgb,
                conf=self.min_confidence,
                persist=True,
                classes=list(self.vehicle_classes.keys()),
                tracker="bytetrack.yaml"
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
                if box.id is not None:
                    track_id = int(box.id[0])
                    tracks[track_id] = {
                        'bbox': box.xyxy[0].tolist(),
                        'class': self.vehicle_classes.get(int(box.cls[0]), 'unknown'),
                        'confidence': float(box.conf[0])
                    }
                    vehicle_boxes.append(box)
            
            # Check for potential accident
            is_accident, overlap, confidence = self._check_for_accident(vehicle_boxes)
            
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
        
        # Calculate final severity
        severity = count_severity * type_multiplier
        return min(severity, 1.0) 