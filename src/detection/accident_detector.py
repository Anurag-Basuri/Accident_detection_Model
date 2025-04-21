import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Union, Tuple
import os
import torch
from torchvision.ops import box_iou
import tempfile
import shutil
import logging
from .tracker import ObjectTracker
import tensorflow as tf

def predict_image(image_path):
    """Predicts if an image contains an accident
    
    Args:
        image_path: Path to the image file
        
    Returns:
        bool: True if accident detected, False otherwise
    """
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), "/models/image_model.h5")
    model = tf.keras.models.load_model(model_path)
    
    # Make prediction
    prediction = model.predict(img)
    return prediction[0][0] > 0.4

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AccidentDetector:
    def __init__(self):
        """Initialize the accident detection system"""
        self.tracker = ObjectTracker()
        self.accident_history = []
        self.frame_buffer = []
        self.max_buffer_size = 30  # Store last 30 frames for analysis
        
        # Define vehicle classes
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        # Accident detection parameters
        self.min_confidence = 0.6
        self.min_vehicles = 2
        self.min_overlap = 0.4
        self.min_speed_change = 0.3
        self.min_frames_for_accident = 3
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame for accident detection"""
        try:
            # Convert frame to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Process frame with tracker
            result = self.tracker.process_frame(frame_rgb)
            if "error" in result:
                return {"accident_detected": False, "error": result["error"]}
            
            # Update frame buffer
            self.frame_buffer.append({
                'frame': frame_rgb,
                'tracks': result['tracks'],
                'detections': result['detections']
            })
            if len(self.frame_buffer) > self.max_buffer_size:
                self.frame_buffer.pop(0)
            
            # Detect collisions
            collisions = self.tracker.detect_collisions(result['tracks'])
            
            # Analyze for accidents
            accident_info = self._analyze_accident(collisions, result['tracks'])
            
            # Update accident history
            if accident_info['accident_detected']:
                self.accident_history.append(accident_info)
            
            # Add vehicle type information
            vehicle_types = {}
            for track in result['tracks'].values():
                vehicle_type = track['class']
                vehicle_types[vehicle_type] = vehicle_types.get(vehicle_type, 0) + 1
            
            return {
                'accident_detected': accident_info['accident_detected'],
                'severity': accident_info['severity'],
                'vehicle_count': len(result['tracks']),
                'vehicle_types': vehicle_types,
                'collisions': collisions,
                'tracks': result['tracks'],
                'detections': result['detections']
            }
            
        except Exception as e:
            return {
                'accident_detected': False,
                'error': f"Error processing frame: {str(e)}"
            }
    
    def _analyze_accident(self, collisions: List[Dict], tracks: Dict) -> Dict:
        """Analyze detected collisions and determine if an accident occurred"""
        if not collisions:
            return {
                'accident_detected': False,
                'severity': {'level': 'None', 'severity_score': 0.0}
            }
        
        # Calculate base severity from collisions
        severity_scores = []
        for collision in collisions:
            # Calculate collision severity
            iou_score = collision['iou']
            speed_change_score = max(collision['speed_changes'])
            
            # Adjust severity based on vehicle types
            type_multiplier = 1.0
            if 'truck' in collision['classes'] or 'bus' in collision['classes']:
                type_multiplier = 1.5
            elif 'motorcycle' in collision['classes']:
                type_multiplier = 1.2
            
            # Calculate collision severity
            collision_severity = (iou_score * 0.6 + speed_change_score * 0.4) * type_multiplier
            severity_scores.append(min(collision_severity, 1.0))
        
        # Calculate overall severity
        overall_severity = max(severity_scores) if severity_scores else 0.0
        
        # Determine if accident occurred
        accident_detected = overall_severity > 0.5
        
        # Determine severity level
        if overall_severity < 0.3:
            level = 'Minor'
        elif overall_severity < 0.7:
            level = 'Moderate'
        else:
            level = 'Severe'
        
        return {
            'accident_detected': accident_detected,
            'severity': {
                'level': level,
                'severity_score': overall_severity
            },
            'collisions': collisions
        }
    
    def _calculate_severity(self, vehicle_count: int, vehicle_types: Dict) -> float:
        """Calculate severity score based on vehicle count and types"""
        # Base severity on vehicle count
        count_severity = min(vehicle_count / 5, 1.0)  # Cap at 5 vehicles
        
        # Adjust severity based on vehicle types
        type_multiplier = 1.0
        if 'truck' in vehicle_types or 'bus' in vehicle_types:
            type_multiplier = 1.5
        elif 'motorcycle' in vehicle_types:
            type_multiplier = 1.2
        
        # Calculate final severity
        severity = count_severity * type_multiplier
        return min(severity, 1.0)  # Cap at 1.0

    def process_image(self, image_path: str) -> Dict:
        """Process a single image for accident detection using custom model"""
        try:
            # Use custom model for prediction
            is_accident = predict_image(image_path)
            
            # Read image for visualization
            img = cv2.imread(image_path)
            if img is None:
                return {"accident_detected": False, "error": "Failed to read image"}
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get vehicle count and types using YOLO
            result = self.tracker.process_frame(img_rgb)
            if "error" in result:
                return {"accident_detected": False, "error": result["error"]}
            
            # Count vehicle types
            vehicle_types = {}
            for track in result['tracks'].values():
                vehicle_type = track['class']
                vehicle_types[vehicle_type] = vehicle_types.get(vehicle_type, 0) + 1
            
            # Calculate severity based on vehicle count and types
            severity_score = self._calculate_severity(len(result['tracks']), vehicle_types)
            
            # Determine severity level
            if severity_score < 0.3:
                level = 'Minor'
            elif severity_score < 0.7:
                level = 'Moderate'
            else:
                level = 'Severe'
            
            return {
                'accident_detected': is_accident,
                'severity': {
                    'level': level,
                    'severity_score': severity_score
                },
                'vehicle_count': len(result['tracks']),
                'vehicle_types': vehicle_types,
                'tracks': result['tracks'],
                'detections': result['detections'],
                'image_path': image_path
            }
            
        except Exception as e:
            return {"accident_detected": False, "error": str(e)}

    def process_video(self, video_path: str) -> Dict:
        """Process a video file for accident detection"""
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"accident_detected": False, "error": "Failed to open video"}
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            # Initialize progress tracking
            results = []
            frame_skip = max(1, int(fps / 5))  # Process 5 frames per second
            
            # Process video frames
            for frame_idx in range(0, frame_count, frame_skip):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                result = self.process_frame(frame)
                results.append(result)
            
            # Release video capture
            cap.release()
            
            # Aggregate results
            if results:
                # Calculate average severity
                severity_scores = [r.get('severity', {}).get('severity_score', 0) for r in results]
                avg_severity = np.mean(severity_scores)
                
                # Determine if accident occurred
                accident_detected = any(r.get('accident_detected', False) for r in results)
                
                # Get vehicle counts
                vehicle_counts = [r.get('vehicle_count', 0) for r in results]
                
                return {
                    'accident_detected': accident_detected,
                    'severity': {
                        'severity_score': avg_severity,
                        'level': 'Minor' if avg_severity < 0.3 else 'Moderate' if avg_severity < 0.7 else 'Severe'
                    },
                    'vehicle_count': int(np.mean(vehicle_counts)),
                    'duration': duration,
                    'frame_results': results
                }
            
            return {"accident_detected": False, "error": "No frames processed"}
            
        except Exception as e:
            return {"accident_detected": False, "error": str(e)}
        finally:
            # Clean up resources
            if cap is not None:
                cap.release() 