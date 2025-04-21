import cv2
import numpy as np
from ultralytics import YOLO
import logging
from typing import Dict, List, Tuple, Optional
import os

class ImageModel:
    def __init__(self, model_path: str = None):
        """Initialize the image detection model"""
        try:
            # Load YOLO model
            if model_path is None:
                # Use YOLOv8n model for better accuracy
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
            self.min_distance = 50  # Minimum distance between vehicles for accident detection
            
            # Configure logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)
            
            # Test model
            self._test_model()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ImageModel: {str(e)}")
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
    
    def process_image(self, image_path: str) -> Dict:
        """Process a single image for accident detection"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return {"accident_detected": False, "error": "Failed to read image"}
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Run YOLO detection
            results = self.model(
                img_rgb,
                conf=self.min_confidence,
                classes=list(self.vehicle_classes.keys())
            )
            
            if not results or len(results) == 0:
                return {
                    'accident_detected': False,
                    'vehicle_count': 0,
                    'severity': {'level': 'None', 'severity_score': 0.0},
                    'tracks': {},
                    'detections': [],
                    'visualization': None
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
            
            # Check for potential accident
            is_accident, overlap, confidence = self._check_for_accident(vehicle_boxes)
            
            # Calculate severity
            severity = self._calculate_severity(len(vehicle_boxes), tracks)
            
            # Create visualization
            visualization = self._create_visualization(img, vehicle_boxes, is_accident)
            
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
                'confidence': confidence,
                'image_path': image_path,
                'visualization': visualization
            }
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return {"accident_detected": False, "error": str(e)}
    
    def _check_for_accident(self, boxes: List) -> Tuple[bool, float, float]:
        """Check if an accident is detected based on vehicle positions"""
        if len(boxes) < self.min_vehicles:
            return False, 0.0, 0.0
        
        try:
            # Calculate overlap between vehicles
            overlap = self._calculate_overlap(boxes)
            
            # Calculate distances between vehicles
            distances = self._calculate_distances(boxes)
            
            # Calculate average confidence
            confidences = [float(box.conf[0]) for box in boxes if hasattr(box, 'conf') and len(box.conf) > 0]
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Check for accident conditions
            is_accident = (
                (overlap >= self.min_overlap or any(d < self.min_distance for d in distances)) and 
                len(boxes) >= self.min_vehicles and 
                avg_confidence >= self.min_confidence
            )
            
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
    
    def _calculate_distances(self, boxes: List) -> List[float]:
        """Calculate distances between vehicle centers"""
        distances = []
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                try:
                    box1 = boxes[i].xyxy[0]
                    box2 = boxes[j].xyxy[0]
                    
                    # Calculate centers
                    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
                    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
                    
                    # Calculate Euclidean distance
                    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                    distances.append(distance)
                except Exception:
                    continue
        
        return distances
    
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
    
    def _create_visualization(self, img: np.ndarray, boxes: List, is_accident: bool) -> np.ndarray:
        """Create visualization of detection results"""
        try:
            # Create a copy of the image
            vis_img = img.copy()
            
            # Define colors
            accident_color = (0, 0, 255)  # Red for accidents
            normal_color = (0, 255, 0)    # Green for normal detection
            
            # Draw bounding boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = accident_color if is_accident else normal_color
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                
                # Add class label and confidence
                class_id = int(box.cls[0])
                class_name = self.vehicle_classes.get(class_id, 'unknown')
                confidence = float(box.conf[0])
                label = f"{class_name} {confidence:.2f}"
                
                # Draw label background
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis_img, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
                
                # Draw label text
                cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add accident status
            status = "ACCIDENT DETECTED!" if is_accident else "No Accident"
            status_color = accident_color if is_accident else normal_color
            cv2.putText(vis_img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            return vis_img
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
            return img 