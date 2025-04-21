import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Union, Tuple, Optional
import os
import torch
from torchvision.ops import box_iou
import tempfile
import shutil
import logging
from dataclasses import dataclass
from enum import Enum
import time

class SeverityLevel(Enum):
    NONE = "None"
    MINOR = "Minor"
    MODERATE = "Moderate"
    SEVERE = "Severe"

@dataclass
class DetectionResult:
    accident_detected: bool
    detections: List
    vehicle_count: int
    overlap: float
    severity: Dict
    tracking_data: Dict
    confidence: float
    error: Optional[str] = None

class ImprovedAccidentDetector:
    def __init__(self):
        """Initialize the improved accident detection system"""
        try:
            # Load YOLOv8 model with GPU acceleration if available
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_path = 'yolov8x.pt'
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            # Initialize model with higher confidence threshold
            self.yolo_model = YOLO(model_path).to(self.device)
            
            # Enhanced vehicle classes with more specific types
            self.vehicle_classes = {
                2: "car",
                3: "motorcycle",
                5: "bus",
                7: "truck",
                1: "bicycle",
                4: "airplane",
                6: "train"
            }
            
            # Optimized detection thresholds
            self.min_confidence = 0.7  # Increased for better accuracy
            self.min_vehicles = 2
            self.min_overlap = 0.3
            self.min_confidence_for_accident = 0.8  # Increased for fewer false positives
            self.min_vehicle_size = 100
            
            # Enhanced motion analysis parameters
            self.motion_threshold = 15
            self.prev_frame = None
            self.prev_detections = None
            self.frame_buffer = []
            self.buffer_size = 5
            
            # Improved tracking parameters
            self.track_history = {}
            self.max_track_history = 30
            self.track_cleanup_interval = 100
            self.frame_counter = 0
            
            # Advanced validation parameters
            self.min_velocity_threshold = 5.0
            self.max_distance_threshold = 200
            self.min_frame_consistency = 3
            self.max_speed_difference = 30.0
            self.min_accident_duration = 0.5  # seconds
            
            # Initialize logging
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ImprovedAccidentDetector: {str(e)}")
            raise
    
    def process_image(self, image_path: str) -> DetectionResult:
        """Process a single image for accident detection with enhanced accuracy"""
        try:
            # Read and validate image
            img = cv2.imread(image_path)
            if img is None:
                return DetectionResult(
                    accident_detected=False,
                    detections=[],
                    vehicle_count=0,
                    overlap=0.0,
                    severity={"level": SeverityLevel.NONE.value, "severity_score": 0.0},
                    tracking_data={},
                    confidence=0.0,
                    error="Failed to read image"
                )
            
            # Convert to RGB and enhance image quality
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = self._enhance_image(img_rgb)
            
            # Run YOLO detection with improved tracking
            results = self.yolo_model.track(
                img_rgb,
                conf=self.min_confidence,
                persist=True,
                classes=list(self.vehicle_classes.keys()),
                tracker="bytetrack.yaml"
            )
            
            # Validate and process results
            if not results or not isinstance(results, list):
                return DetectionResult(
                    accident_detected=False,
                    detections=[],
                    vehicle_count=0,
                    overlap=0.0,
                    severity={"level": SeverityLevel.NONE.value, "severity_score": 0.0},
                    tracking_data={},
                    confidence=0.0,
                    error="No detections found"
                )
            
            # Get the first result
            results = results[0] if len(results) > 0 else None
            if not results:
                return DetectionResult(
                    accident_detected=False,
                    detections=[],
                    vehicle_count=0,
                    overlap=0.0,
                    severity={"level": SeverityLevel.NONE.value, "severity_score": 0.0},
                    tracking_data={},
                    confidence=0.0,
                    error="No detections found"
                )
            
            # Process detections with enhanced validation
            vehicle_boxes = self._process_detections(results)
            
            # Check for potential accident with improved criteria
            is_accident, overlap, confidence = self._check_for_accident(vehicle_boxes)
            
            # Calculate severity with enhanced metrics
            severity = self._calculate_severity(vehicle_boxes, overlap, confidence)
            
            # Update tracking with improved accuracy
            self._update_tracking(vehicle_boxes)
            
            return DetectionResult(
                accident_detected=is_accident,
                detections=vehicle_boxes,
                vehicle_count=len(vehicle_boxes),
                overlap=overlap,
                severity=severity,
                tracking_data=self.track_history,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Error in process_image: {str(e)}")
            return DetectionResult(
                accident_detected=False,
                detections=[],
                vehicle_count=0,
                overlap=0.0,
                severity={"level": SeverityLevel.NONE.value, "severity_score": 0.0},
                tracking_data={},
                confidence=0.0,
                error=str(e)
            )
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality for better detection"""
        try:
            # Apply CLAHE for better contrast
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge((l,a,b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            # Apply Gaussian blur to reduce noise
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            return enhanced
        except Exception as e:
            self.logger.warning(f"Error in image enhancement: {str(e)}")
            return image
    
    def _process_detections(self, results) -> List:
        """Process and validate detections with enhanced checks"""
        vehicle_boxes = []
        
        if not hasattr(results, 'boxes') or results.boxes is None:
            return vehicle_boxes
        
        for box in results.boxes:
            try:
                # Enhanced validation
                if not self._validate_detection(box):
                    continue
                
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
                width = x2 - x1
                height = y2 - y1
                
                # Check vehicle size and aspect ratio
                if (width >= self.min_vehicle_size or height >= self.min_vehicle_size) and \
                   (0.3 <= width/height <= 3.0):  # Valid aspect ratio for vehicles
                    vehicle_boxes.append(box)
                    
            except Exception as e:
                self.logger.warning(f"Error processing box: {str(e)}")
                continue
        
        return vehicle_boxes
    
    def _validate_detection(self, box) -> bool:
        """Enhanced validation for a single detection"""
        try:
            # Check required attributes
            if not all(hasattr(box, attr) for attr in ['cls', 'xyxy', 'conf']):
                return False
            
            # Check confidence
            if len(box.conf) == 0 or float(box.conf[0]) < self.min_confidence:
                return False
            
            # Check class
            if len(box.cls) == 0 or int(box.cls[0]) not in self.vehicle_classes:
                return False
            
            # Check coordinates
            if len(box.xyxy) == 0 or len(box.xyxy[0]) < 4:
                return False
            
            # Check box size
            x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
            width = x2 - x1
            height = y2 - y1
            
            return width >= self.min_vehicle_size or height >= self.min_vehicle_size
            
        except Exception:
            return False
    
    def _check_for_accident(self, vehicle_boxes: List) -> Tuple[bool, float, float]:
        """Enhanced check for accident detection with multiple criteria"""
        if len(vehicle_boxes) < self.min_vehicles:
            return False, 0.0, 0.0
        
        try:
            # Calculate overlap
            overlap = self._calculate_overlap(vehicle_boxes)
            
            # Check high confidence vehicles
            high_confidence_vehicles = sum(1 for box in vehicle_boxes 
                                        if hasattr(box, 'conf') and 
                                        len(box.conf) > 0 and 
                                        float(box.conf[0]) >= self.min_confidence_for_accident)
            
            # Check vehicle positions and velocities
            position_check = self._check_vehicle_positions(vehicle_boxes)
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(vehicle_boxes, overlap, high_confidence_vehicles)
            
            # Determine if accident occurred
            is_accident = (overlap >= self.min_overlap and 
                         high_confidence_vehicles >= 2 and 
                         position_check and 
                         confidence >= self.min_confidence_for_accident)
            
            return is_accident, overlap, confidence
            
        except Exception as e:
            self.logger.error(f"Error in accident detection: {str(e)}")
            return False, 0.0, 0.0
    
    def _calculate_confidence(self, boxes: List, overlap: float, high_confidence_vehicles: int) -> float:
        """Calculate overall confidence score for accident detection"""
        try:
            # Base confidence on overlap
            confidence = min(1.0, overlap * 1.5)
            
            # Adjust based on number of high confidence vehicles
            confidence *= (high_confidence_vehicles / len(boxes))
            
            # Adjust based on vehicle types
            vehicle_types = set()
            for box in boxes:
                if hasattr(box, 'cls') and len(box.cls) > 0:
                    cls = int(box.cls[0])
                    if cls in self.vehicle_classes:
                        vehicle_types.add(self.vehicle_classes[cls])
            
            # Higher confidence for multiple vehicle types
            if len(vehicle_types) > 1:
                confidence *= 1.2
            
            # Adjust based on velocities if available
            if hasattr(boxes[0], 'id') and boxes[0].id in self.track_history:
                velocities = []
                for box in boxes:
                    if hasattr(box, 'id') and box.id in self.track_history:
                        velocity = self._calculate_velocity(box.id)
                        if velocity is not None:
                            velocities.append(velocity)
                
                if velocities:
                    try:
                        speed = np.mean([np.sqrt(vx**2 + vy**2) for vx, vy in velocities])
                        confidence *= min(1.0, speed / 50)  # Normalize speed
                    except Exception:
                        pass
            
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0
    
    def _calculate_severity(self, boxes: List, overlap: float, confidence: float) -> Dict:
        """Calculate severity with enhanced metrics"""
        try:
            if not boxes:
                return {
                    "level": SeverityLevel.NONE.value,
                    "severity_score": 0.0,
                    "vehicle_count": 0,
                    "overlap": 0.0
                }
            
            # Base severity on multiple factors
            vehicle_count = len(boxes)
            severity_score = min(1.0, (vehicle_count / 4) * 0.3 + overlap * 0.4 + confidence * 0.3)
            
            # Adjust based on vehicle types
            vehicle_types = []
            for box in boxes:
                try:
                    if hasattr(box, 'cls') and len(box.cls) > 0:
                        cls = int(box.cls[0])
                        if cls in self.vehicle_classes:
                            vehicle_types.append(self.vehicle_classes[cls])
                except (IndexError, ValueError, AttributeError):
                    continue
            
            # Higher severity for larger vehicles
            if "truck" in vehicle_types or "bus" in vehicle_types:
                severity_score = min(1.0, severity_score + 0.2)
            
            # Adjust based on velocities
            if hasattr(boxes[0], 'id') and boxes[0].id in self.track_history:
                velocities = []
                for box in boxes:
                    try:
                        if hasattr(box, 'id') and box.id in self.track_history:
                            velocity = self._calculate_velocity(box.id)
                            if velocity is not None:
                                velocities.append(velocity)
                    except Exception:
                        continue
                
                if velocities:
                    try:
                        speed = np.mean([np.sqrt(vx**2 + vy**2) for vx, vy in velocities])
                        severity_score = min(1.0, severity_score + speed / 100)
                    except Exception:
                        pass
            
            # Determine severity level
            if severity_score < 0.3:
                level = SeverityLevel.MINOR
            elif severity_score < 0.7:
                level = SeverityLevel.MODERATE
            else:
                level = SeverityLevel.SEVERE
            
            return {
                "level": level.value,
                "severity_score": severity_score,
                "vehicle_count": vehicle_count,
                "overlap": overlap,
                "vehicle_types": vehicle_types,
                "confidence": confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating severity: {str(e)}")
            return {
                "level": SeverityLevel.NONE.value,
                "severity_score": 0.0,
                "vehicle_count": 0,
                "overlap": 0.0,
                "vehicle_types": [],
                "confidence": 0.0
            }
    
    def _check_vehicle_positions(self, boxes: List) -> bool:
        """Enhanced check for vehicle positions indicating a potential accident"""
        if len(boxes) < 2:
            return False
        
        try:
            # Calculate center points and velocities
            centers = []
            velocities = []
            valid_boxes = []
            
            for box in boxes:
                if not self._validate_detection(box):
                    continue
                    
                x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                centers.append((center_x, center_y))
                valid_boxes.append(box)
                
                # Calculate velocity if tracking data available
                if hasattr(box, 'id') and box.id in self.track_history:
                    velocity = self._calculate_velocity(box.id)
                    if velocity is not None:
                        velocities.append(velocity)
            
            if len(valid_boxes) < 2:
                return False
            
            # Check vehicle distances
            min_distance = float('inf')
            for i, (x1, y1) in enumerate(centers):
                for x2, y2 in centers[i+1:]:
                    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    min_distance = min(min_distance, distance)
            
            # Check vehicle convergence
            is_converging = False
            if len(velocities) >= 2:
                relative_velocity = np.array(velocities[0]) - np.array(velocities[1])
                is_converging = np.dot(relative_velocity, np.array(centers[1]) - np.array(centers[0])) < 0
            
            # Check speed difference
            speed_difference_valid = True
            if len(velocities) >= 2:
                speed1 = np.sqrt(velocities[0][0]**2 + velocities[0][1]**2)
                speed2 = np.sqrt(velocities[1][0]**2 + velocities[1][1]**2)
                speed_difference_valid = abs(speed1 - speed2) <= self.max_speed_difference
            
            return (min_distance < self.max_distance_threshold and 
                   is_converging and 
                   self._check_velocity_threshold(velocities) and
                   speed_difference_valid)
            
        except Exception as e:
            self.logger.error(f"Error in vehicle position check: {str(e)}")
            return False
    
    def _calculate_velocity(self, track_id: int) -> Optional[Tuple[float, float]]:
        """Calculate vehicle velocity from tracking history"""
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            return None
        
        try:
            history = self.track_history[track_id]
            dx = history[-1][0] - history[-2][0]
            dy = history[-1][1] - history[-2][1]
            
            # Apply smoothing to reduce noise
            if len(history) >= 3:
                dx = (dx + (history[-2][0] - history[-3][0])) / 2
                dy = (dy + (history[-2][1] - history[-3][1])) / 2
            
            return (dx, dy)
        except Exception:
            return None
    
    def _check_velocity_threshold(self, velocities: List) -> bool:
        """Check if vehicle velocities exceed threshold"""
        if not velocities:
            return False
        
        try:
            # Calculate average velocity magnitude
            velocity_magnitudes = [np.sqrt(vx**2 + vy**2) for vx, vy in velocities]
            avg_velocity = np.mean(velocity_magnitudes)
            
            return avg_velocity > self.min_velocity_threshold
            
        except Exception:
            return False
    
    def _update_tracking(self, boxes: List) -> None:
        """Update tracking history with enhanced cleanup"""
        self.frame_counter += 1
        
        # Cleanup old tracking data periodically
        if self.frame_counter % self.track_cleanup_interval == 0:
            self._cleanup_tracking_history()
        
        # Update tracking for current frame
        for box in boxes:
            if hasattr(box, 'id'):
                track_id = box.id
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                
                # Store box coordinates and timestamp
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                self.track_history[track_id].append((center_x, center_y, self.frame_counter))
                
                # Keep only recent history
                if len(self.track_history[track_id]) > self.max_track_history:
                    self.track_history[track_id] = self.track_history[track_id][-self.max_track_history:]
    
    def _cleanup_tracking_history(self) -> None:
        """Remove old tracking data"""
        current_time = self.frame_counter
        self.track_history = {
            track_id: history for track_id, history in self.track_history.items()
            if current_time - history[-1][2] < self.max_track_history
        }
    
    def _calculate_overlap(self, boxes: List) -> float:
        """Calculate maximum overlap between vehicle bounding boxes"""
        if len(boxes) < 2:
            return 0.0
            
        max_overlap = 0.0
        for i, box1 in enumerate(boxes):
            for box2 in boxes[i+1:]:
                # Convert boxes to tensor format
                box1_tensor = torch.tensor(box1.xyxy[0]).unsqueeze(0)
                box2_tensor = torch.tensor(box2.xyxy[0]).unsqueeze(0)
                
                # Calculate IoU using PyTorch
                iou = box_iou(box1_tensor, box2_tensor).item()
                max_overlap = max(max_overlap, iou)
        
        return max_overlap 