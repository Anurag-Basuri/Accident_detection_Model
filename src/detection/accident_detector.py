import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Union, Tuple
import os
from collections import defaultdict

class AccidentDetector:
    def __init__(self):
        """Initialize the accident detection system"""
        # Load YOLOv8 model
        self.yolo_model = YOLO('yolov8x.pt')
        
        # Load your image model
        self.image_model_path = os.path.join(os.path.dirname(__file__), "../../models/image_model.h5")
        self.image_model = self._load_image_model()
        
        # Vehicle classes with base values and characteristics
        self.vehicle_classes = {
            2: {"name": "car", "base_value": 25000, "damage_factor": 1.0, "size_factor": 1.0},
            3: {"name": "motorcycle", "base_value": 8000, "damage_factor": 0.8, "size_factor": 0.5},
            5: {"name": "bus", "base_value": 100000, "damage_factor": 1.5, "size_factor": 2.0},
            7: {"name": "truck", "base_value": 80000, "damage_factor": 1.3, "size_factor": 1.8}
        }
        
        # Severity thresholds
        self.severity_thresholds = {
            "minor": {"vehicle_count": 2, "speed_factor": 0.3, "overlap_factor": 0.3},
            "moderate": {"vehicle_count": 4, "speed_factor": 0.6, "overlap_factor": 0.6},
            "severe": {"vehicle_count": 6, "speed_factor": 0.8, "overlap_factor": 0.8}
        }
        
        # Detection thresholds
        self.min_confidence = 0.5
        self.min_vehicles = 2
        self.min_overlap = 0.3
        
        # Advanced detection parameters
        self.impact_thresholds = {
            "high_speed": 60,  # km/h
            "sudden_stop": 0.7,  # speed reduction ratio
            "multiple_vehicles": 3,  # number of vehicles
            "severe_overlap": 0.5,  # overlap ratio
            "close_proximity": 0.2  # proximity ratio
        }
        
        # Vehicle tracking
        self.tracked_vehicles = defaultdict(dict)
        self.frame_rate = 30  # Default frame rate, will be updated from video
        
    def _load_image_model(self):
        """Load your existing image model"""
        import tensorflow as tf
        return tf.keras.models.load_model(self.image_model_path)
    
    def process_image(self, image_path: str) -> Dict:
        """
        Process a single image for accident detection with enhanced analysis
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing detection results
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return {"accident_detected": False, "error": "Failed to read image"}
            
            # Run YOLO detection
            results = self.yolo_model(img)
            
            # Process detection results
            if isinstance(results, list):
                if not results:
                    return {"accident_detected": False, "detections": []}
                results = results[0]
            
            # Get vehicle detections
            vehicle_boxes = []
            for box in results.boxes:
                cls = int(box.cls)
                conf = float(box.conf[0])
                if cls in self.vehicle_classes and conf >= self.min_confidence:
                    vehicle_boxes.append(box)
            
            # Enhanced accident detection
            is_accident = False
            accident_factors = {
                "vehicle_count": len(vehicle_boxes),
                "overlap": 0,
                "proximity": 0,
                "impact_angle": 0,
                "severity_score": 0
            }
            
            if len(vehicle_boxes) >= self.min_vehicles:
                # Calculate various factors
                overlap = self._calculate_overlap(vehicle_boxes)
                proximity = self._calculate_proximity(vehicle_boxes)
                impact_angle = self._calculate_impact_angle(vehicle_boxes)
                
                # Update factors
                accident_factors["overlap"] = overlap
                accident_factors["proximity"] = proximity
                accident_factors["impact_angle"] = impact_angle
                
                # Calculate severity score
                severity_score = (
                    (len(vehicle_boxes) / self.impact_thresholds["multiple_vehicles"]) * 0.3 +
                    overlap * 0.3 +
                    proximity * 0.2 +
                    (impact_angle / 180) * 0.2
                )
                accident_factors["severity_score"] = severity_score
                
                # Determine if accident occurred
                is_accident = (
                    len(vehicle_boxes) >= self.impact_thresholds["multiple_vehicles"] or
                    overlap >= self.impact_thresholds["severe_overlap"] or
                    (proximity >= self.impact_thresholds["close_proximity"] and 
                     impact_angle > 45)  # Significant angle change
                )
            
            # Prepare result
            result = {
                "accident_detected": is_accident,
                "detections": vehicle_boxes,
                "factors": accident_factors,
                "severity": self._determine_severity(accident_factors)
            }
            
            return result
            
        except Exception as e:
            return {"accident_detected": False, "error": str(e)}
    
    def process_video(self, video_path: str) -> Dict:
        """
        Process video using YOLO model with enhanced detection logic
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing detection results
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"accident_detected": False, "error": "Failed to open video"}
            
            self.frame_rate = cap.get(cv2.CAP_PROP_FPS)
            self.tracked_vehicles.clear()
            
            frames = []
            frame_number = 0
            accident_frames = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 5th frame to save computation
                if frame_number % 5 == 0:
                    # Run YOLOv8 detection
                    results = self.yolo_model(frame)
                    
                    # Check for vehicles and potential accidents
                    if self._check_vehicles(results):
                        # Enhanced accident detection
                        is_accident = self._detect_accident_in_frame(results, frame_number)
                        
                        if is_accident:
                            accident_frames.append({
                                "frame_number": frame_number,
                                "detections": results,
                                "frame": frame.copy()
                            })
                
                frame_number += 1
            
            cap.release()
            
            # Analyze frames for accidents
            if accident_frames:
                # Group consecutive accident frames
                grouped_frames = self._group_consecutive_frames(accident_frames)
                
                # Analyze each group
                for group in grouped_frames:
                    severity = self._calculate_severity(group)
                    insurance = self._assess_insurance(group)
                    
                    if severity["level"] != "Unknown":
                        return {
                            "accident_detected": True,
                            "severity": severity,
                            "insurance": insurance,
                            "frames": group
                        }
            
            return {"accident_detected": False}
            
        except Exception as e:
            return {"accident_detected": False, "error": str(e)}
    
    def _detect_accident_in_frame(self, results, frame_number: int) -> bool:
        """Enhanced accident detection in a single frame"""
        # Check vehicle count
        vehicle_count = sum(1 for box in results.boxes if int(box.cls) in self.vehicle_classes)
        if vehicle_count < self.min_vehicles:
            return False
        
        # Check for vehicle overlap
        overlap = self._calculate_overlap(results.boxes)
        if overlap < self.min_overlap:
            return False
        
        # Check for high speeds
        speeds = self._get_vehicle_speeds(results.boxes, frame_number)
        if any(speed > 80 for speed in speeds.values()):  # High speed threshold
            return True
        
        # Check for multiple vehicles in close proximity
        if vehicle_count >= 3 and overlap > 0.5:
            return True
        
        return False
    
    def _group_consecutive_frames(self, frames: List[Dict]) -> List[List[Dict]]:
        """Group consecutive frames where accidents are detected"""
        if not frames:
            return []
        
        groups = []
        current_group = [frames[0]]
        
        for i in range(1, len(frames)):
            if frames[i]["frame_number"] - frames[i-1]["frame_number"] <= 10:  # Within 10 frames
                current_group.append(frames[i])
            else:
                if len(current_group) >= 3:  # Minimum 3 frames for confirmation
                    groups.append(current_group)
                current_group = [frames[i]]
        
        if len(current_group) >= 3:
            groups.append(current_group)
        
        return groups
    
    def _get_vehicle_speeds(self, boxes, frame_number: int) -> Dict[str, float]:
        """Get speeds of detected vehicles"""
        speeds = {}
        for box in boxes:
            cls = int(box.cls)
            if cls in self.vehicle_classes:
                vehicle_type = self.vehicle_classes[cls]["name"]
                speed = self._estimate_speed(box, frame_number)
                speeds[vehicle_type] = speed
        return speeds
    
    def _check_vehicles(self, results) -> bool:
        """Check if there are vehicles in the frame"""
        vehicle_count = sum(1 for box in results.boxes 
                          if int(box.cls) in self.vehicle_classes 
                          and float(box.conf[0]) > self.min_confidence)
        return vehicle_count >= self.min_vehicles
    
    def visualize_detections(self, frame, yolo_results, accident_detected=False):
        """
        Draw bounding boxes and labels on the frame
        
        Args:
            frame: Input frame
            yolo_results: YOLOv8 detection results
            accident_detected: Whether an accident was detected
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw vehicle bounding boxes
        for result in yolo_results:
            for box in result.boxes:
                cls = int(box.cls)
                if cls in self.vehicle_classes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw rectangle
                    color = (0, 0, 255) if accident_detected else (0, 255, 0)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f"{self.vehicle_classes[cls]['name']} {box.conf[0]:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add accident alert if detected
        if accident_detected:
            cv2.putText(annotated_frame, "ACCIDENT DETECTED!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return annotated_frame
    
    def _calculate_severity(self, frames: List[Dict]) -> Dict:
        """Calculate accident severity with detailed analysis"""
        if not frames:
            return {"level": "Unknown", "factors": {}}
            
        # Initialize metrics
        vehicle_counts = {}
        max_speeds = {}
        overlap_areas = []
        
        # Analyze each frame
        for frame in frames:
            frame_vehicles = {}
            
            for result in frame["detections"]:
                for box in result.boxes:
                    cls = int(box.cls)
                    if cls in self.vehicle_classes:
                        # Count vehicles
                        vehicle_type = self.vehicle_classes[cls]["name"]
                        frame_vehicles[vehicle_type] = frame_vehicles.get(vehicle_type, 0) + 1
                        
                        # Calculate speed (simple approximation)
                        if len(frames) > 1:
                            speed = self._estimate_speed(box, frame["frame_number"])
                            max_speeds[vehicle_type] = max(max_speeds.get(vehicle_type, 0), speed)
            
            # Update vehicle counts
            for v_type, count in frame_vehicles.items():
                vehicle_counts[v_type] = max(vehicle_counts.get(v_type, 0), count)
            
            # Calculate overlap areas
            overlap = self._calculate_overlap(result.boxes)
            overlap_areas.append(overlap)
        
        # Calculate severity factors
        total_vehicles = sum(vehicle_counts.values())
        avg_speed = np.mean(list(max_speeds.values())) if max_speeds else 0
        max_overlap = max(overlap_areas) if overlap_areas else 0
        
        # Determine severity level
        severity_score = (
            (total_vehicles / self.severity_thresholds["severe"]["vehicle_count"]) * 0.4 +
            (avg_speed / 100) * 0.3 +  # Assuming max speed of 100
            max_overlap * 0.3
        )
        
        if severity_score < 0.3:
            level = "Minor"
        elif severity_score < 0.7:
            level = "Moderate"
        else:
            level = "Severe"
        
        return {
            "level": level,
            "factors": {
                "vehicle_count": total_vehicles,
                "vehicle_types": vehicle_counts,
                "max_speeds": max_speeds,
                "max_overlap": max_overlap,
                "severity_score": severity_score
            }
        }
    
    def _assess_insurance(self, frames: List[Dict]) -> Dict:
        """Assess insurance details with vehicle-specific calculations"""
        if not frames:
            return {"estimated_damage": 0, "vehicle_count": 0}
            
        # Initialize damage assessment
        total_damage = 0
        vehicle_details = {}
        
        # Analyze each frame
        for frame in frames:
            for result in frame["detections"]:
                for box in result.boxes:
                    cls = int(box.cls)
                    if cls in self.vehicle_classes:
                        vehicle_type = self.vehicle_classes[cls]["name"]
                        
                        # Calculate damage based on vehicle type and confidence
                        base_value = self.vehicle_classes[cls]["base_value"]
                        damage_factor = self.vehicle_classes[cls]["damage_factor"]
                        confidence = float(box.conf[0])
                        
                        # Damage calculation
                        damage = base_value * damage_factor * confidence
                        
                        # Update vehicle details
                        if vehicle_type not in vehicle_details:
                            vehicle_details[vehicle_type] = {
                                "count": 0,
                                "total_damage": 0,
                                "max_damage": 0
                            }
                        
                        vehicle_details[vehicle_type]["count"] += 1
                        vehicle_details[vehicle_type]["total_damage"] += damage
                        vehicle_details[vehicle_type]["max_damage"] = max(
                            vehicle_details[vehicle_type]["max_damage"],
                            damage
                        )
        
        # Calculate total damage
        total_damage = sum(v["total_damage"] for v in vehicle_details.values())
        total_vehicles = sum(v["count"] for v in vehicle_details.values())
        
        # Calculate repair estimate (70-90% of total damage)
        repair_estimate = total_damage * np.random.uniform(0.7, 0.9)
        
        return {
            "estimated_damage": total_damage,
            "vehicle_count": total_vehicles,
            "repair_estimate": repair_estimate,
            "vehicle_details": vehicle_details
        }
    
    def _estimate_speed(self, box, frame_number) -> float:
        """Estimate vehicle speed based on bounding box movement"""
        cls = int(box.cls)
        if cls not in self.vehicle_classes:
            return 0.0
            
        # Get current box center
        x1, y1, x2, y2 = box.xyxy[0]
        current_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Get previous position if exists
        vehicle_id = f"{cls}_{frame_number}"
        if vehicle_id in self.tracked_vehicles:
            prev_center = self.tracked_vehicles[vehicle_id]["center"]
            prev_frame = self.tracked_vehicles[vehicle_id]["frame"]
            
            # Calculate pixel distance moved
            pixel_distance = np.sqrt(
                (current_center[0] - prev_center[0])**2 + 
                (current_center[1] - prev_center[1])**2
            )
            
            # Calculate time difference
            time_diff = (frame_number - prev_frame) / self.frame_rate
            
            # Convert to speed (pixels per second)
            speed = pixel_distance / time_diff if time_diff > 0 else 0
            
            # Convert to km/h (assuming 1 pixel = 0.1 meters)
            speed_kmh = speed * 0.1 * 3.6
            
            # Update tracking
            self.tracked_vehicles[vehicle_id] = {
                "center": current_center,
                "frame": frame_number,
                "speed": speed_kmh
            }
            
            return speed_kmh
        else:
            # First detection of this vehicle
            self.tracked_vehicles[vehicle_id] = {
                "center": current_center,
                "frame": frame_number,
                "speed": 0.0
            }
            return 0.0
    
    def _calculate_overlap(self, boxes):
        """Calculate maximum overlap between vehicle bounding boxes"""
        if len(boxes) < 2:
            return 0
            
        max_overlap = 0
        for i, box1 in enumerate(boxes):
            for box2 in boxes[i+1:]:
                # Get box coordinates
                x1 = max(box1.xyxy[0][0], box2.xyxy[0][0])
                y1 = max(box1.xyxy[0][1], box2.xyxy[0][1])
                x2 = min(box1.xyxy[0][2], box2.xyxy[0][2])
                y2 = min(box1.xyxy[0][3], box2.xyxy[0][3])
                
                if x2 > x1 and y2 > y1:
                    # Calculate intersection area
                    intersection = (x2 - x1) * (y2 - y1)
                    
                    # Calculate areas of both boxes
                    area1 = (box1.xyxy[0][2] - box1.xyxy[0][0]) * (box1.xyxy[0][3] - box1.xyxy[0][1])
                    area2 = (box2.xyxy[0][2] - box2.xyxy[0][0]) * (box2.xyxy[0][3] - box2.xyxy[0][1])
                    
                    # Calculate IoU
                    iou = intersection / (area1 + area2 - intersection)
                    max_overlap = max(max_overlap, iou)
        
        return max_overlap
    
    def _calculate_proximity(self, boxes):
        """Calculate proximity between vehicles"""
        if len(boxes) < 2:
            return 0
            
        min_distance = float('inf')
        for i, box1 in enumerate(boxes):
            for box2 in boxes[i+1:]:
                # Get box centers
                x1, y1, x2, y2 = box1.xyxy[0]
                center1 = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                x1, y1, x2, y2 = box2.xyxy[0]
                center2 = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                # Calculate distance
                distance = np.sqrt(
                    (center1[0] - center2[0])**2 + 
                    (center1[1] - center2[1])**2
                )
                
                # Get average box size
                size1 = np.sqrt((box1.xyxy[0][2] - box1.xyxy[0][0])**2 + 
                              (box1.xyxy[0][3] - box1.xyxy[0][1])**2)
                size2 = np.sqrt((box2.xyxy[0][2] - box2.xyxy[0][0])**2 + 
                              (box2.xyxy[0][3] - box2.xyxy[0][1])**2)
                avg_size = (size1 + size2) / 2
                
                # Normalize distance
                normalized_distance = distance / avg_size
                min_distance = min(min_distance, normalized_distance)
        
        # Convert to proximity score (closer = higher score)
        return 1.0 - min(min_distance, 1.0)
    
    def _calculate_impact_angle(self, boxes):
        """Calculate potential impact angle between vehicles"""
        if len(boxes) < 2:
            return 0
            
        max_angle = 0
        for i, box1 in enumerate(boxes):
            for box2 in boxes[i+1:]:
                # Get box orientations
                x1, y1, x2, y2 = box1.xyxy[0]
                angle1 = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                x1, y1, x2, y2 = box2.xyxy[0]
                angle2 = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # Calculate angle difference
                angle_diff = abs(angle1 - angle2)
                angle_diff = min(angle_diff, 360 - angle_diff)
                
                max_angle = max(max_angle, angle_diff)
        
        return max_angle
    
    def _determine_severity(self, factors):
        """Determine accident severity based on multiple factors"""
        severity_score = factors["severity_score"]
        
        if severity_score < 0.3:
            return {
                "level": "Minor",
                "description": "Minor collision with low impact",
                "color": "green"
            }
        elif severity_score < 0.6:
            return {
                "level": "Moderate",
                "description": "Moderate collision with significant impact",
                "color": "orange"
            }
        else:
            return {
                "level": "Severe",
                "description": "Severe collision with high impact",
                "color": "red"
            } 