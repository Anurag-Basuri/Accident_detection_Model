import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Union
import os

class AccidentDetector:
    def __init__(self):
        """Initialize the accident detection system"""
        # Load YOLOv8 model
        self.yolo_model = YOLO('yolov8x.pt')
        
        # Load your image model
        self.image_model_path = os.path.join(os.path.dirname(__file__), "../../models/image_model.h5")
        self.image_model = self._load_image_model()
        
        # Vehicle classes with base values
        self.vehicle_classes = {
            2: {"name": "car", "base_value": 25000, "damage_factor": 1.0},
            3: {"name": "motorcycle", "base_value": 8000, "damage_factor": 0.8},
            5: {"name": "bus", "base_value": 100000, "damage_factor": 1.5},
            7: {"name": "truck", "base_value": 80000, "damage_factor": 1.3}
        }
        
        # Severity thresholds
        self.severity_thresholds = {
            "minor": {"vehicle_count": 2, "speed_factor": 0.3, "overlap_factor": 0.3},
            "moderate": {"vehicle_count": 4, "speed_factor": 0.6, "overlap_factor": 0.6},
            "severe": {"vehicle_count": 6, "speed_factor": 0.8, "overlap_factor": 0.8}
        }
        
    def _load_image_model(self):
        """Load your existing image model"""
        import tensorflow as tf
        return tf.keras.models.load_model(self.image_model_path)
    
    def process_image(self, image_path: str) -> Dict:
        """
        Process a single image using your image model
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing detection results
        """
        # Read and preprocess image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = self.image_model.predict(img)
        is_accident = prediction[0][0] > 0.5
        
        return {
            "accident_detected": bool(is_accident),
            "confidence": float(prediction[0][0]),
            "type": "image"
        }
    
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
    
    def process_video(self, video_path: str) -> Dict:
        """
        Process a video using both YOLOv8 and your image model
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing detection results and visualization frames
        """
        cap = cv2.VideoCapture(video_path)
        results = {
            "accident_detected": False,
            "type": "video",
            "frames": [],
            "visualization_frames": [],
            "severity": None,
            "insurance_details": None
        }
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every 5th frame to save computation
            if frame_count % 5 == 0:
                # YOLOv8 detection
                yolo_results = self.yolo_model(frame)
                
                # Check for vehicles
                if self._check_vehicles(yolo_results):
                    # Validate with image model
                    img = cv2.resize(frame, (224, 224))
                    img = img / 255.0
                    img = np.expand_dims(img, axis=0)
                    
                    accident_detected = self.image_model.predict(img)[0][0] > 0.5
                    if accident_detected:
                        results["accident_detected"] = True
                        results["frames"].append({
                            "frame_number": frame_count,
                            "detections": yolo_results
                        })
                
                # Visualize detections
                vis_frame = self.visualize_detections(frame, yolo_results, 
                                                    results["accident_detected"])
                results["visualization_frames"].append(vis_frame)
            
            frame_count += 1
            
        cap.release()
        
        # If accident detected, analyze severity
        if results["accident_detected"]:
            results["severity"] = self._calculate_severity(results["frames"])
            results["insurance_details"] = self._assess_insurance(results["frames"])
            
        return results
    
    def _check_vehicles(self, yolo_results) -> bool:
        """Check if there are vehicles in the frame"""
        for result in yolo_results:
            for box in result.boxes:
                if int(box.cls) in self.vehicle_classes:
                    return True
        return False
    
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
    
    def _estimate_speed(self, box, frame_number):
        """Simple speed estimation based on bounding box movement"""
        # This is a placeholder - implement proper speed estimation
        return np.random.uniform(0, 100)  # Random speed for demonstration
    
    def _calculate_overlap(self, boxes):
        """Calculate maximum overlap between vehicle bounding boxes"""
        if len(boxes) < 2:
            return 0
            
        max_overlap = 0
        for i, box1 in enumerate(boxes):
            for box2 in boxes[i+1:]:
                if int(box1.cls) in self.vehicle_classes and int(box2.cls) in self.vehicle_classes:
                    # Calculate IoU
                    x1 = max(box1.xyxy[0][0], box2.xyxy[0][0])
                    y1 = max(box1.xyxy[0][1], box2.xyxy[0][1])
                    x2 = min(box1.xyxy[0][2], box2.xyxy[0][2])
                    y2 = min(box1.xyxy[0][3], box2.xyxy[0][3])
                    
                    if x2 > x1 and y2 > y1:
                        intersection = (x2 - x1) * (y2 - y1)
                        area1 = (box1.xyxy[0][2] - box1.xyxy[0][0]) * (box1.xyxy[0][3] - box1.xyxy[0][1])
                        area2 = (box2.xyxy[0][2] - box2.xyxy[0][0]) * (box2.xyxy[0][3] - box2.xyxy[0][1])
                        iou = intersection / (area1 + area2 - intersection)
                        max_overlap = max(max_overlap, iou)
        
        return max_overlap 