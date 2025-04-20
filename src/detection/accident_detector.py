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
        
        # Vehicle classes
        self.vehicle_classes = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
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
                    label = f"{self.vehicle_classes[cls]} {box.conf[0]:.2f}"
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
    
    def _calculate_severity(self, frames: List[Dict]) -> str:
        """Calculate accident severity based on detections"""
        if not frames:
            return "Unknown"
            
        # Simple severity calculation based on vehicle count
        vehicle_count = 0
        for frame in frames:
            for result in frame["detections"]:
                vehicle_count += len([box for box in result.boxes 
                                    if int(box.cls) in self.vehicle_classes])
        
        if vehicle_count <= 2:
            return "Minor"
        elif vehicle_count <= 4:
            return "Moderate"
        else:
            return "Severe"
    
    def _assess_insurance(self, frames: List[Dict]) -> Dict:
        """Assess insurance details based on detections"""
        if not frames:
            return {"estimated_damage": 0, "vehicle_count": 0}
            
        vehicle_count = 0
        for frame in frames:
            for result in frame["detections"]:
                vehicle_count += len([box for box in result.boxes 
                                    if int(box.cls) in self.vehicle_classes])
        
        # Simple damage estimation based on vehicle count
        base_damage = 5000  # Base damage per vehicle
        total_damage = vehicle_count * base_damage
        
        return {
            "estimated_damage": total_damage,
            "vehicle_count": vehicle_count,
            "repair_estimate": total_damage * 0.8  # 80% of total damage
        } 