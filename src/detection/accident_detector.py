import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Union, Tuple
import os

class AccidentDetector:
    def __init__(self):
        """Initialize the accident detection system"""
        # Load YOLOv8 model
        self.yolo_model = YOLO('yolov8n.pt')  # Using smaller model for faster inference
        
        # Vehicle classes
        self.vehicle_classes = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }
        
        # Detection thresholds
        self.min_confidence = 0.5
        self.min_vehicles = 2
        self.min_overlap = 0.3
    
    def process_image(self, image_path: str) -> Dict:
        """
        Process a single image for accident detection
        
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
            results = self.yolo_model(img, conf=self.min_confidence)
            
            # Process detection results
            if isinstance(results, list):
                if not results:
                    return {"accident_detected": False, "detections": []}
                results = results[0]
            
            # Get vehicle detections
            vehicle_boxes = []
            for box in results.boxes:
                cls = int(box.cls)
                if cls in self.vehicle_classes:
                    vehicle_boxes.append(box)
            
            # Check for potential accident
            is_accident = False
            overlap = 0.0
            if len(vehicle_boxes) >= self.min_vehicles:
                # Calculate overlap between vehicles
                overlap = self._calculate_overlap(vehicle_boxes)
                if overlap >= self.min_overlap:
                    is_accident = True
            
            # Calculate severity
            severity = self._calculate_severity(vehicle_boxes, overlap)
            
            # Prepare result
            result = {
                "accident_detected": is_accident,
                "detections": vehicle_boxes,
                "vehicle_count": len(vehicle_boxes),
                "overlap": overlap,
                "severity": severity
            }
            
            return result
            
        except Exception as e:
            return {"accident_detected": False, "error": str(e)}
    
    def process_video(self, video_path: str) -> Dict:
        """
        Process a video file for accident detection
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing detection results
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"accident_detected": False, "error": "Failed to open video"}
            
            frames = []
            frame_number = 0
            accident_frames = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 5th frame
                if frame_number % 5 == 0:
                    # Save frame to temporary file
                    temp_path = f"temp_frame_{frame_number}.jpg"
                    cv2.imwrite(temp_path, frame)
                    
                    # Process frame
                    result = self.process_image(temp_path)
                    
                    if result["accident_detected"]:
                        accident_frames.append({
                            "frame": frame,
                            "frame_number": frame_number,
                            "result": result
                        })
                    
                    # Clean up
                    os.remove(temp_path)
                
                frame_number += 1
            
            cap.release()
            
            # Determine if accident was detected
            is_accident = len(accident_frames) > 0
            
            # Calculate overall severity
            severity = self._calculate_overall_severity(accident_frames)
            
            return {
                "accident_detected": is_accident,
                "frames": accident_frames,
                "severity": severity
            }
            
        except Exception as e:
            return {"accident_detected": False, "error": str(e)}
    
    def _calculate_overlap(self, boxes: List) -> float:
        """Calculate maximum overlap between vehicle bounding boxes"""
        if len(boxes) < 2:
            return 0.0
            
        max_overlap = 0.0
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
    
    def _calculate_severity(self, boxes: List, overlap: float) -> Dict:
        """Calculate severity of potential accident"""
        # Base severity on number of vehicles and overlap
        vehicle_count = len(boxes)
        severity_score = min(1.0, (vehicle_count / 4) * 0.5 + overlap * 0.5)
        
        # Determine severity level
        if severity_score < 0.3:
            level = "Minor"
        elif severity_score < 0.7:
            level = "Moderate"
        else:
            level = "Severe"
        
        return {
            "level": level,
            "severity_score": severity_score,
            "vehicle_count": vehicle_count,
            "overlap": overlap
        }
    
    def _calculate_overall_severity(self, accident_frames: List) -> Dict:
        """Calculate overall severity from multiple frames"""
        if not accident_frames:
            return {
                "level": "Unknown",
                "severity_score": 0.0
            }
        
        # Calculate average severity
        total_severity = sum(frame["result"]["severity"]["severity_score"] for frame in accident_frames)
        avg_severity = total_severity / len(accident_frames)
        
        # Determine severity level
        if avg_severity < 0.3:
            level = "Minor"
        elif avg_severity < 0.7:
            level = "Moderate"
        else:
            level = "Severe"
        
        return {
            "level": level,
            "severity_score": avg_severity,
            "frame_count": len(accident_frames)
        } 