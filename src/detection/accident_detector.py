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
            if len(vehicle_boxes) >= self.min_vehicles:
                # Calculate overlap between vehicles
                overlap = self._calculate_overlap(vehicle_boxes)
                if overlap >= self.min_overlap:
                    is_accident = True
            
            # Prepare result
            result = {
                "accident_detected": is_accident,
                "detections": vehicle_boxes,
                "vehicle_count": len(vehicle_boxes),
                "overlap": overlap if len(vehicle_boxes) >= 2 else 0
            }
            
            return result
            
        except Exception as e:
            return {"accident_detected": False, "error": str(e)}
    
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