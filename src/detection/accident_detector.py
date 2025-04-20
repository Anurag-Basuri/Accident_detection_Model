import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Union, Tuple
import os
import torch
from torchvision.ops import box_iou

class AccidentDetector:
    def __init__(self):
        """Initialize the accident detection system"""
        # Load YOLOv8 model with GPU acceleration if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo_model = YOLO('yolov8x.pt').to(self.device)  # Using larger model for better accuracy
        
        # Vehicle classes with more specific types
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
        self.min_confidence_for_accident = 0.7
        self.min_vehicle_size = 100  # Minimum pixel size for a vehicle
        
        # Motion analysis parameters
        self.motion_threshold = 10  # Minimum motion magnitude
        self.prev_frame = None
        self.prev_detections = None
        
        # Initialize tracking
        self.track_history = {}
        self.max_track_history = 30  # Number of frames to keep in history
    
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
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Run YOLO detection with tracking
            results = self.yolo_model.track(img_rgb, 
                                         conf=self.min_confidence,
                                         persist=True,
                                         classes=list(self.vehicle_classes.keys()))
            
            # Process detection results
            if isinstance(results, list):
                if not results:
                    return {"accident_detected": False, "detections": []}
                results = results[0]
            
            # Get vehicle detections with additional filtering
            vehicle_boxes = []
            for box in results.boxes:
                cls = int(box.cls)
                if cls in self.vehicle_classes:
                    # Check vehicle size
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    width = x2 - x1
                    height = y2 - y1
                    if width >= self.min_vehicle_size or height >= self.min_vehicle_size:
                        vehicle_boxes.append(box)
            
            # Check for potential accident
            is_accident = False
            overlap = 0.0
            if len(vehicle_boxes) >= self.min_vehicles:
                # Calculate overlap between vehicles
                overlap = self._calculate_overlap(vehicle_boxes)
                
                # Check for high confidence detections
                high_confidence_vehicles = sum(1 for box in vehicle_boxes 
                                            if float(box.conf[0]) >= self.min_confidence_for_accident)
                
                # Additional checks for accident detection
                if (overlap >= self.min_overlap and 
                    high_confidence_vehicles >= 2 and 
                    self._check_vehicle_positions(vehicle_boxes) and
                    self._check_motion(img_rgb)):
                    is_accident = True
            
            # Calculate severity
            severity = self._calculate_severity(vehicle_boxes, overlap)
            
            # Update tracking history
            self._update_tracking(vehicle_boxes)
            
            # Prepare result
            result = {
                "accident_detected": is_accident,
                "detections": vehicle_boxes,
                "vehicle_count": len(vehicle_boxes),
                "overlap": overlap,
                "severity": severity,
                "tracking_data": self.track_history
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
                    
                    # Check for temporal consistency
                    if result["accident_detected"]:
                        if self._check_temporal_consistency(result, frame):
                            accident_frames.append({
                                "frame": frame,
                                "frame_number": frame_number,
                                "result": result
                            })
                    
                    # Update previous frame and detections
                    self.prev_frame = frame
                    self.prev_detections = result["detections"]
                    
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
                # Convert boxes to tensor format
                box1_tensor = torch.tensor(box1.xyxy[0]).unsqueeze(0)
                box2_tensor = torch.tensor(box2.xyxy[0]).unsqueeze(0)
                
                # Calculate IoU using PyTorch
                iou = box_iou(box1_tensor, box2_tensor).item()
                max_overlap = max(max_overlap, iou)
        
        return max_overlap
    
    def _check_vehicle_positions(self, boxes: List) -> bool:
        """Check if vehicle positions indicate a potential accident"""
        if len(boxes) < 2:
            return False
        
        # Calculate center points and velocities of vehicles
        centers = []
        velocities = []
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append((center_x, center_y))
            
            # Calculate velocity if tracking data available
            if hasattr(box, 'id') and box.id in self.track_history:
                velocity = self._calculate_velocity(box.id)
                velocities.append(velocity)
        
        # Check if vehicles are close to each other
        min_distance = float('inf')
        for i, (x1, y1) in enumerate(centers):
            for x2, y2 in centers[i+1:]:
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                min_distance = min(min_distance, distance)
        
        # Check if vehicles are moving towards each other
        is_converging = False
        if len(velocities) >= 2:
            relative_velocity = np.array(velocities[0]) - np.array(velocities[1])
            is_converging = np.dot(relative_velocity, np.array(centers[1]) - np.array(centers[0])) < 0
        
        # If vehicles are very close and converging, it might indicate an accident
        return min_distance < 100 and is_converging
    
    def _check_motion(self, current_frame: np.ndarray) -> bool:
        """Check for significant motion in the scene"""
        if self.prev_frame is None:
            self.prev_frame = current_frame
            return False
        
        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_RGB2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Calculate motion magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        mean_magnitude = np.mean(magnitude)
        
        # Update previous frame
        self.prev_frame = current_frame
        
        return mean_magnitude > self.motion_threshold
    
    def _update_tracking(self, boxes: List) -> None:
        """Update vehicle tracking history"""
        for box in boxes:
            if hasattr(box, 'id'):
                if box.id not in self.track_history:
                    self.track_history[box.id] = []
                
                # Add current position to history
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                self.track_history[box.id].append((center_x, center_y))
                
                # Keep only recent history
                if len(self.track_history[box.id]) > self.max_track_history:
                    self.track_history[box.id].pop(0)
    
    def _calculate_velocity(self, track_id: int) -> Tuple[float, float]:
        """Calculate vehicle velocity from tracking history"""
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            return (0, 0)
        
        history = self.track_history[track_id]
        dx = history[-1][0] - history[-2][0]
        dy = history[-1][1] - history[-2][1]
        
        return (dx, dy)
    
    def _check_temporal_consistency(self, current_result: Dict, current_frame: np.ndarray) -> bool:
        """Check if the current detection is consistent with previous frames"""
        if self.prev_detections is None or self.prev_frame is None:
            return True
        
        # Check if the number of vehicles is consistent
        if abs(len(current_result["detections"]) - len(self.prev_detections)) > 2:
            return False
        
        # Check if the overlap is consistent
        current_overlap = current_result["overlap"]
        prev_overlap = self._calculate_overlap(self.prev_detections)
        if abs(current_overlap - prev_overlap) > 0.3:
            return False
        
        # Check motion consistency
        if not self._check_motion(current_frame):
            return False
        
        return True
    
    def _calculate_severity(self, boxes: List, overlap: float) -> Dict:
        """Calculate severity of potential accident"""
        # Base severity on number of vehicles and overlap
        vehicle_count = len(boxes)
        severity_score = min(1.0, (vehicle_count / 4) * 0.5 + overlap * 0.5)
        
        # Adjust severity based on vehicle types
        vehicle_types = [self.vehicle_classes[int(box.cls)] for box in boxes]
        if "truck" in vehicle_types or "bus" in vehicle_types:
            severity_score = min(1.0, severity_score + 0.2)
        
        # Adjust severity based on velocities
        if hasattr(boxes[0], 'id') and boxes[0].id in self.track_history:
            velocities = [self._calculate_velocity(box.id) for box in boxes 
                        if hasattr(box, 'id') and box.id in self.track_history]
            if velocities:
                speed = np.mean([np.sqrt(vx**2 + vy**2) for vx, vy in velocities])
                severity_score = min(1.0, severity_score + speed / 100)
        
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