import numpy as np
import cv2
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

class ObjectTracker:
    def __init__(self, model_path: str = 'yolov8n.pt'):
        """Initialize the object tracker with YOLOv8 and ByteTrack"""
        self.yolo = YOLO(model_path)
        self.track_history = defaultdict(lambda: [])
        self.frame_count = 0
        
        # Define vehicle and person classes
        self.vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        self.person_class = 0
        
        # Tracking parameters
        self.min_tracking_frames = 5  # Minimum frames to track before considering speed
        self.speed_threshold = 0.3  # Threshold for sudden speed changes
        self.iou_threshold = 0.4  # Threshold for collision detection
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame for object detection and tracking"""
        try:
            # Run YOLO detection
            results = self.yolo.track(
                frame,
                persist=True,
                classes=list(self.vehicle_classes.keys()) + [self.person_class],
                tracker="bytetrack.yaml"
            )
            
            if not results or len(results) == 0:
                return {"detections": [], "tracks": {}}
            
            # Get detections and tracks
            detections = results[0].boxes
            tracks = {}
            
            # Process each detection
            for box in detections:
                if not hasattr(box, 'id') or box.id is None:
                    continue
                
                track_id = int(box.id)
                cls = int(box.cls[0]) if len(box.cls) > 0 else None
                
                if cls is None:
                    continue
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Update track history
                self.track_history[track_id].append((center_x, center_y, self.frame_count))
                
                # Calculate speed if enough history
                speed = self._calculate_speed(track_id)
                
                # Store track information
                tracks[track_id] = {
                    'class': 'person' if cls == self.person_class else self.vehicle_classes.get(cls, 'unknown'),
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'speed': speed,
                    'confidence': float(box.conf[0]) if len(box.conf) > 0 else 0.0
                }
            
            self.frame_count += 1
            return {"detections": detections, "tracks": tracks}
            
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_speed(self, track_id: int) -> float:
        """Calculate speed of a tracked object"""
        history = self.track_history[track_id]
        if len(history) < self.min_tracking_frames:
            return 0.0
        
        # Calculate average speed over last few frames
        speeds = []
        for i in range(1, len(history)):
            prev_x, prev_y, prev_frame = history[i-1]
            curr_x, curr_y, curr_frame = history[i]
            
            if curr_frame == prev_frame:
                continue
                
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            distance = np.sqrt(dx**2 + dy**2)
            time_diff = curr_frame - prev_frame
            
            if time_diff > 0:
                speed = distance / time_diff
                speeds.append(speed)
        
        return np.mean(speeds) if speeds else 0.0
    
    def detect_collisions(self, tracks: Dict) -> List[Dict]:
        """Detect potential collisions between objects"""
        collisions = []
        track_ids = list(tracks.keys())
        
        for i, id1 in enumerate(track_ids):
            for id2 in track_ids[i+1:]:
                track1 = tracks[id1]
                track2 = tracks[id2]
                
                # Calculate IoU
                iou = self._calculate_iou(track1['bbox'], track2['bbox'])
                
                # Check for collision
                if iou > self.iou_threshold:
                    # Check for sudden speed changes
                    speed_change1 = abs(track1['speed'] - self._get_previous_speed(id1))
                    speed_change2 = abs(track2['speed'] - self._get_previous_speed(id2))
                    
                    if speed_change1 > self.speed_threshold or speed_change2 > self.speed_threshold:
                        collisions.append({
                            'objects': [id1, id2],
                            'iou': iou,
                            'speed_changes': [speed_change1, speed_change2],
                            'classes': [track1['class'], track2['class']]
                        })
        
        return collisions
    
    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _get_previous_speed(self, track_id: int) -> float:
        """Get the previous speed of a tracked object"""
        history = self.track_history[track_id]
        if len(history) < 2:
            return 0.0
        
        prev_x, prev_y, prev_frame = history[-2]
        curr_x, curr_y, curr_frame = history[-1]
        
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        distance = np.sqrt(dx**2 + dy**2)
        time_diff = curr_frame - prev_frame
        
        return distance / time_diff if time_diff > 0 else 0.0 