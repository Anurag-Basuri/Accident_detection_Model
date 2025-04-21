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

class AccidentDetector:
    def __init__(self):
        """Initialize the accident detection system"""
        try:
            # Load YOLOv8 model with GPU acceleration if available
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_path = 'yolov8x.pt'  # Using the largest model for better accuracy
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            # Initialize model with higher confidence threshold
            self.yolo_model = YOLO(model_path).to(self.device)
            
            # Vehicle classes with more specific types
            self.vehicle_classes = {
                2: "car",
                3: "motorcycle",
                5: "bus",
                7: "truck"
            }
            
            # Enhanced detection thresholds
            self.min_confidence = 0.6  # Increased from 0.5
            self.min_vehicles = 2
            self.min_overlap = 0.3
            self.min_confidence_for_accident = 0.75  # Increased from 0.7
            self.min_vehicle_size = 100  # Minimum pixel size for a vehicle
            
            # Motion analysis parameters
            self.motion_threshold = 10
            self.prev_frame = None
            self.prev_detections = None
            
            # Initialize tracking with cleanup
            self.track_history = {}
            self.max_track_history = 30
            self.track_cleanup_interval = 100
            self.frame_counter = 0
            
            # Additional validation parameters
            self.min_velocity_threshold = 5.0  # Minimum velocity for accident detection
            self.max_distance_threshold = 200  # Maximum distance between vehicles for accident
            self.min_frame_consistency = 3  # Minimum number of consistent frames for accident
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize AccidentDetector: {str(e)}")
    
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
            
            # Run YOLO detection with simplified tracking
            results = self.yolo_model.track(
                img_rgb,
                conf=self.min_confidence,
                persist=True,
                classes=list(self.vehicle_classes.keys()),
                tracker="bytetrack.yaml"  # Use ByteTrack instead of default tracker
            )
            
            # Process detection results
            if not results or not isinstance(results, list):
                return {"accident_detected": False, "detections": [], "error": "No detections found"}
            
            # Get the first result if available
            if len(results) > 0:
                results = results[0]
            else:
                return {"accident_detected": False, "detections": [], "error": "No detections found"}
            
            # Initialize empty detections list
            vehicle_boxes = []
            
            # Safely get boxes if they exist
            if hasattr(results, 'boxes') and results.boxes is not None:
                for box in results.boxes:
                    try:
                        # Check if box has required attributes
                        if not hasattr(box, 'cls') or not hasattr(box, 'xyxy'):
                            continue
                            
                        cls = int(box.cls[0]) if len(box.cls) > 0 else None
                        if cls is None or cls not in self.vehicle_classes:
                            continue
                            
                        # Safely get coordinates
                        if len(box.xyxy) > 0 and len(box.xyxy[0]) >= 4:
                            x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
                            width = x2 - x1
                            height = y2 - y1
                            
                            # Check vehicle size
                            if width >= self.min_vehicle_size or height >= self.min_vehicle_size:
                                vehicle_boxes.append(box)
                    except (IndexError, ValueError, AttributeError) as e:
                        logging.warning(f"Error processing box: {str(e)}")
                        continue
            
            # Check for potential accident
            is_accident = False
            overlap = 0.0
            
            if len(vehicle_boxes) >= self.min_vehicles:
                try:
                    # Calculate overlap between vehicles
                    overlap = self._calculate_overlap(vehicle_boxes)
                    
                    # Check for high confidence detections
                    high_confidence_vehicles = sum(1 for box in vehicle_boxes 
                                                if hasattr(box, 'conf') and 
                                                len(box.conf) > 0 and 
                                                float(box.conf[0]) >= self.min_confidence_for_accident)
                    
                    # Additional checks for accident detection
                    if (overlap >= self.min_overlap and 
                        high_confidence_vehicles >= 2 and 
                        self._check_vehicle_positions(vehicle_boxes)):
                        is_accident = True
                except Exception as e:
                    logging.error(f"Error in accident detection: {str(e)}")
                    is_accident = False
            
            # Calculate severity
            try:
                severity = self._calculate_severity(vehicle_boxes, overlap)
            except Exception as e:
                logging.error(f"Error calculating severity: {str(e)}")
                severity = {"level": "Unknown", "severity_score": 0.0}
            
            # Update tracking history
            try:
                self._update_tracking(vehicle_boxes)
            except Exception as e:
                logging.error(f"Error updating tracking: {str(e)}")
            
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
            logging.error(f"Error in process_image: {str(e)}")
            return {"accident_detected": False, "error": str(e)}
    
    def process_video(self, video_path: str) -> Dict:
        """
        Process a video file for accident detection
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing detection results
        """
        cap = None
        temp_dir = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"accident_detected": False, "error": "Failed to open video"}
            
            # Create temporary directory for frame storage
            temp_dir = tempfile.mkdtemp()
            frames = []
            frame_number = 0
            accident_frames = []
            frame_skip = 5  # Process every 5th frame
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every Nth frame
                if frame_number % frame_skip == 0:
                    # Save frame to temporary file
                    temp_path = os.path.join(temp_dir, f"frame_{frame_number}.jpg")
                    cv2.imwrite(temp_path, frame)
                    
                    try:
                        # Process frame
                        result = self.process_image(temp_path)
                        
                        # Check for temporal consistency
                        if result["accident_detected"]:
                            if self._check_temporal_consistency(result, frame):
                                # Store only necessary information to save memory
                                accident_frames.append({
                                    "frame_number": frame_number,
                                    "timestamp": frame_number / fps,
                                    "result": {
                                        "accident_detected": result["accident_detected"],
                                        "severity": result.get("severity"),
                                        "detections": result.get("detections", [])
                                    }
                                })
                    finally:
                        # Clean up temporary frame file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                
                frame_number += 1
            
            # Determine if accident was detected
            is_accident = len(accident_frames) > 0
            
            # Calculate overall severity
            severity = self._calculate_overall_severity(accident_frames)
            
            return {
                "accident_detected": is_accident,
                "frames": accident_frames,
                "severity": severity,
                "total_frames": total_frames,
                "processed_frames": frame_number // frame_skip
            }
            
        except Exception as e:
            return {"accident_detected": False, "error": str(e)}
        finally:
            # Clean up resources
            if cap is not None:
                cap.release()
            if temp_dir is not None and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logging.error(f"Error cleaning up temporary directory: {str(e)}")
    
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
    
    def _validate_detection(self, box) -> bool:
        """Validate a single detection"""
        try:
            # Check if box has required attributes
            if not hasattr(box, 'cls') or not hasattr(box, 'xyxy') or not hasattr(box, 'conf'):
                return False
            
            # Check confidence
            if len(box.conf) == 0 or float(box.conf[0]) < self.min_confidence:
                return False
            
            # Check class
            if len(box.cls) == 0 or int(box.cls[0]) not in self.vehicle_classes:
                return False
            
            # Check size
            if len(box.xyxy) == 0 or len(box.xyxy[0]) < 4:
                return False
            
            x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
            width = x2 - x1
            height = y2 - y1
            
            return width >= self.min_vehicle_size or height >= self.min_vehicle_size
            
        except Exception:
            return False

    def _check_vehicle_positions(self, boxes: List) -> bool:
        """Enhanced check for vehicle positions indicating a potential accident"""
        if len(boxes) < 2:
            return False
        
        try:
            # Calculate center points and velocities of vehicles
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
            
            # Enhanced accident detection criteria
            return (min_distance < self.max_distance_threshold and 
                   is_converging and 
                   self._check_velocity_threshold(velocities))
            
        except Exception as e:
            logging.error(f"Error in vehicle position check: {str(e)}")
            return False

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

    def _check_motion(self, current_frame: np.ndarray) -> bool:
        """Check for significant motion in the scene using frame differencing"""
        if self.prev_frame is None:
            self.prev_frame = current_frame
            return False
        
        try:
            # Ensure frames have the same dimensions
            if self.prev_frame.shape != current_frame.shape:
                self.prev_frame = cv2.resize(self.prev_frame, (current_frame.shape[1], current_frame.shape[0]))
            
            # Convert frames to grayscale
            prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_RGB2GRAY)
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur to reduce noise
            prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
            current_gray = cv2.GaussianBlur(current_gray, (21, 21), 0)
            
            # Calculate absolute difference between frames
            frame_diff = cv2.absdiff(prev_gray, current_gray)
            
            # Apply threshold to get significant changes
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            
            # Dilate the thresholded image to fill in holes
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Calculate the percentage of changed pixels
            changed_pixels = np.sum(thresh > 0)
            total_pixels = thresh.size
            change_percentage = (changed_pixels / total_pixels) * 100
            
            # Update previous frame
            self.prev_frame = current_frame.copy()
            
            # Return True if significant motion is detected
            return change_percentage > 1.0  # 1% of pixels changed
            
        except Exception as e:
            logging.error(f"Error in motion detection: {str(e)}")
            # Reset previous frame on error
            self.prev_frame = current_frame.copy()
            return False
    
    def _update_tracking(self, boxes: List) -> None:
        """Update tracking history with cleanup"""
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
        # Remove tracks that haven't been updated in the last max_track_history frames
        self.track_history = {
            track_id: history for track_id, history in self.track_history.items()
            if current_time - history[-1][2] < self.max_track_history
        }
    
    def _calculate_velocity(self, track_id: int) -> Tuple[float, float]:
        """Calculate vehicle velocity from tracking history"""
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            return (0, 0)
        
        history = self.track_history[track_id]
        dx = history[-1][0] - history[-2][0]
        dy = history[-1][1] - history[-2][1]
        
        return (dx, dy)
    
    def _check_temporal_consistency(self, current_result: Dict, current_frame: np.ndarray) -> bool:
        """Enhanced check for temporal consistency in accident detection"""
        if self.prev_detections is None or self.prev_frame is None:
            self.prev_detections = current_result["detections"]
            self.prev_frame = current_frame
            return True
        
        try:
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
            
            # Update previous frame and detections
            self.prev_detections = current_result["detections"]
            self.prev_frame = current_frame
            
            return True
            
        except Exception as e:
            logging.error(f"Error in temporal consistency check: {str(e)}")
            return False
    
    def _calculate_severity(self, boxes: List, overlap: float) -> Dict:
        """Calculate severity of potential accident"""
        try:
            if not boxes:
                return {
                    "level": "None",
                    "severity_score": 0.0,
                    "vehicle_count": 0,
                    "overlap": 0.0
                }
            
            # Base severity on number of vehicles and overlap
            vehicle_count = len(boxes)
            severity_score = min(1.0, (vehicle_count / 4) * 0.5 + overlap * 0.5)
            
            # Adjust severity based on vehicle types
            vehicle_types = []
            for box in boxes:
                try:
                    if hasattr(box, 'cls') and len(box.cls) > 0:
                        cls = int(box.cls[0])
                        if cls in self.vehicle_classes:
                            vehicle_types.append(self.vehicle_classes[cls])
                except (IndexError, ValueError, AttributeError):
                    continue
            
            # Adjust severity based on vehicle types
            if "truck" in vehicle_types or "bus" in vehicle_types:
                severity_score = min(1.0, severity_score + 0.2)
            
            # Adjust severity based on velocities if tracking data is available
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
                level = "Minor"
            elif severity_score < 0.7:
                level = "Moderate"
            else:
                level = "Severe"
            
            return {
                "level": level,
                "severity_score": severity_score,
                "vehicle_count": vehicle_count,
                "overlap": overlap,
                "vehicle_types": vehicle_types
            }
            
        except Exception as e:
            logging.error(f"Error in _calculate_severity: {str(e)}")
            return {
                "level": "Unknown",
                "severity_score": 0.0,
                "vehicle_count": 0,
                "overlap": 0.0,
                "vehicle_types": []
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

    def _load_model(self):
        """Load the YOLO model"""
        try:
            model = YOLO('yolov8n.pt')
            return model
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame for accident detection"""
        try:
            # Convert frame to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Run detection
            results = self.model(frame_rgb, conf=self.min_confidence)
            
            # Process detections
            detections = results[0].boxes
            vehicle_count = 0
            vehicle_types = {}
            
            # Count vehicles and their types
            for box in detections:
                if hasattr(box, 'cls') and len(box.cls) > 0:
                    cls = int(box.cls[0])
                    if cls in self.vehicle_classes:
                        vehicle_count += 1
                        vehicle_type = self.vehicle_classes[cls]
                        vehicle_types[vehicle_type] = vehicle_types.get(vehicle_type, 0) + 1
            
            # Calculate severity based on vehicle count and types
            severity_score = self._calculate_severity(vehicle_count, vehicle_types)
            
            # Determine if accident is detected
            accident_detected = severity_score > 0.5
            
            return {
                'accident_detected': accident_detected,
                'severity': {
                    'severity_score': severity_score,
                    'level': 'Minor' if severity_score < 0.3 else 'Moderate' if severity_score < 0.7 else 'Severe'
                },
                'vehicle_count': vehicle_count,
                'vehicle_types': vehicle_types,
                'detections': detections
            }
            
        except Exception as e:
            return {
                'accident_detected': False,
                'error': f"Error processing frame: {str(e)}"
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