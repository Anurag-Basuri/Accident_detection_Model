import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Any
import cv2
import streamlit as st

def create_severity_gauge(severity_score: float) -> plt.Figure:
    """Create an enhanced gauge chart for severity visualization"""
    try:
        # Create figure and polar axis
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': 'polar'})
        
        # Set the angle range
        angles = np.linspace(0, np.pi, 100)
        
        # Create the gauge with gradient colors
        ax.plot(angles, np.ones_like(angles), color='lightgray', linewidth=20, alpha=0.3)
        
        # Add severity zones with gradient colors
        minor_zone = np.linspace(0, np.pi/3, 100)
        moderate_zone = np.linspace(np.pi/3, 2*np.pi/3, 100)
        severe_zone = np.linspace(2*np.pi/3, np.pi, 100)
        
        # Create gradient colors
        minor_colors = plt.cm.Greens(np.linspace(0.3, 0.7, len(minor_zone)))
        moderate_colors = plt.cm.Oranges(np.linspace(0.3, 0.7, len(moderate_zone)))
        severe_colors = plt.cm.Reds(np.linspace(0.3, 0.7, len(severe_zone)))
        
        # Fill zones with gradients
        for i, (angle, color) in enumerate(zip(minor_zone, minor_colors)):
            ax.fill_between([angle, angle+0.01], 0, 1, color=color, alpha=0.3)
        for i, (angle, color) in enumerate(zip(moderate_zone, moderate_colors)):
            ax.fill_between([angle, angle+0.01], 0, 1, color=color, alpha=0.3)
        for i, (angle, color) in enumerate(zip(severe_zone, severe_colors)):
            ax.fill_between([angle, angle+0.01], 0, 1, color=color, alpha=0.3)
        
        # Add severity indicator with animation effect
        severity_angle = severity_score * np.pi
        ax.plot([severity_angle, severity_angle], [0, 1], color='black', linewidth=3)
        ax.scatter(severity_angle, 1, color='black', s=100, zorder=5)
        
        # Customize the plot
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_yticklabels([])
        ax.set_xticklabels(['Minor', '', 'Moderate', '', 'Severe'], fontsize=12)
        
        # Add title and score
        plt.title(f'Accident Severity Level: {severity_score:.2f}', pad=20, fontsize=14)
        
        return fig
    except Exception as e:
        st.error(f"Error creating severity gauge: {str(e)}")
        return plt.figure()

def create_damage_bar_chart(damage_details: Dict[str, Dict[str, float]]) -> plt.Figure:
    """Create an enhanced bar chart for damage visualization"""
    try:
        if not damage_details:
            return plt.figure(figsize=(8, 4))
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        vehicle_types = list(damage_details.keys())
        total_damages = [details.get('total_damage', 0) for details in damage_details.values()]
        
        # Create gradient colors for bars
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(vehicle_types)))
        
        bars = ax.bar(vehicle_types, total_damages, color=colors)
        
        # Add value labels with animation effect
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}',
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        
        # Customize the plot
        ax.set_title('Damage Assessment by Vehicle', fontsize=14, pad=20)
        ax.set_ylabel('Damage Amount ($)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        return fig
    except Exception as e:
        st.error(f"Error creating damage bar chart: {str(e)}")
        return plt.figure()

def create_impact_visualization(overlap: float, vehicle_types: List[str]) -> plt.Figure:
    """Create a visualization for impact/overlap with vehicle types"""
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Ensure overlap is between 0 and 1
        overlap = max(0.0, min(1.0, overlap))
        
        # Create circles for vehicles with different colors based on type
        colors = {
            'car': '#2196F3',
            'truck': '#F44336',
            'bus': '#FF9800',
            'motorcycle': '#4CAF50'
        }
        
        # Create first vehicle
        circle1 = plt.Circle((0.3, 0.5), 0.2, 
                           color=colors.get(vehicle_types[0], '#2196F3'), 
                           alpha=0.7)
        
        # Create second vehicle with overlap
        circle2 = plt.Circle((0.7 - overlap * 0.4, 0.5), 0.2,
                           color=colors.get(vehicle_types[1], '#F44336'),
                           alpha=0.7)
        
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        
        # Add impact effect
        if overlap > 0.3:
            impact_center = (0.5, 0.5)
            impact_radius = overlap * 0.1
            impact = plt.Circle(impact_center, impact_radius,
                              color='red', alpha=0.5)
            ax.add_patch(impact)
        
        # Customize the plot
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Impact Visualization', fontsize=14, pad=20)
        
        return fig
    except Exception as e:
        st.error(f"Error creating impact visualization: {str(e)}")
        return plt.figure()

def draw_detection_boxes(frame: np.ndarray, detections: List[Any], 
                        accident_detected: bool = False) -> Tuple[np.ndarray, Dict[str, int]]:
    """Draw enhanced detection boxes with animations and effects"""
    try:
        if frame is None or not detections:
            return frame, {}
            
        annotated_frame = frame.copy()
        detected_objects = {}
        
        # Define colors based on accident status
        base_color = (0, 0, 255) if accident_detected else (0, 255, 0)
        
        for box in detections:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls)
                conf = float(box.conf[0])
                
                # Update detected objects count
                class_name = box.names[cls]
                detected_objects[class_name] = detected_objects.get(class_name, 0) + 1
                
                # Create gradient color based on confidence
                alpha = conf
                color = tuple(int(c * alpha + (255 - c) * (1 - alpha)) for c in base_color)
                
                # Draw rectangle with thicker lines and shadow
                thickness = max(2, int(conf * 3))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Add shadow effect
                shadow_offset = 2
                cv2.rectangle(annotated_frame, 
                            (x1 + shadow_offset, y1 + shadow_offset),
                            (x2 + shadow_offset, y2 + shadow_offset),
                            (0, 0, 0), thickness)
                
                # Add label with background
                label = f"{class_name} {conf:.2f}"
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Draw label background with gradient
                for i in range(label_height + 10):
                    y = y1 - i - 5
                    alpha = i / (label_height + 10)
                    current_color = tuple(int(c * (1 - alpha) + 255 * alpha) for c in color)
                    cv2.rectangle(annotated_frame, 
                                (x1, y),
                                (x1 + label_width, y + 1),
                                current_color, -1)
                
                # Add label text with shadow
                cv2.putText(annotated_frame, label,
                          (x1 + 1, y1 - 7),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                          (0, 0, 0), 2)
                cv2.putText(annotated_frame, label,
                          (x1, y1 - 8),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                          (255, 255, 255), 2)
                
            except Exception as e:
                print(f"Error drawing box: {str(e)}")
                continue
        
        return annotated_frame, detected_objects
    except Exception as e:
        st.error(f"Error in detection box drawing: {str(e)}")
        return frame, {} 