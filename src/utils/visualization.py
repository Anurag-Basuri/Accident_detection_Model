import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st
from typing import Dict, List, Tuple, Any

def create_severity_gauge(severity_score: float) -> plt.Figure:
    """Create a gauge chart for severity visualization"""
    # Create figure and polar axis
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    # Set the angle range
    angles = np.linspace(0, np.pi, 100)
    
    # Create the gauge
    ax.plot(angles, np.ones_like(angles), color='lightgray', linewidth=20)
    
    # Add severity zones
    ax.fill_between(angles, 0, 1, where=angles < np.pi/3, color='green', alpha=0.3)
    ax.fill_between(angles, 0, 1, where=(angles >= np.pi/3) & (angles < 2*np.pi/3), color='orange', alpha=0.3)
    ax.fill_between(angles, 0, 1, where=angles >= 2*np.pi/3, color='red', alpha=0.3)
    
    # Add severity indicator
    severity_angle = severity_score * np.pi
    ax.plot([severity_angle, severity_angle], [0, 1], color='black', linewidth=3)
    
    # Customize the plot
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_yticklabels([])
    ax.set_xticklabels(['Minor', '', 'Moderate', '', 'Severe'])
    
    # Add title
    plt.title('Accident Severity Level', pad=20)
    
    return fig

def create_damage_bar_chart(damage_details: Dict[str, Dict[str, float]]) -> plt.Figure:
    """Create a bar chart for damage visualization"""
    if not damage_details:
        return plt.figure(figsize=(8, 4))
        
    fig, ax = plt.subplots(figsize=(8, 4))
    
    vehicle_types = list(damage_details.keys())
    total_damages = [details.get('total_damage', 0) for details in damage_details.values()]
    
    bars = ax.bar(vehicle_types, total_damages, color='#2196F3')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}',
                ha='center', va='bottom')
    
    ax.set_title('Damage by Vehicle Type')
    ax.set_ylabel('Damage Amount ($)')
    plt.xticks(rotation=45)
    
    return fig

def create_speed_heatmap(speeds: Dict[str, float]) -> plt.Figure:
    """Create a heatmap for speed visualization"""
    if not speeds:
        return plt.figure(figsize=(6, 3))
        
    fig, ax = plt.subplots(figsize=(6, 3))
    
    vehicle_types = list(speeds.keys())
    speed_values = list(speeds.values())
    
    # Create heatmap
    im = ax.imshow([speed_values], cmap='YlOrRd')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im)
    cbar.ax.set_ylabel('Speed (km/h)', rotation=-90, va="bottom")
    
    # Add labels
    ax.set_xticks(range(len(vehicle_types)))
    ax.set_xticklabels(vehicle_types)
    ax.set_yticks([])
    ax.set_title('Vehicle Speeds')
    
    return fig

def draw_vehicle_boxes(frame: np.ndarray, detections: List[Any], 
                      accident_detected: bool = False) -> np.ndarray:
    """Draw vehicle bounding boxes with enhanced visualization"""
    if frame is None or not detections:
        return frame
        
    annotated_frame = frame.copy()
    
    # Define colors
    color = (0, 0, 255) if accident_detected else (0, 255, 0)
    
    for box in detections:
        try:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Draw rectangle with thicker lines
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            
            # Add label with background
            label = f"{box.cls} {box.conf[0]:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw label background
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - label_height - 10),
                         (x1 + label_width, y1),
                         color, -1)
            
            # Add label text
            cv2.putText(annotated_frame, label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 2)
        except (AttributeError, IndexError, TypeError) as e:
            print(f"Error drawing box: {e}")
            continue
    
    return annotated_frame

def create_impact_visualization(overlap: float) -> plt.Figure:
    """Create a visualization for impact/overlap"""
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Ensure overlap is between 0 and 1
    overlap = max(0.0, min(1.0, overlap))
    
    # Create circles for vehicles
    circle1 = plt.Circle((0.3, 0.5), 0.2, color='#2196F3', alpha=0.5)
    circle2 = plt.Circle((0.7, 0.5), 0.2, color='#F44336', alpha=0.5)
    
    # Adjust overlap
    circle2.center = (0.7 - overlap * 0.4, 0.5)
    
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Impact Visualization')
    
    return fig 