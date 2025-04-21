import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Any
import cv2
import streamlit as st
import seaborn as sns

def create_severity_gauge(severity_score: float) -> plt.Figure:
    """Create a gauge chart for severity visualization"""
    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': 'polar'})
    
    # Define angles and colors
    angles = np.linspace(0, np.pi, 100)
    colors = ['#4CAF50', '#FF9800', '#F44336']
    thresholds = [0.3, 0.7, 1.0]
    
    # Create gauge background
    for i in range(len(thresholds)):
        if i == 0:
            start = 0
        else:
            start = thresholds[i-1]
        end = thresholds[i]
        angle_range = np.linspace(start * np.pi, end * np.pi, 100)
        ax.fill_between(angle_range, 0, 1, color=colors[i], alpha=0.5)
    
    # Add needle
    severity_angle = severity_score * np.pi
    ax.plot([severity_angle, severity_angle], [0, 1], color='black', linewidth=2)
    
    # Add labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Severity Score', pad=20)
    
    # Add severity level text
    if severity_score < 0.3:
        level = "Minor"
    elif severity_score < 0.7:
        level = "Moderate"
    else:
        level = "Severe"
    
    ax.text(0, 1.1, f"{level}", ha='center', va='center', fontsize=12)
    
    return fig

def create_damage_bar_chart(damage_details: Dict) -> plt.Figure:
    """Create a bar chart for damage assessment"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Prepare data
    vehicles = list(damage_details.keys())
    damages = [details['total_damage'] for details in damage_details.values()]
    
    # Create bar chart
    bars = ax.bar(vehicles, damages, color='#FF9800')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}',
                ha='center', va='bottom')
    
    # Customize plot
    ax.set_title('Estimated Damage by Vehicle')
    ax.set_ylabel('Estimated Damage ($)')
    plt.xticks(rotation=45)
    
    return fig

def create_impact_visualization(overlap: float, vehicle_types: List[str]) -> plt.Figure:
    """Create a visualization of the impact between vehicles"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Create impact circles
    circle1 = plt.Circle((0.3, 0.5), 0.2, color='#4CAF50', alpha=0.5)
    circle2 = plt.Circle((0.7, 0.5), 0.2, color='#FF9800', alpha=0.5)
    
    # Add circles to plot
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    
    # Add vehicle labels
    ax.text(0.3, 0.5, vehicle_types[0] if vehicle_types else 'Vehicle 1',
            ha='center', va='center')
    ax.text(0.7, 0.5, vehicle_types[1] if len(vehicle_types) > 1 else 'Vehicle 2',
            ha='center', va='center')
    
    # Add overlap indicator
    overlap_text = f"Overlap: {overlap:.2%}"
    ax.text(0.5, 0.1, overlap_text, ha='center', va='center')
    
    # Set plot limits and remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Impact Visualization')
    
    return fig

def draw_detection_boxes(image: np.ndarray, detections: List, is_accident: bool) -> Tuple[np.ndarray, Dict[str, int]]:
    """Draw detection boxes on the image and count object types"""
    img = image.copy()
    detected_objects = {}
    
    # Define colors
    box_color = (0, 255, 0) if not is_accident else (0, 0, 255)
    text_color = (255, 255, 255)
    
    for box in detections:
        try:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
            
            # Get class and confidence
            if hasattr(box, 'cls') and len(box.cls) > 0:
                cls = int(box.cls[0])
                conf = float(box.conf[0]) if hasattr(box, 'conf') and len(box.conf) > 0 else 0.0
                
                # Get class name
                class_name = "Unknown"
                if hasattr(box, 'names') and cls in box.names:
                    class_name = box.names[cls]
                
                # Update object count
                detected_objects[class_name] = detected_objects.get(class_name, 0) + 1
                
                # Add label
                label = f"{class_name} {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        except Exception as e:
            print(f"Error drawing box: {str(e)}")
            continue
    
    return img, detected_objects 