import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import sys
import asyncio
import warnings
import logging
from datetime import datetime
import time

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTORCH_WARN_ONCE'] = '0'

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import local modules
from src.detection.accident_detector import AccidentDetector

# Set event loop policy for Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = AccidentDetector()

def process_image(image_path):
    """Process image file with error handling"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to read image file")
        
        # Run detection
        result = st.session_state.detector.process_image(image_path)
        if "error" in result:
            raise ValueError(result["error"])
        
        # Draw detection boxes
        annotated_image = image.copy()
        detected_objects = {}
        
        # Draw vehicle boxes
        for box in result["detections"]:
            cls = int(box.cls)
            conf = float(box.conf[0])
            vehicle_type = st.session_state.detector.vehicle_classes[cls]
            
            # Update detected objects count
            detected_objects[vehicle_type] = detected_objects.get(vehicle_type, 0) + 1
            
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Draw box
            color = (0, 0, 255) if result["accident_detected"] else (0, 255, 0)
            thickness = max(1, int(conf * 3))
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
            
            # Add label
            label = f"{vehicle_type} {conf:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw label background
            cv2.rectangle(annotated_image,
                         (x1, y1 - label_height - 10),
                         (x1 + label_width, y1),
                         color, -1)
            
            # Add label text
            cv2.putText(annotated_image, label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 2)
        
        # Add accident status
        status = "ACCIDENT DETECTED!" if result["accident_detected"] else "No Accident Detected"
        color = (0, 0, 255) if result["accident_detected"] else (0, 255, 0)
        cv2.putText(annotated_image, status,
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1,
                   color, 2)
        
        return {
            'success': True,
            'image': annotated_image,
            'detected_objects': detected_objects,
            'accident_detected': result["accident_detected"],
            'vehicle_count': result["vehicle_count"],
            'overlap': result["overlap"]
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def main():
    st.title("🚗 AI-Powered Accident Detection System")
    st.markdown("### 🔍 Upload an image for accident detection")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, PNG"
    )
    
    if uploaded_file is not None:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        
        try:
            # Process image
            with st.spinner('🔍 Analyzing...'):
                result = process_image(file_path)
                
                if not result['success']:
                    st.error(f"Error processing image: {result['error']}")
                    return
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.image(cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB),
                            caption="Detection Results",
                            use_column_width=True)
                
                with col2:
                    # Display detection summary
                    st.subheader("🎯 Detection Summary")
                    st.write(f"Total Vehicles: {result['vehicle_count']}")
                    st.write(f"Overlap: {result['overlap']:.2%}")
                    
                    if result['accident_detected']:
                        st.error("🚨 Potential Accident Detected!")
                    else:
                        st.success("✅ No Accident Detected")
                    
                    # Display detected objects
                    if result['detected_objects']:
                        st.subheader("Detected Vehicles")
                        for vehicle_type, count in result['detected_objects'].items():
                            st.write(f"- {vehicle_type.title()}: {count}")
        
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
        finally:
            # Clean up
            try:
                os.unlink(file_path)
            except Exception as e:
                st.warning(f"Warning: Could not delete temporary file: {str(e)}")
    else:
        st.info("👆 Upload an image to begin analysis")

if __name__ == "__main__":
    main() 