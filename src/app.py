import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import asyncio
import warnings
import logging
import time
from typing import Dict, List, Any
from detection.accident_detector import AccidentDetector
from utils.visualization import (
    create_severity_gauge,
    create_damage_bar_chart,
    create_impact_visualization,
    draw_detection_boxes
)

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Set event loop policy for Windows
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = AccidentDetector()
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_detections': 0,
        'accidents_detected': 0,
        'false_positives': 0
    }

# Custom theme
st.set_page_config(
    page_title="AI Accident Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .emergency-info {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .detection-box {
        border: 2px solid #4CAF50;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .severity-indicator {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .severity-minor { color: #4CAF50; }
    .severity-moderate { color: #FF9800; }
    .severity-severe { color: #F44336; }
    </style>
    """, unsafe_allow_html=True)

def display_detection_progress():
    """Display an animated progress bar during detection"""
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.01)
    progress_bar.empty()

def process_image(image_file) -> Dict:
    """Process an uploaded image"""
    temp_path = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(image_file.getvalue())
            temp_path = tmp_file.name
        
        # Process image
        result = st.session_state.detector.process_image(temp_path)
        
        return result
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return {"accident_detected": False, "error": str(e)}
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logging.error(f"Error cleaning up temporary file: {str(e)}")

def process_video(video_file) -> Dict:
    """Process an uploaded video"""
    temp_path = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.getvalue())
            temp_path = tmp_file.name
        
        # Process video
        result = st.session_state.detector.process_video(temp_path)
        
        return result
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return {"accident_detected": False, "error": str(e)}
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logging.error(f"Error cleaning up temporary file: {str(e)}")

def display_detection_results(result: Dict, file_type: str):
    """Display detection results with enhanced visualization"""
    if "error" in result:
        st.error(result["error"])
        return
    
    # Display severity gauge
    if "severity" in result:
        severity = result["severity"]
        st.markdown(f"<div class='severity-indicator severity-{severity['level'].lower()}'>"
                   f"Severity Level: {severity['level']}</div>", unsafe_allow_html=True)
        
        # Create and display severity gauge
        fig = create_severity_gauge(severity["severity_score"])
        st.pyplot(fig)
    
    # Display detection boxes
    if file_type == "image" and "detections" in result:
        # Convert image to RGB for visualization
        img = cv2.imread(result["image_path"])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw detection boxes
        annotated_img, detected_objects = draw_detection_boxes(
            img_rgb, result["detections"], result["accident_detected"])
        
        # Display annotated image
        st.image(annotated_img, caption="Detection Results", use_column_width=True)
        
        # Display object summary
        st.markdown("### Detected Objects")
        for obj_type, count in detected_objects.items():
            st.write(f"- {obj_type}: {count}")
    
    # Display video results
    elif file_type == "video" and "frames" in result:
        st.markdown("### Accident Frames")
        for frame_data in result["frames"]:
            frame = frame_data["frame"]
            frame_number = frame_data["frame_number"]
            
            # Draw detection boxes
            annotated_frame, _ = draw_detection_boxes(
                frame, frame_data["result"]["detections"], True)
            
            # Display frame
            st.image(annotated_frame, caption=f"Frame {frame_number}", use_column_width=True)
    
    # Display impact visualization
    if "overlap" in result and "detections" in result:
        vehicle_types = [st.session_state.detector.vehicle_classes[int(box.cls)] 
                        for box in result["detections"]]
        fig = create_impact_visualization(result["overlap"], vehicle_types)
        st.pyplot(fig)
    
    # Display damage assessment
    if "severity" in result and "detections" in result:
        damage_details = {
            vehicle_type: {"total_damage": result["severity"]["severity_score"] * 10000}
            for vehicle_type in set(vehicle_types)
        }
        fig = create_damage_bar_chart(damage_details)
        st.pyplot(fig)

def main():
    """Main application function"""
    st.title("🚗 AI-Powered Accident Detection System")
    
    # Sidebar
    with st.sidebar:
        st.header("Upload Media")
        file_type = st.radio("Select file type:", ["Image", "Video"])
        
        if file_type == "Image":
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        else:
            uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        
        st.header("Emergency Contacts")
        st.markdown("""
            - Emergency Services: 911
            - Roadside Assistance: 1-800-ROAD-HELP
            - Insurance Hotline: 1-800-CLAIM-NOW
        """)
    
    # Main content
    if uploaded_file is not None:
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "FileType": uploaded_file.type,
            "FileSize": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.write(file_details)
        
        # Process file
        with st.spinner("Processing..."):
            if file_type == "Image":
                result = process_image(uploaded_file)
            else:
                result = process_video(uploaded_file)
        
        # Display results
        display_detection_results(result, file_type.lower())
        
        # Update statistics
        st.session_state.stats["total_detections"] += 1
        if result["accident_detected"]:
            st.session_state.stats["accidents_detected"] += 1
        
        # Display statistics
        st.markdown("### Detection Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Detections", st.session_state.stats["total_detections"])
        with col2:
            st.metric("Accidents Detected", st.session_state.stats["accidents_detected"])
        with col3:
            st.metric("False Positives", st.session_state.stats["false_positives"])
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>AI Accident Detection System v2.0</p>
            <p>For emergency assistance, please contact the numbers listed in the sidebar.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 