import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import asyncio
import warnings
import logging
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
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
        'false_positives': 0,
        'vehicle_counts': [],
        'severity_scores': [],
        'detection_times': [],
        'vehicle_types': {},
        'accident_locations': []
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
    .stats-card {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
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
        
        # Add image path to result
        result["image_path"] = temp_path
        
        return result
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return {"accident_detected": False, "error": str(e)}
    finally:
        # Note: We don't delete the temp file here as it's needed for display
        # It will be cleaned up after display_detection_results is done
        pass

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

def update_statistics(result: Dict):
    """Update detection statistics"""
    current_time = datetime.now()
    
    # Update basic statistics
    st.session_state.stats['total_detections'] += 1
    if result.get('accident_detected', False):
        st.session_state.stats['accidents_detected'] += 1
    
    # Update vehicle counts
    if 'vehicle_count' in result:
        st.session_state.stats['vehicle_counts'].append(result['vehicle_count'])
    
    # Update severity scores
    if 'severity' in result:
        st.session_state.stats['severity_scores'].append(result['severity']['severity_score'])
    
    # Update detection times
    st.session_state.stats['detection_times'].append(current_time)
    
    # Update vehicle types
    if 'detections' in result:
        for box in result['detections']:
            if hasattr(box, 'cls') and len(box.cls) > 0:
                cls = int(box.cls[0])
                if cls in st.session_state.detector.vehicle_classes:
                    vehicle_type = st.session_state.detector.vehicle_classes[cls]
                    st.session_state.stats['vehicle_types'][vehicle_type] = \
                        st.session_state.stats['vehicle_types'].get(vehicle_type, 0) + 1

def display_statistics():
    """Display comprehensive statistics"""
    st.markdown("### 📊 Detection Statistics")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class='stats-card'>
                <h3>Total Detections</h3>
                <h2>{}</h2>
            </div>
        """.format(st.session_state.stats['total_detections']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='stats-card'>
                <h3>Accidents Detected</h3>
                <h2>{}</h2>
            </div>
        """.format(st.session_state.stats['accidents_detected']), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='stats-card'>
                <h3>False Positives</h3>
                <h2>{}</h2>
            </div>
        """.format(st.session_state.stats['false_positives']), unsafe_allow_html=True)
    
    with col4:
        avg_vehicles = np.mean(st.session_state.stats['vehicle_counts']) if st.session_state.stats['vehicle_counts'] else 0
        st.markdown("""
            <div class='stats-card'>
                <h3>Avg. Vehicles/Detection</h3>
                <h2>{:.1f}</h2>
            </div>
        """.format(avg_vehicles), unsafe_allow_html=True)
    
    # Create tabs for detailed statistics
    tab1, tab2, tab3 = st.tabs(["📈 Trends", "🚗 Vehicle Analysis", "⚠️ Severity Analysis"])
    
    with tab1:
        # Display detection trends
        if len(st.session_state.stats['detection_times']) > 1:
            df = pd.DataFrame({
                'Time': st.session_state.stats['detection_times'],
                'Vehicle Count': st.session_state.stats['vehicle_counts']
            })
            fig = px.line(df, x='Time', y='Vehicle Count', title='Vehicle Detection Trends')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Display vehicle type distribution
        if st.session_state.stats['vehicle_types']:
            df = pd.DataFrame({
                'Vehicle Type': list(st.session_state.stats['vehicle_types'].keys()),
                'Count': list(st.session_state.stats['vehicle_types'].values())
            })
            fig = px.pie(df, values='Count', names='Vehicle Type', title='Vehicle Type Distribution')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Display severity distribution
        if st.session_state.stats['severity_scores']:
            df = pd.DataFrame({
                'Severity Score': st.session_state.stats['severity_scores']
            })
            fig = px.histogram(df, x='Severity Score', title='Severity Score Distribution')
            st.plotly_chart(fig, use_container_width=True)

def display_detection_results(result: Dict, file_type: str):
    """Display detection results with enhanced visualization"""
    if "error" in result:
        st.error(result["error"])
        return
    
    # Create columns for main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display detection boxes
        if file_type == "image" and "detections" in result:
            try:
                if "image_path" in result:
                    img = cv2.imread(result["image_path"])
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Draw detection boxes
                        annotated_img, detected_objects = draw_detection_boxes(
                            img_rgb, result["detections"], result["accident_detected"])
                        
                        # Display annotated image
                        st.image(annotated_img, caption="Detection Results", use_column_width=True)
                        
                        # Clean up temporary file
                        if os.path.exists(result["image_path"]):
                            os.remove(result["image_path"])
            except Exception as e:
                st.error(f"Error displaying image: {str(e)}")
    
    with col2:
        # Display severity information
        if "severity" in result:
            severity = result["severity"]
            st.markdown(f"<div class='severity-indicator severity-{severity['level'].lower()}'>"
                       f"Severity Level: {severity['level']}</div>", unsafe_allow_html=True)
            
            # Create and display severity gauge
            fig = create_severity_gauge(severity["severity_score"])
            st.pyplot(fig)
            
            # Display vehicle count
            st.markdown(f"**Vehicles Detected:** {result.get('vehicle_count', 0)}")
            
            # Display overlap information
            if "overlap" in result:
                st.markdown(f"**Overlap Score:** {result['overlap']:.2f}")
    
    # Display additional information
    st.markdown("### 📋 Detection Details")
    
    # Create columns for details
    col3, col4 = st.columns(2)
    
    with col3:
        # Display object summary
        if "detections" in result:
            st.markdown("#### Detected Objects")
            for obj_type, count in detected_objects.items():
                st.markdown(f"- **{obj_type}:** {count}")
    
    with col4:
        # Display impact visualization
        if "overlap" in result and "detections" in result:
            try:
                vehicle_types = [st.session_state.detector.vehicle_classes[int(box.cls)] 
                               for box in result["detections"]]
                fig = create_impact_visualization(result["overlap"], vehicle_types)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error creating impact visualization: {str(e)}")
    
    # Display damage assessment
    if "severity" in result and "detections" in result:
        try:
            vehicle_types = [st.session_state.detector.vehicle_classes[int(box.cls)] 
                           for box in result["detections"]]
            damage_details = {
                vehicle_type: {"total_damage": result["severity"]["severity_score"] * 10000}
                for vehicle_type in set(vehicle_types)
            }
            fig = create_damage_bar_chart(damage_details)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating damage assessment: {str(e)}")

def main():
    """Main application function"""
    st.title("🚗 AI-Powered Accident Detection System")
    
    # Sidebar
    with st.sidebar:
        st.header("📤 Upload Media")
        file_type = st.radio("Select file type:", ["Image", "Video"])
        
        if file_type == "Image":
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        else:
            uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        
        st.header("🆘 Emergency Contacts")
        st.markdown("""
            - Emergency Services: 911
            - Roadside Assistance: 1-800-ROAD-HELP
            - Insurance Hotline: 1-800-CLAIM-NOW
        """)
        
        st.header("⚙️ Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.05)
        if 'detector' in st.session_state:
            st.session_state.detector.min_confidence = confidence_threshold
    
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
        update_statistics(result)
        
        # Display statistics
        display_statistics()
    
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