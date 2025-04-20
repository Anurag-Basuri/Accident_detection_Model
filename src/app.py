import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import sys
import matplotlib.pyplot as plt
import asyncio
import warnings
import logging
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import local modules
from src.detection.accident_detector import AccidentDetector
from src.utils.visualization import create_severity_gauge, create_damage_bar_chart

# Set page config
st.set_page_config(
    page_title="Accident Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = AccidentDetector()
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_processed': 0,
        'accidents_detected': 0,
        'false_positives': 0,
        'processing_time': 0,
        'last_update': datetime.now()
    }

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .frame-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: center;
    }
    .frame-item {
        flex: 0 0 calc(33.33% - 10px);
        max-width: calc(33.33% - 10px);
    }
    .severity-container {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
    .damage-container {
        margin: 20px 0;
    }
    .emergency-info {
        background-color: #ffebee;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .stats-card {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

def update_stats(accident_detected, processing_time):
    """Update statistics"""
    st.session_state.stats['total_processed'] += 1
    if accident_detected:
        st.session_state.stats['accidents_detected'] += 1
    st.session_state.stats['processing_time'] = processing_time
    st.session_state.stats['last_update'] = datetime.now()

def display_stats():
    """Display statistics in the sidebar"""
    with st.sidebar:
        st.header("📊 Statistics")
        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
        st.metric("Total Files Processed", st.session_state.stats['total_processed'])
        st.metric("Accidents Detected", st.session_state.stats['accidents_detected'])
        st.metric("Processing Time (s)", f"{st.session_state.stats['processing_time']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.header("⚙️ Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        frame_skip = st.slider("Frame Skip", 1, 10, 5)
        
        st.header("🚨 Emergency Contacts")
        emergency_contacts = {
            "Police": "911",
            "Ambulance": "911",
            "Fire Department": "911",
            "Roadside Assistance": "1-800-ROAD-HELP"
        }
        for service, number in emergency_contacts.items():
            st.markdown(f"**{service}**: {number}")
        
        st.header("📝 Insurance Claim")
        with st.form("claim_form"):
            name = st.text_input("Full Name")
            policy_number = st.text_input("Policy Number")
            phone = st.text_input("Phone Number")
            email = st.text_input("Email")
            accident_date = st.date_input("Accident Date")
            accident_time = st.time_input("Accident Time")
            description = st.text_area("Accident Description")
            
            if st.form_submit_button("Generate Claim Form"):
                if all([name, policy_number, phone, email]):
                    st.success("Claim form generated successfully!")
                    st.download_button(
                        label="Download Claim Form",
                        data=f"""
                        Insurance Claim Form
                        --------------------
                        Name: {name}
                        Policy Number: {policy_number}
                        Phone: {phone}
                        Email: {email}
                        Accident Date: {accident_date}
                        Accident Time: {accident_time}
                        Description: {description}
                        """,
                        file_name=f"insurance_claim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                else:
                    st.error("Please fill in all required fields")

def main():
    st.title("🚗 Accident Detection System")
    
    # Display statistics in sidebar
    display_stats()
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi"])
    
    if uploaded_file is not None:
        start_time = datetime.now()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        
        try:
            # Process file based on type
            if uploaded_file.type.startswith('image'):
                result = st.session_state.detector.process_image(file_path)
                display_image_results(result, file_path)
            else:
                result = st.session_state.detector.process_video(file_path)
                display_video_results(result, file_path)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            update_stats(result.get("accident_detected", False), processing_time)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
        finally:
            # Clean up
            os.unlink(file_path)

def display_image_results(result, image_path):
    """Display results for image processing"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Display original image
        image = Image.open(image_path)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        if result["accident_detected"]:
            st.error("🚨 Accident Detected!")
            display_severity_info(result)
            display_emergency_info()
        else:
            st.success("✅ No Accident Detected")
            st.info("The image appears to be safe.")

def display_video_results(result, video_path):
    """Display results for video processing"""
    if result["accident_detected"]:
        st.error("🚨 Accident Detected!")
        
        # Display severity and insurance info
        col1, col2 = st.columns([1, 1])
        
        with col1:
            display_severity_info(result)
        
        with col2:
            display_insurance_info(result)
        
        # Display accident frames
        st.subheader("Accident Frames")
        display_accident_frames(result["frames"])
        
        # Display emergency info
        display_emergency_info()
    else:
        st.success("✅ No Accident Detected")
        st.info("The video appears to be safe.")

def display_severity_info(result):
    """Display severity information"""
    st.subheader("Accident Severity")
    severity = result.get("severity", {})
    
    if severity and severity.get("level") != "Unknown":
        # Create and display severity gauge
        fig = create_severity_gauge(severity.get("severity_score", 0))
        st.pyplot(fig)
        
        # Display severity level
        level = severity["level"]
        if level == "Minor":
            st.info(f"Severity: {level} - Minor damage, likely repairable")
        elif level == "Moderate":
            st.warning(f"Severity: {level} - Significant damage, may require extensive repairs")
        else:
            st.error(f"Severity: {level} - Severe damage, likely total loss")

def display_insurance_info(result):
    """Display insurance information"""
    st.subheader("Insurance Assessment")
    insurance = result.get("insurance", {})
    
    if insurance:
        # Display damage bar chart
        fig = create_damage_bar_chart(insurance.get("vehicle_details", {}))
        st.pyplot(fig)
        
        # Display estimated costs
        st.write(f"Estimated Total Damage: ${insurance.get('estimated_damage', 0):,.2f}")
        st.write(f"Repair Estimate: ${insurance.get('repair_estimate', 0):,.2f}")

def display_accident_frames(frames):
    """Display accident frames in a grid"""
    st.markdown('<div class="frame-container">', unsafe_allow_html=True)
    
    for frame_data in frames:
        frame = frame_data["frame"]
        frame_number = frame_data["frame_number"]
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame to be smaller
        height, width = frame_rgb.shape[:2]
        new_height = int(height * 0.4)
        new_width = int(width * 0.4)
        frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
        
        # Display frame
        st.markdown(f'<div class="frame-item">', unsafe_allow_html=True)
        st.image(frame_resized, caption=f"Frame {frame_number}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_emergency_info():
    """Display emergency contact information"""
    st.markdown("""
        <div class="emergency-info">
            <h3>🚨 Emergency Contacts</h3>
            <p>If you're involved in an accident:</p>
            <ul>
                <li>Call Emergency Services: 911</li>
                <li>Contact Insurance Provider: 1-800-INSURANCE</li>
                <li>Police Department: 1-800-POLICE</li>
            </ul>
            <p>Stay calm and ensure everyone's safety first.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 