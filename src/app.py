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
import time
import traceback

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
from src.utils.visualization import create_severity_gauge, create_damage_bar_chart

# Set event loop policy for Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Custom theme
custom_theme = {
    'primary': '#2196F3',
    'danger': '#f44336',
    'success': '#4CAF50',
    'warning': '#ff9800',
    'info': '#03a9f4',
    'background': '#f5f5f5',
    'text': '#212121'
}

# Set page config
st.set_page_config(
    page_title="AI Accident Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'detector' not in st.session_state:
    try:
        st.session_state.detector = AccidentDetector()
    except Exception as e:
        st.error(f"Failed to initialize accident detector: {str(e)}")
        st.stop()
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_processed': 0,
        'accidents_detected': 0,
        'false_positives': 0,
        'processing_time': 0,
        'last_update': datetime.now(),
        'detected_objects': {}
    }
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #2196F3;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .frame-container {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        justify-content: center;
        padding: 20px;
    }
    .frame-item {
        flex: 0 0 calc(33.33% - 20px);
        max-width: calc(33.33% - 20px);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .frame-item:hover {
        transform: translateY(-5px);
    }
    .severity-container, .damage-container {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 20px 0;
        transition: all 0.3s ease;
    }
    .severity-container:hover, .damage-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .emergency-info {
        background-color: #ffebee;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-left: 5px solid #f44336;
    }
    .stats-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stats-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .detection-box {
        position: relative;
        border: 3px solid #2196F3;
        border-radius: 8px;
        margin: 8px;
        padding: 8px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(33, 150, 243, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(33, 150, 243, 0); }
        100% { box-shadow: 0 0 0 0 rgba(33, 150, 243, 0); }
    }
    .detection-label {
        position: absolute;
        top: -25px;
        left: 0;
        background-color: #2196F3;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 14px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .upload-container {
        background-color: white;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 20px 0;
        border: 2px dashed #2196F3;
        text-align: center;
    }
    .metric-container {
        display: flex;
        gap: 20px;
        margin: 20px 0;
    }
    .metric-card {
        flex: 1;
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

def draw_detection_boxes(image, detections):
    """Draw detection boxes with enhanced visualization"""
    img = image.copy()
    height, width = img.shape[:2]
    detected_objects = {}
    
    for box in detections.boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls)
        conf = float(box.conf[0])
        
        # Get object class name
        class_name = st.session_state.detector.vehicle_classes.get(cls, {}).get('name', 'unknown')
        
        # Update detected objects count
        detected_objects[class_name] = detected_objects.get(class_name, 0) + 1
        
        # Generate unique color for each class
        color = plt.cm.rainbow(hash(class_name) % 256 / 256)[:3]
        color = tuple(int(c * 255) for c in color)
        
        # Draw rectangle with animation effect
        thickness = 3
        for i in range(thickness):
            alpha = (thickness - i) / thickness
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        
        # Add label with enhanced design
        label = f"{class_name} {conf:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw label background with gradient
        gradient_colors = [(c * 0.7, c * 0.85, c) for c in color]
        for i in range(label_height + 10):
            y = y1 - i - 5
            alpha = i / (label_height + 10)
            current_color = tuple(int(c[0] * (1-alpha) + c[1] * alpha) for c in zip(*gradient_colors))
            cv2.rectangle(img, (x1, y), (x1 + label_width, y+1), current_color, -1)
        
        # Add label text with shadow
        cv2.putText(img, label, (x1+1, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        cv2.putText(img, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    return img, detected_objects

def display_detection_progress():
    """Display an animated progress bar for detection process"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        progress = i + 1
        progress_bar.progress(progress)
        if progress < 30:
            status_text.text("🔍 Initializing detection models...")
        elif progress < 60:
            status_text.text("🎯 Analyzing objects in the scene...")
        elif progress < 90:
            status_text.text("⚡ Processing detection results...")
        else:
            status_text.text("✨ Finalizing analysis...")
        time.sleep(0.02)
    
    progress_bar.empty()
    status_text.empty()

def display_object_summary(detected_objects):
    """Display a summary of detected objects"""
    if detected_objects:
        st.subheader("🎯 Detection Summary")
        cols = st.columns(len(detected_objects))
        for col, (obj_type, count) in zip(cols, detected_objects.items()):
            with col:
                st.metric(
                    label=obj_type.title(),
                    value=count,
                    delta=f"+{count} detected"
                )

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

def validate_file(file):
    """Validate uploaded file"""
    if file is None:
        return False, "No file uploaded"
    
    # Check file size (max 100MB)
    max_size = 100 * 1024 * 1024  # 100MB
    if file.size > max_size:
        return False, "File size exceeds 100MB limit"
    
    # Check file type
    allowed_types = ['image/jpeg', 'image/png', 'image/jpg', 'video/mp4', 'video/avi']
    if file.type not in allowed_types:
        return False, f"Unsupported file type: {file.type}. Please upload an image (JPEG, PNG) or video (MP4, AVI)."
    
    return True, ""

def process_image(image_path):
    """Process image file with error handling"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to read image file")
        
        # Run detection
        detections = st.session_state.detector.yolo_model(image)
        
        # Check if detections is a list (YOLO v8 format)
        if isinstance(detections, list):
            if not detections:
                return {
                    'success': True,
                    'image': image,
                    'detected_objects': {},
                    'accident_detected': False
                }
            # Get the first detection result
            detections = detections[0]
        
        # Check if detections has boxes attribute
        if not hasattr(detections, 'boxes'):
            # If no boxes, return empty results
            return {
                'success': True,
                'image': image,
                'detected_objects': {},
                'accident_detected': False
            }
        
        # Draw detection boxes
        annotated_image, detected_objects = draw_detection_boxes(image, detections)
        
        return {
            'success': True,
            'image': annotated_image,
            'detected_objects': detected_objects,
            'accident_detected': len(detected_objects) >= 2
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def process_video(video_path):
    """Process video file with error handling"""
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video file")
        
        # Process video
        result = st.session_state.detector.process_video(video_path)
        if not isinstance(result, dict):
            raise ValueError("Invalid video processing results")
        
        return {
            'success': True,
            'result': result
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    finally:
        if 'cap' in locals():
            cap.release()

def main():
    st.title("🚗 AI-Powered Accident Detection System")
    st.markdown("### 🔍 Upload an image or video for accident detection and analysis")
    
    # Display statistics in sidebar
    display_stats()
    
    # File uploader with enhanced UI
    with st.container():
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag and drop files here or click to browse",
            type=["jpg", "jpeg", "png", "mp4", "avi"],
            help="Supported formats: JPG, PNG, MP4, AVI"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Validate file
        is_valid, error_message = validate_file(uploaded_file)
        if not is_valid:
            st.error(error_message)
            return
        
        start_time = datetime.now()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        
        try:
            # Show processing progress
            with st.spinner('🔍 Analyzing...'):
                display_detection_progress()
                
                # Process file based on type
                if uploaded_file.type.startswith('image'):
                    result = process_image(file_path)
                    if not result['success']:
                        st.error(f"Error processing image: {result['error']}")
                        if st.checkbox("Show detailed error information"):
                            st.code(result['traceback'])
                        return
                    
                    # Display results
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.image(cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB),
                                caption="Detection Results",
                                use_column_width=True)
                    
                    with col2:
                        display_object_summary(result['detected_objects'])
                        
                        if result['accident_detected']:
                            st.error("🚨 Potential Accident Detected!")
                            display_severity_info({"severity": {"level": "Moderate", "score": 0.6}})
                        else:
                            st.success("✅ No Accident Detected")
                
                else:
                    result = process_video(file_path)
                    if not result['success']:
                        st.error(f"Error processing video: {result['error']}")
                        if st.checkbox("Show detailed error information"):
                            st.code(result['traceback'])
                        return
                    
                    display_video_results(result['result'], file_path)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            update_stats(result.get("accident_detected", False), processing_time)
            
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            if st.checkbox("Show detailed error information"):
                st.code(traceback.format_exc())
        finally:
            # Clean up
            try:
                os.unlink(file_path)
            except Exception as e:
                st.warning(f"Warning: Could not delete temporary file: {str(e)}")
    else:
        # Display sample images or instructions
        st.info("👆 Upload an image or video to begin analysis")
        
        # Display features
        st.markdown("### ✨ Key Features")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("🎯 **Real-time Detection**")
            st.markdown("Instantly detect vehicles and accidents")
        with col2:
            st.markdown("📊 **Detailed Analysis**")
            st.markdown("Get comprehensive accident reports")
        with col3:
            st.markdown("🚑 **Emergency Response**")
            st.markdown("Quick access to emergency contacts")

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