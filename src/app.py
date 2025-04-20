import streamlit as st
import tempfile
import os
import cv2
import numpy as np
from detection.accident_detector import AccidentDetector
from utils.visualization import (
    create_severity_gauge,
    create_damage_bar_chart,
    create_speed_heatmap,
    draw_vehicle_boxes,
    create_impact_visualization
)
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="AI Accident Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #2196F3;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stAlert {
        border-radius: 5px;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.title("🚗 AI-Powered Accident Detection System")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
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
    
    # Main content
    uploaded_file = st.file_uploader("📁 Upload an image or video", 
                                    type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name
            
        # Initialize detector
        detector = AccidentDetector()
        
        # Process based on file type
        if uploaded_file.type.startswith('image'):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📸 Original Image")
                st.image(uploaded_file, use_column_width=True)
            
            with st.spinner('🔍 Processing image...'):
                results = detector.process_image(file_path)
                
                # Visualize detections
                img = cv2.imread(file_path)
                yolo_results = detector.yolo_model(img)
                vis_img = draw_vehicle_boxes(img, yolo_results, 
                                          results["accident_detected"])
                
                with col2:
                    st.subheader("🎯 Detection Results")
                    st.image(vis_img, channels="BGR", use_column_width=True)
                
                if results["accident_detected"]:
                    st.error("🚨 Accident Detected!")
                    
                    # Create columns for metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Confidence", f"{results['confidence']:.2%}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Calculate severity and insurance
                    severity = detector._calculate_severity([{"detections": yolo_results}])
                    insurance = detector._assess_insurance([{"detections": yolo_results}])
                    
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Severity Level", severity['level'])
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Estimated Damage", f"${insurance['estimated_damage']:,.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Visualizations
                    st.subheader("📊 Analysis")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(create_severity_gauge(severity['factors']['severity_score']))
                    
                    with col2:
                        st.pyplot(create_impact_visualization(severity['factors']['max_overlap']))
                    
                    # Damage chart
                    st.pyplot(create_damage_bar_chart(insurance['vehicle_details']))
                    
                    # Vehicle details
                    with st.expander("🚗 Vehicle Details"):
                        for vehicle_type, details in insurance["vehicle_details"].items():
                            st.markdown(f"### {vehicle_type.title()}")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Count", details['count'])
                            with col2:
                                st.metric("Total Damage", f"${details['total_damage']:,.2f}")
                            with col3:
                                st.metric("Max Damage", f"${details['max_damage']:,.2f}")
                else:
                    st.success("✅ No Accident Detected")
                
        elif uploaded_file.type.startswith('video'):
            st.video(uploaded_file)
            
            with st.spinner('🔍 Processing video...'):
                results = detector.process_video(file_path)
                
                if results["accident_detected"]:
                    st.error("🚨 ACCIDENT DETECTED! 🚨")
                    
                    # Create columns for metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Severity Level", results["severity"]['level'])
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Vehicle Count", results["severity"]["factors"]["vehicle_count"])
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Estimated Damage", f"${results['insurance']['estimated_damage']:,.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Visualizations
                    st.subheader("📊 Analysis")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(create_severity_gauge(results["severity"]['factors']['severity_score']))
                    
                    with col2:
                        st.pyplot(create_speed_heatmap(results["severity"]["factors"]["max_speeds"]))
                    
                    # Impact visualization
                    st.pyplot(create_impact_visualization(results["severity"]["factors"]["max_overlap"]))
                    
                    # Damage chart
                    st.pyplot(create_damage_bar_chart(results["insurance"]["vehicle_details"]))
                    
                    # Vehicle details
                    with st.expander("🚗 Vehicle Details"):
                        for vehicle_type, details in results["insurance"]["vehicle_details"].items():
                            st.markdown(f"### {vehicle_type.title()}")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Count", details['count'])
                            with col2:
                                st.metric("Total Damage", f"${details['total_damage']:,.2f}")
                            with col3:
                                st.metric("Max Damage", f"${details['max_damage']:,.2f}")
                    
                    # Display video frames
                    st.subheader("🎥 Key Frames")
                    for frame_data in results["frames"]:
                        frame = frame_data["frame"]
                        st.image(frame, caption=f"Frame {frame_data['frame_number']}")
                else:
                    st.success("✅ No Accident Detected")
        
        # Clean up temporary file
        os.unlink(file_path)
            
    else:
        st.info("📁 Please upload an image or video file to begin analysis.")

if __name__ == "__main__":
    main() 