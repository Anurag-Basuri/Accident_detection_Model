import streamlit as st
import tempfile
import os
import cv2
import numpy as np
from detection.accident_detector import AccidentDetector

def main():
    st.title("🚗 Accident Detection System")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
        frame_skip = st.slider("Frame Skip", 1, 10, 5)
    
    # Initialize detector
    detector = AccidentDetector()
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image or video", type=['jpg', 'jpeg', 'png', 'mp4'])
    
    if uploaded_file is not None:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
            
        # Process based on file type
        if uploaded_file.type.startswith('image'):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(uploaded_file, use_column_width=True)
            
            with st.spinner('Processing image...'):
                results = detector.process_image(file_path)
                
                # Visualize detections
                img = cv2.imread(file_path)
                yolo_results = detector.yolo_model(img)
                vis_img = detector.visualize_detections(img, yolo_results, 
                                                      results["accident_detected"])
                
                with col2:
                    st.subheader("Detection Results")
                    st.image(vis_img, channels="BGR", use_column_width=True)
                
                if results["accident_detected"]:
                    st.error("🚨 Accident Detected!")
                    st.write(f"Confidence: {results['confidence']:.2f}")
                else:
                    st.success("✅ No Accident Detected")
                
        elif uploaded_file.type.startswith('video'):
            st.video(uploaded_file)
            
            with st.spinner('Processing video...'):
                results = detector.process_video(file_path)
                
                if results["accident_detected"]:
                    st.error("🚨 Accident Detected!")
                    
                    # Show severity and insurance details
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Severity Analysis")
                        st.write(f"Level: {results['severity']}")
                        st.write(f"Vehicles Involved: {results['insurance_details']['vehicle_count']}")
                    
                    with col2:
                        st.subheader("Insurance Assessment")
                        st.write(f"Estimated Damage: ${results['insurance_details']['estimated_damage']:,.2f}")
                        st.write(f"Repair Estimate: ${results['insurance_details']['repair_estimate']:,.2f}")
                    
                    # Show visualization frames
                    st.subheader("Detection Visualization")
                    for frame in results["visualization_frames"]:
                        st.image(frame, channels="BGR", use_column_width=True)
                else:
                    st.success("✅ No Accident Detected")
        
        # Clean up temporary file
        os.unlink(file_path)

if __name__ == "__main__":
    main() 