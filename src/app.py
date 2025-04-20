import streamlit as st
import tempfile
import os
import cv2
import numpy as np
from detection.accident_detector import AccidentDetector
from datetime import datetime

def main():
    st.title("AI-Powered Accident Detection System")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
        frame_skip = st.slider("Frame Skip", 1, 10, 5)
        
        # Sidebar for emergency contacts and claim form
        st.header("Emergency Contacts")
        st.write("Police: 911")
        st.write("Ambulance: 911")
        st.write("Fire Department: 911")
        st.write("Roadside Assistance: 1-800-ROAD-HELP")
        
        st.header("Insurance Claim Form")
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
    
    # Initialize detector
    detector = AccidentDetector()
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
            
        # Process video
        with st.spinner("Processing video..."):
            results = detector.process_video(video_path)
            
        # Clean up temporary file
        os.unlink(video_path)
        
        if results["accident_detected"]:
            st.error("🚨 ACCIDENT DETECTED! 🚨")
            
            # Display severity information
            st.subheader("Accident Severity")
            severity = results["severity"]
            st.write(f"Level: {severity['level']}")
            
            # Display severity factors
            with st.expander("View Severity Details"):
                st.write("Vehicle Count:", severity["factors"]["vehicle_count"])
                st.write("Vehicle Types:", severity["factors"]["vehicle_types"])
                st.write("Max Speeds:", severity["factors"]["max_speeds"])
                st.write("Max Overlap:", f"{severity['factors']['max_overlap']:.2f}")
                st.write("Severity Score:", f"{severity['factors']['severity_score']:.2f}")
            
            # Display insurance information
            st.subheader("Insurance Assessment")
            insurance = results["insurance"]
            st.write(f"Estimated Damage: ${insurance['estimated_damage']:,.2f}")
            st.write(f"Repair Estimate: ${insurance['repair_estimate']:,.2f}")
            
            # Display vehicle details
            with st.expander("View Vehicle Details"):
                for vehicle_type, details in insurance["vehicle_details"].items():
                    st.write(f"\n{vehicle_type.title()}:")
                    st.write(f"Count: {details['count']}")
                    st.write(f"Total Damage: ${details['total_damage']:,.2f}")
                    st.write(f"Max Damage: ${details['max_damage']:,.2f}")
            
            # Display video frames
            st.subheader("Accident Frames")
            for frame_data in results["frames"]:
                frame = frame_data["frame"]
                st.image(frame, caption=f"Frame {frame_data['frame_number']}")
                
        else:
            st.success("No accident detected in the video.")
            
    else:
        st.info("Please upload a video file to begin analysis.")

if __name__ == "__main__":
    main() 