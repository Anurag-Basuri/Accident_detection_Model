import streamlit as st
import tempfile
import os
from detection.accident_detector import AccidentDetector

def main():
    st.title("Accident Detection System")
    
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
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            with st.spinner('Processing image...'):
                results = detector.process_image(file_path)
                
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
                st.write(f"Severity: {results['severity']}")
                st.write("Insurance Details:")
                st.json(results["insurance_details"])
            else:
                st.success("✅ No Accident Detected")
        
        # Clean up temporary file
        os.unlink(file_path)

if __name__ == "__main__":
    main() 