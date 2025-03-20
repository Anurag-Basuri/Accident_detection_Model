import cv2
import numpy as np
import tensorflow as tf
from cnn_lstm_efficientnet import CNN_LSTM_EfficientNet
from utils.preprocessing import preprocess_video

# Load trained model
MODEL_PATH = "models/model_checkpoints/model_epoch_best.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
CLASS_NAMES = ["Non-Accident", "Accident"]

# Load video
VIDEO_PATH = "test_video.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

frames = []
frame_count = 0
SEQUENCE_LENGTH = 30  # Match the training sequence length

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    frame = preprocess_video(frame)  # Resize, normalize, etc.
    frames.append(frame)
    frame_count += 1

    if frame_count == SEQUENCE_LENGTH:
        frames_array = np.expand_dims(np.array(frames), axis=0)  # Shape: (1, 30, 224, 224, 3)
        prediction = model.predict(frames_array)
        predicted_class = np.argmax(prediction)
        print(f"🔹 Prediction: {CLASS_NAMES[predicted_class]} (Confidence: {prediction[0][predicted_class]:.4f})")
        
        # Reset for next sequence
        frames = []
        frame_count = 0

cap.release()
cv2.destroyAllWindows()