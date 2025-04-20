import tensorflow as tf
import cv2
import numpy as np
import os

# Dynamically resolve the model path
model_path = os.path.join(os.path.dirname(__file__), "../models/image_model.h5")
model = tf.keras.models.load_model(model_path)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Change according to model input
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict_image(image_path):
    input_img = preprocess_image(image_path)
    prediction = model.predict(input_img)
    print("Prediction:", prediction)
    return prediction[0][0] > 0.5

import os

# Path to the folder containing test images
image_folder = os.path.join(os.path.dirname(__file__), "../data/image")  # adjust if needed

# Loop through each file in the folder
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(image_folder, filename)
        try:
            is_accident = predict_image(image_path)
            status = "Accident" if is_accident else "No Accident"
            print(f"{filename}: {status}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
