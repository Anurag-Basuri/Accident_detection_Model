import tensorflow as tf
import numpy as np
from cnn_lstm_efficientnet import CNN_LSTM_EfficientNet
from utils.data_loader import load_video_data

# Load evaluation dataset
_, _, val_data, val_labels = load_video_data(split_ratio=0.8)

# Load the best trained model
MODEL_PATH = "models/model_checkpoints/model_epoch_best.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Evaluate
print("📊 Evaluating model...")
loss, accuracy = model.evaluate(val_data, val_labels)
print(f"✅ Evaluation Complete: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
