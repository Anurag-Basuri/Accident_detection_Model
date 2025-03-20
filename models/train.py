import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from cnn_lstm_efficientnet import CNN_LSTM_EfficientNet
from utils.data_loader import load_video_data  # Import a utility function for loading video data

# Paths
MODEL_CHECKPOINT_DIR = "models/model_checkpoints"
os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
INPUT_SHAPE = (30, 224, 224, 3)  # 30 frames per video

# Load dataset
train_data, train_labels, val_data, val_labels = load_video_data(split_ratio=0.8)

# Initialize model
model = CNN_LSTM_EfficientNet(input_shape=INPUT_SHAPE, num_classes=2)
model.compile_model(learning_rate=LEARNING_RATE)

# Callbacks for training
callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(MODEL_CHECKPOINT_DIR, "model_epoch_{epoch:02d}.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
]

# Training
print("🚀 Training Started...")
model.model.fit(
    train_data, train_labels,
    validation_data=(val_data, val_labels),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks
)
print("✅ Training Completed & Model Saved!")
