import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from data_loader import AccidentDataLoader
from models.cnn_lstm_efficientnet import build_model  # Your custom model builder

# ========================
# CONFIGURATION
# ========================
BATCH_SIZE = 8
SEQUENCE_LENGTH = 30
IMAGE_SIZE = (224, 224)
EPOCHS = 50
MODEL_SAVE_PATH = "model_checkpoints/best_model.h5"
TRAIN_DIR = "processed-datasets/train"
VAL_DIR = "processed-datasets/val"

# ========================
# DATA GENERATORS
# ========================
train_loader = AccidentDataLoader(
    directory=TRAIN_DIR,
    batch_size=BATCH_SIZE,
    shuffle=True,
    augment=True
)

val_loader = AccidentDataLoader(
    directory=VAL_DIR,
    batch_size=BATCH_SIZE,
    shuffle=False,
    augment=False
)

# ========================
# BUILD MODEL
# ========================
model = build_model(
    sequence_length=SEQUENCE_LENGTH,
    img_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    fine_tune_at=100  # Modify if needed
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ========================
# CALLBACKS
# ========================
callbacks = [
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', mode='min', verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

# ========================
# TRAINING
# ========================
model.fit(
    train_loader,
    validation_data=val_loader,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ========================
# EVALUATION
# ========================
print("\n🔍 Evaluating on validation set...")
y_true, y_pred = [], []

for X_batch, y_batch in val_loader:
    preds = model.predict(X_batch)
    preds = (preds > 0.5).astype("int32")
    y_true.extend(y_batch)
    y_pred.extend(preds.flatten())

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Non Accident", "Accident"]))
