import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
from models.cnn_lstm_efficientnet import build_model  # Import the model

# ========================
# CONFIGURATION
# ========================
BATCH_SIZE = 16
SEQUENCE_LENGTH = 10  # Number of frames per video sequence
IMAGE_SIZE = (224, 224)
EPOCHS = 50
MODEL_SAVE_PATH = "model_checkpoints/best_model.h5"
TRAIN_DIR = "processed-datasets/train"
VAL_DIR = "processed-datasets/val"

# ========================
# DATA AUGMENTATION
# ========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255)  # No augmentation for validation

# ========================
# DATA GENERATORS
# ========================
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# ========================
# BUILD & COMPILE MODEL
# ========================
model = build_model(sequence_length=SEQUENCE_LENGTH, img_size=(224, 224, 3), fine_tune_at=100)

# ========================
# CALLBACKS
# ========================
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

callbacks = [checkpoint, early_stopping, reduce_lr]

# ========================
# TRAINING
# ========================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ========================
# EVALUATION
# ========================
y_true = val_generator.classes
y_pred = (model.predict(val_generator) > 0.5).astype("int32")

print("\n🔍 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=["Non Accident", "Accident"]))
