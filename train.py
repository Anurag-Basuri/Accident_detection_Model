import tensorflow as tf
from models.cnn_lstm_efficientnet import build_model
from utils.data_loader import AccidentDataLoader

# Load dataset
train_loader = AccidentDataLoader("processed-datasets/train")
val_loader = AccidentDataLoader("processed-datasets/val")

# Build Model
model = build_model()
model.summary()

# Callbacks
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("model_checkpoints/best_model.h5", save_best_only=True)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=5)

# Train
history = model.fit(train_loader, validation_data=val_loader, epochs=20, callbacks=[checkpoint_callback, early_stopping])
