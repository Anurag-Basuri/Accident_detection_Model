import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense,
    Dropout, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau

def build_model(sequence_length=10, img_size=(224, 224, 3), fine_tune_at=100):
    """
    Builds a CNN-LSTM model with EfficientNetB0 as the feature extractor.
    
    Parameters:
    - sequence_length: Number of frames per sequence
    - img_size: Size of each input frame (224,224,3)
    - fine_tune_at: Number of layers to unfreeze for fine-tuning
    
    Returns:
    - A compiled TensorFlow Keras model
    """

    # Load EfficientNetB0 as a feature extractor
    base_model = EfficientNetB0(
    weights="/kaggle/input/efficient_net/other/default/1/efficientnetb0_notop.h5",
    include_top=False,
    input_shape=img_size
)

    base_model.trainable = False  # Freeze base model initially

    # Unfreeze some layers for fine-tuning
    for layer in base_model.layers[-fine_tune_at:]:
        layer.trainable = True

    model = Sequential([
        TimeDistributed(base_model, input_shape=(sequence_length, *img_size)),
        TimeDistributed(GlobalAveragePooling2D()),  # Reduce dimensionality
        TimeDistributed(BatchNormalization()),  # Normalization for stability

        # LSTM for temporal sequence learning
        LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
        LSTM(64, dropout=0.3, recurrent_dropout=0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(1, activation='sigmoid')  # Binary classification
    ])

    # Use AdamW optimizer with learning rate scheduler
    optimizer = AdamW(learning_rate=1e-4)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    model.summary()  # Print the model architecture
    return model
