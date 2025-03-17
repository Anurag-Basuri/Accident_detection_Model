import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, TimeDistributed, GlobalAveragePooling2D, Dropout, Input)

# Model Hyperparameters
SEQUENCE_LENGTH = 10  # Number of frames per video segment
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3

def build_cnn_lstm():
    """
    Builds the CNN-LSTM model for accident detection.
    - CNN (EfficientNetV2) extracts spatial features.
    - LSTM captures temporal dependencies.
    """

    # EfficientNetV2B0 as feature extractor
    base_cnn = EfficientNetV2B0(weights="imagenet", include_top=False, 
                                input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    base_cnn.trainable = False  # Freeze CNN weights

    cnn_output = GlobalAveragePooling2D()(base_cnn.output)
    feature_extractor = Model(inputs=base_cnn.input, outputs=cnn_output)

    # LSTM Model
    inputs = Input(shape=(SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, CHANNELS))

    # Apply CNN feature extractor on each frame
    x = TimeDistributed(feature_extractor)(inputs)

    # LSTM layers for temporal dependencies
    x = LSTM(512, return_sequences=True)(x)
    x = LSTM(256, return_sequences=False)(x)
    x = Dropout(0.5)(x)

    # Fully connected layers
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)  # Binary classification

    # Compile the model
    model = Model(inputs, output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss="binary_crossentropy", 
                  metrics=["accuracy"])

    return model

# Instantiate and print model summary
cnn_lstm_model = build_cnn_lstm()
cnn_lstm_model.summary()
