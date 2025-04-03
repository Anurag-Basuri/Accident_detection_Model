import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential

def build_model(sequence_length=10, img_size=(224, 224, 3)):
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=img_size)
    base_model.trainable = False  # Freeze EfficientNet for transfer learning

    model = Sequential([
        TimeDistributed(base_model, input_shape=(sequence_length, *img_size)),
        TimeDistributed(Conv2D(32, (3,3), activation='relu')),
        TimeDistributed(MaxPooling2D(2,2)),
        TimeDistributed(Flatten()),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model
