import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    TimeDistributed, LSTM, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Input
)


class CNN_LSTM_EfficientNet:
    def __init__(self, input_shape=(None, 224, 224, 3), num_classes=2, lstm_units=256, dropout_rate=0.3):
        """
        Initializes the CNN-LSTM model using EfficientNet as a feature extractor.

        Args:
            input_shape (tuple): Shape of the input video frame sequence (timesteps, height, width, channels).
            num_classes (int): Number of output classes (e.g., accident vs. non-accident).
            lstm_units (int): Number of LSTM units for temporal modeling.
            dropout_rate (float): Dropout rate for regularization.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = self.build_model()

    def build_model(self):
        """
        Builds the CNN-LSTM model.
        """
        # Define input layer (sequence of frames)
        video_input = Input(shape=self.input_shape, name="video_input")

        # Load EfficientNetB0 as feature extractor (without top layers)
        base_cnn = EfficientNetB0(weights="imagenet", include_top=False)

        # Make the EfficientNet layers non-trainable (fine-tuning can be enabled later)
        for layer in base_cnn.layers:
            layer.trainable = False

        # Apply CNN to each frame (TimeDistributed)
        cnn_features = TimeDistributed(base_cnn, name="cnn_feature_extractor")(video_input)
        cnn_features = TimeDistributed(GlobalAveragePooling2D(), name="global_avg_pool")(cnn_features)

        # LSTM for temporal feature modeling
        x = LSTM(self.lstm_units, return_sequences=False, name="lstm_layer")(cnn_features)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)

        # Fully connected layer
        x = Dense(128, activation="relu", name="dense_128")(x)
        x = Dropout(self.dropout_rate)(x)

        # Output layer
        output = Dense(self.num_classes, activation="softmax", name="output_layer")(x)

        # Create model
        model = Model(inputs=video_input, outputs=output, name="CNN_LSTM_EfficientNet")

        return model

    def compile_model(self, learning_rate=1e-4):
        """
        Compiles the model with an optimizer and loss function.

        Args:
            learning_rate (float): Learning rate for the optimizer.
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

    def summary(self):
        """
        Prints the model architecture.
        """
        self.model.summary()


# Example usage
if __name__ == "__main__":
    model = CNN_LSTM_EfficientNet(input_shape=(30, 224, 224, 3), num_classes=2)
    model.compile_model()
    model.summary()
