import tensorflow as tf
from tensorflow.keras import layers, models


def build_mobilenet_v2(num_classes: int = 2, img_size=(224, 224)) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    model = models.Sequential([
        layers.Rescaling(1./127.5, offset=-1),
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"]) 
    return model
