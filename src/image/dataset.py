import os
import tensorflow as tf
from typing import Tuple

# tolerate class name variants like "Non Accident", "non-accident" by mapping to "Non_Accident"
CLASS_NAME_MAP = {
    "non accident": "Non_Accident",
    "non-accident": "Non_Accident",
    "non_accident": "Non_Accident",
}


def load_image_datasets(root_dir: str, img_size: Tuple[int, int] = (224, 224), batch_size: int = 32):
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")
    test_dir = os.path.join(root_dir, "test")

    # Normalize class folder names if needed
    for split_dir in (train_dir, val_dir, test_dir):
        if os.path.isdir(split_dir):
            for name in list(os.listdir(split_dir)):
                src = os.path.join(split_dir, name)
                norm = CLASS_NAME_MAP.get(name.lower())
                if norm and os.path.isdir(src):
                    dst = os.path.join(split_dir, norm)
                    if not os.path.isdir(dst):
                        os.rename(src, dst)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, image_size=img_size, batch_size=batch_size, label_mode="categorical"
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir, image_size=img_size, batch_size=batch_size, label_mode="categorical"
    )
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir, image_size=img_size, batch_size=batch_size, label_mode="categorical"
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)
    test_ds = test_ds.prefetch(buffer_size=autotune)

    return train_ds, val_ds, test_ds
