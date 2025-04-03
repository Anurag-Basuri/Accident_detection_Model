import os
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ========================
# CONFIGURATION
# ========================
MODEL_PATH = "model_checkpoints/best_model.h5"
TEST_DIR = "processed-datasets/test"
BATCH_SIZE = 16
IMAGE_SIZE = (224, 224)

# ========================
# LOAD MODEL
# ========================
print("🔹 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# ========================
# TEST DATA LOADER
# ========================
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  # Ensures correct label matching
)

# ========================
# RUN INFERENCE
# ========================
print("🔍 Running inference on test dataset...")
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = (y_pred_probs > 0.5).astype("int32")

# ========================
# SAVE PREDICTIONS
# ========================
output_df = pd.DataFrame({"Filename": test_generator.filenames, "Actual": y_true, "Predicted": y_pred.flatten()})
output_df.to_csv("predictions.csv", index=False)
print("✅ Predictions saved to predictions.csv")

# ========================
# EVALUATION METRICS
# ========================
print("\n🔍 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=["Non Accident", "Accident"]))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=["Non Accident", "Accident"], yticklabels=["Non Accident", "Accident"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

print("✅ Inference complete.")
