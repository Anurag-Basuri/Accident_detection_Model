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
VAL_DIR = "processed-datasets/val"
BATCH_SIZE = 16
IMAGE_SIZE = (224, 224)

# ========================
# LOAD MODEL
# ========================
print("🔹 Loading model for evaluation...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# ========================
# VALIDATION DATA LOADER
# ========================
val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  # Ensures correct label matching
)

# ========================
# RUN EVALUATION
# ========================
print("🔍 Evaluating model on validation dataset...")
y_true = val_generator.classes
y_pred_probs = model.predict(val_generator, verbose=1)
y_pred = (y_pred_probs > 0.5).astype("int32")

# ========================
# SAVE EVALUATION RESULTS
# ========================
output_df = pd.DataFrame({"Filename": val_generator.filenames, "Actual": y_true, "Predicted": y_pred.flatten()})
output_df.to_csv("evaluation_results.csv", index=False)
print("✅ Evaluation results saved to evaluation_results.csv")

# ========================
# PRINT & SAVE CLASSIFICATION REPORT
# ========================
print("\n🔍 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=["Non Accident", "Accident"]))

# ========================
# CONFUSION MATRIX
# ========================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=["Non Accident", "Accident"], yticklabels=["Non Accident", "Accident"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("evaluation_confusion_matrix.png")
plt.show()

# ========================
# PLOT LOSS & ACCURACY CURVES
# ========================
history_path = "model_checkpoints/training_history.npy"

if os.path.exists(history_path):
    print("📊 Loading training history...")
    history = np.load(history_path, allow_pickle=True).item()
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.savefig("evaluation_curves.png")
    plt.show()
else:
    print("⚠️ Training history not found. Skipping loss/accuracy plot.")

print("✅ Evaluation complete.")
