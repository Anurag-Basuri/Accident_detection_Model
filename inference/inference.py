import tensorflow as tf
from utils.data_loader import AccidentDataLoader

# Load Model
model = tf.keras.models.load_model("model_checkpoints/best_model.h5")

# Load Test Data
test_loader = AccidentDataLoader("processed-datasets/test")

# Evaluate
loss, acc = model.evaluate(test_loader)
print(f"Test Accuracy: {acc*100:.2f}%")
