import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from src.common.config import get_paths
from src.image.dataset import load_image_datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--img-size", type=int, nargs=2, default=(224, 224))
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    paths = get_paths()
    data_dir = args.data_dir or os.path.join(paths["data_root"], paths.get("image_data_dir", "traffic_binary"))
    train_ds, val_ds, test_ds = load_image_datasets(data_dir, tuple(args.img_size), args.batch_size)

    model_path = args.model or os.path.join(paths["models_root"], "image_model.h5")
    model = tf.keras.models.load_model(model_path)

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    y_true, y_pred, y_prob = [], [], []
    for images, labels in test_ds:
        preds = model.predict(images)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_prob.extend(preds)
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    report = classification_report(y_true, y_pred, target_names=train_ds.class_names)
    print("Classification Report:\n", report)

    if y_prob.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        print(f"ROC AUC: {auc(fpr, tpr):.4f}")


if __name__ == "__main__":
    main()
