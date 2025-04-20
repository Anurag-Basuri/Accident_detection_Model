import os
from ultralytics import YOLO

# Define paths
image_folder = r"D:\projects\accident_detection_app\data\image"
model_path = r"D:\projects\accident_detection_app\models\severity_model.pt"  # Update if needed

# Load YOLO model
model = YOLO(model_path)

# Get list of image files (filter only image formats)
image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)]

if not image_files:
    print("❌ No image files found in the folder.")
else:
    print(f"📁 Found {len(image_files)} image(s) to process...\n")

# Loop through each image and make predictions
for filename in image_files:
    image_path = os.path.join(image_folder, filename)
    results = model(image_path)
    pred = results[0]

    print(f"\n🖼️ Image: {filename}")
    if pred.boxes.shape[0] == 0:
        print("⚠️ No accident detected.")
    else:
        for i, box in enumerate(pred.boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            print(f"✅ Detected: {class_name} (Confidence: {conf:.2f})")
