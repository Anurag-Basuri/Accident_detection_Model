import os
import cv2
from clarifai_grpc.grpc.api import resources_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.service_pb2 import PostModelOutputsRequest
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from google.protobuf.struct_pb2 import Value

# 🪟 Fix for Windows: Clarifai SDK needs HOME
if "HOME" not in os.environ:
    os.environ["HOME"] = os.path.expanduser("~")

# 👤 Clarifai credentials
USER_ID = "dc1tjovav2po"
APP_ID = "123456"
PAT = "c6a6b4336e6c41f4b426ca12ed8e8cad"
MODEL_ID = "general-image-recognition"
MODEL_VERSION_ID = ""  # Leave empty for latest

# 🔗 Setup connection and auth
channel = ClarifaiChannel.get_grpc_channel()
stub = service_pb2_grpc.V2Stub(channel)

metadata = (("authorization", f"Key {PAT}"),)

userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)


# 🖼️ Frame extraction
def extract_frames(video_path, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (416, 416))
        frame_path = f"temp_frame_{count}.jpg"
        cv2.imwrite(frame_path, resized)
        frames.append(frame_path)
        count += 1

    cap.release()
    return frames


# 🔍 Predict using Clarifai
def predict_image(image_path):
    with open(image_path, "rb") as f:
        file_bytes = f.read()

    request = PostModelOutputsRequest(
        user_app_id=userDataObject,
        model_id=MODEL_ID,
        version_id=MODEL_VERSION_ID,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(
                    image=resources_pb2.Image(base64=file_bytes)
                )
            )
        ]
    )

    response = stub.PostModelOutputs(request, metadata=metadata)
    os.remove(image_path)

    if response.status.code != 10000:
        raise Exception(f"Prediction failed: {response.status.description}")

    return response.outputs[0].data.concepts


# 🚗 Analyze the video for accidents
def analyze_video(video_path):
    frames = extract_frames(video_path, max_frames=10)
    detected = False

    for i, frame_path in enumerate(frames):
        print(f"Analyzing Frame {i+1}...")
        try:
            concepts = predict_image(frame_path)

            for concept in concepts:
                print(f"{concept.name}: {concept.value:.2f}")
                if concept.name.lower() in ["accident", "crash", "collision"] and concept.value > 0.6:
                    print(f"⚠️ Accident detected in Frame {i+1} with confidence {concept.value:.2f}")
                    detected = True
                    break

            if detected:
                break

        except Exception as e:
            print(f"Error analyzing frame {i+1}: {e}")

    if not detected:
        print("✅ No accident detected in the video.")


# 🧪 Run
analyze_video("data\videos\collision_with_motorcycle_20.mp4")
