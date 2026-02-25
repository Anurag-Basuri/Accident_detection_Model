import os
import numpy as np
from PIL import Image
import cv2

from src.services.image_service import predict_image
from src.services.video_service import predict_video


def create_dummy_image(path: str):
    img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype('uint8'))
    img.save(path)


def create_dummy_video(path: str, frames: int = 16):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 10.0, (112, 112))
    for _ in range(frames):
        frame = (np.random.rand(112, 112, 3) * 255).astype('uint8')
        out.write(frame)
    out.release()


def test_image_service_tmp():
    os.makedirs('temp', exist_ok=True)
    img_path = os.path.join('temp', 'dummy.jpg')
    create_dummy_image(img_path)
    try:
        res = predict_image(img_path)
        assert 'class_index' in res and 'score' in res
    finally:
        os.remove(img_path)


def test_video_service_tmp():
    os.makedirs('temp', exist_ok=True)
    vid_path = os.path.join('temp', 'dummy.mp4')
    create_dummy_video(vid_path)
    try:
        res = predict_video(vid_path)
        assert 'class_index' in res and 'score' in res
    finally:
        os.remove(vid_path)
