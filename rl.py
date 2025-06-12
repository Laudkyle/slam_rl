# ============================================================================
# SECTION 1: SETUP, INSTALLATIONS & IMPORTS
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import warnings
warnings.filterwarnings('ignore')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def upload_video_colab():
    from google.colab import files
    print("Please upload your 15-second video...")
    uploaded = files.upload()
    if not uploaded:
        print("No file uploaded. Using demo video.")
        return None
    return list(uploaded.keys())[0]

def extract_frames_from_video(video_path, target_frames=200):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_interval = max(1, total_frames // target_frames)
    
    frames = []
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            if width > 1280:
                scale = 1280 / width
                frame = cv2.resize(frame, (1280, int(height*scale)))
            frames.append(frame)
    cap.release()
    print(f"Extracted {len(frames)} frames")
    return frames, fps

def create_demo_office_video():
    frames = []
    for i in range(40):
        img = np.ones((480, 640, 3), dtype=np.uint8) * 240
        cv2.rectangle(img, (0, 300), (640, 480), (200, 200, 200), -1)
        cv2.rectangle(img, (0, 0), (100, 480), (150, 150, 150), -1)
        cv2.rectangle(img, (540, 0), (640, 480), (150, 150, 150), -1)
        if i > 20:
            desk_x = 150 + (i - 20) * 5
            cv2.rectangle(img, (desk_x, 200), (desk_x + 80, 280), (139, 69, 19), -1)
            cv2.rectangle(img, (desk_x + 100, 220), (desk_x + 150, 290), (100, 100, 200), -1)
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(img)
    print(f"Created {len(frames)} demo frames")
    return frames

# Load frames
try:
    video_path = upload_video_colab()
    if video_path:
        frames, fps = extract_frames_from_video(video_path, target_frames=200)
    else:
        frames = create_demo_office_video()
        fps = 30.0
except Exception:
    frames = create_demo_office_video()
    fps = 30.0

