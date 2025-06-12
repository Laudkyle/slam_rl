
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
    
    
# DEPTH ESTIMATION & OCCUPANCY GRIDS
class MiDaSDepthEstimator:
    def __init__(self, device='cuda', model_type='DPT_Large'):
        self.device = device
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True).to(device)
            self.model.eval()
            transforms_hub = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = transforms_hub.dpt_transform
            self.use_midas = True
        except:
            self.use_midas = False
            self._init_simple_model()

    def _init_simple_model(self):
        class SimpleDepthNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.ReLU(),
                    nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.ReLU(),
                    nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU())
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
                    nn.ConvTranspose2d(64,1,4,stride=2,padding=1), nn.Sigmoid())
            def forward(self,x): return self.decoder(self.encoder(x))
        self.model = SimpleDepthNet().to(self.device).eval()
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((256,512)),
                                             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    def estimate_depth(self, image):
        if isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = np.array(image)
        with torch.no_grad():
            if self.use_midas:
                input_batch = self.transform(image_rgb).to(self.device)
                prediction = self.model(input_batch)
                prediction = F.interpolate(prediction.unsqueeze(1), size=image_rgb.shape[:2], mode="bicubic", align_corners=False).squeeze()
                depth = prediction.cpu().numpy()
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            else:
                img_tensor = self.transform(Image.fromarray(image_rgb)).unsqueeze(0).to(self.device)
                depth = self.model(img_tensor).squeeze().cpu().numpy()
        return depth

depth_estimator = MiDaSDepthEstimator(device=device)
depth_maps = [depth_estimator.estimate_depth(f) for f in frames]

class SmartOccupancyGridBuilder:
    def __init__(self, grid_size=(30,40), close_threshold=0.65):
        self.grid_size = grid_size
        self.close_threshold = close_threshold

    def depth_to_occupancy(self, depth_map):
        depth_resized = cv2.resize(depth_map, (self.grid_size[1], self.grid_size[0]), interpolation=cv2.INTER_NEAREST)
        occupancy = (depth_resized > self.close_threshold).astype(np.float32)
        floor_region = occupancy[int(self.grid_size[0]*0.7):, :]
        floor_region *= 0.3
        return occupancy

grid_builder = SmartOccupancyGridBuilder(grid_size=(30,40))
occupancy_grids = [grid_builder.depth_to_occupancy(d) for d in depth_maps]

