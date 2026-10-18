import json
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class LegoDataset(Dataset):
    def __init__(self, data_dir="./data/lego", split="train", image_size=400):
        self.data_dir = Path(data_dir)
        json_path = self.data_dir / f"transforms_{split}.json"

        with open(json_path, "r") as f:
            meta = json.load(f)

        self.camera_angle_x = meta["camera_angle_x"]
        self.frames = meta["frames"]
        self.image_size = image_size

        self.image_tensors = []
        self.poses = []

        print(f'Loading {len(self.frames)} images for split "{split}" from {data_dir}...')

        for frame in self.frames:
            img_relative_path = frame["file_path"]
            if not img_relative_path.endswith(".png"):
                img_relative_path += ".png"
            img_path = self.data_dir / img_relative_path

            # Load image (already RGBA but just to be sure)
            image = Image.open(img_path).convert("RGBA")

            # Downscale for faster run on laptop (original 800)
            image = image.resize((self.image_size, self.image_size))

            # Convert to tensor with range [0, 1]
            img_tensor = torch.tensor(np.array(image), dtype=torch.float32) / 255.0

            # Make background white
            img_tensor = img_tensor[..., :3] * img_tensor[..., 3:] + (1.0 - img_tensor[..., 3:])

            self.image_tensors.append(img_tensor)
            self.poses.append(torch.tensor(frame["transform_matrix"], dtype=torch.float32))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.image_tensors[idx], self.poses[idx]
