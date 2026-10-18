import json
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class LegoDataset(Dataset):
    def __init__(self, data_dir="./data/lego", split="train", image_size=400, max_files: int | None = None):
        self.data_dir = Path(data_dir)
        json_path = self.data_dir / f"transforms_{split}.json"

        with open(json_path, "r") as f:
            meta = json.load(f)

        self.camera_angle_x = meta["camera_angle_x"]
        self.frames = meta["frames"]
        self.image_size = image_size

        self.image_tensors = []
        self.poses = []

        files_to_load = min(max_files, len(self.frames)) if max_files else len(self.frames)
        print(f'Loading {files_to_load}/{len(self.frames)} images for split "{split}" from {data_dir}...')

        file_count = 0
        for frame in self.frames:
            if file_count >= files_to_load:
                break
            img_relative_path = frame["file_path"]
            _, img_tensor = self.load_image(self.data_dir / img_relative_path, image_size=image_size)

            self.image_tensors.append(img_tensor)
            self.poses.append(torch.tensor(frame["transform_matrix"], dtype=torch.float32))

            file_count += 1

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.image_tensors[idx], self.poses[idx]

    def load_image(self, img_path: Path | str, image_size=400):
        # Load Image
        img_path = str(img_path)
        if not img_path.endswith(".png"):
            img_path += ".png"
        image = Image.open(img_path).convert("RGBA")

        # Downscale for faster run on laptop (original 800)
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)

        # Convert to tensor with range [0, 1]
        img_tensor = torch.tensor(np.array(image), dtype=torch.float32) / 255.0

        # Make background white
        img_tensor = img_tensor[..., :3] * img_tensor[..., 3:] + (1.0 - img_tensor[..., 3:])

        return image, img_tensor
