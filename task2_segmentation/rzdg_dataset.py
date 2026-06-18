import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class RZDGSegmentation(Dataset):
    def __init__(self, root, split="train", transform=None):
        """
        root: path to RZDG_real_seg
        split: train / val / test
        """
        self.root = root
        self.split = split
        self.transform = transform

        self.img_dir = os.path.join(root, "img_dir", split)
        self.ann_dir = os.path.join(root, "ann_dir", split)

        self.images = sorted(os.listdir(self.img_dir))
        
        print(f"RZDG {split}: Found {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(self.ann_dir, img_name)

        # Load RGB image
        image = Image.open(img_path).convert("RGB")

        # Load segmentation mask (already in correct format: 0, 1, 2)
        mask = Image.open(ann_path)

        # Apply transforms BEFORE converting to numpy
        if self.transform:
            image, mask = self.transform(image, mask)

        # Now convert mask to numpy/tensor
        # IMPORTANT: Convert to numpy AFTER transform
        mask = np.array(mask, dtype=np.int64)
        
        # Verify values are in valid range
        if mask.max() >= 3:
            print(f"WARNING: {img_name} has invalid mask value {mask.max()}")
            print(f"Unique values: {np.unique(mask)}")
            # Clip to valid range
            mask = np.clip(mask, 0, 2)
        
        mask = torch.from_numpy(mask)

        return image, mask