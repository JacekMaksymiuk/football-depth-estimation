import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


class PlayerDataset(Dataset):

    _SIZE = (128, 64)

    def __init__(self, image_paths: Path, diff_paths: Path, mask_paths: Path, no_masking: bool = False):
        self.mask_paths = [mask_paths / x for x in sorted(os.listdir(mask_paths))]
        self.image_paths = [image_paths / x for x in sorted(os.listdir(image_paths))]
        self.depth_paths = [diff_paths / x for x in sorted(os.listdir(diff_paths))]
        self._no_masking = no_masking

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.load_image(self.image_paths[idx], size=self._SIZE)
        depth = self.load_diff(self.depth_paths[idx], size=self._SIZE)
        mask = self.load_mask(self.mask_paths[idx], size=self._SIZE)
        if self._no_masking:
            mask = torch.ones_like(mask)

        image, depth, mask = image.unsqueeze(0), depth.unsqueeze(0), mask.unsqueeze(0)
        return image, depth, mask

    @staticmethod
    def load_image(file_path, size):
        raw_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(raw_image, (size[1], size[0]), interpolation=cv2.INTER_AREA)
        image = image - image.min()
        image = image / image.max()
        return torch.from_numpy(image).float()

    @staticmethod
    def load_diff(file_path, size):
        diff = np.load(file_path)
        diff = torch.from_numpy(diff)
        diff = F.interpolate(diff.unsqueeze(0).unsqueeze(0), size, mode='bilinear', align_corners=False)
        return diff.squeeze(0).squeeze(0)

    @staticmethod
    def load_mask(file_path, size):
        mask = np.load(file_path)
        mask = torch.from_numpy(mask)
        mask = F.interpolate(mask.float().unsqueeze(0).unsqueeze(0), size, mode='bilinear', align_corners=False)
        return mask.squeeze(0).squeeze(0) > 0.5
