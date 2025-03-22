import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DepthDataset(Dataset):

    def __init__(self, image_paths, depth_paths, mask_paths, transform):
        self._image_paths = [f'{image_paths}/{x}' for x in sorted(os.listdir(image_paths))]
        self._depth_paths = [f'{depth_paths}/{x}' for x in sorted(os.listdir(depth_paths))]
        self._mask_paths = [f'{mask_paths}/{x}' for x in sorted(os.listdir(mask_paths))]
        self._transform = transform

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        image = self.load_image(self._image_paths[idx], self._transform)
        depth = self.load_depth(self._depth_paths[idx])
        mask = self.load_mask(self._mask_paths[idx])
        return image, depth, mask

    @staticmethod
    def load_image(file_path: str, transform):
        raw_image = cv2.imread(file_path)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        image = transform({'image': image})['image']
        image = torch.from_numpy(image)
        return image

    @staticmethod
    def load_depth(file_path: str):
        depth = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        depth = cv2.resize(depth, (924, 518), interpolation=cv2.INTER_AREA)
        depth = depth / 1.0
        depth = torch.from_numpy(depth).unsqueeze(0)
        return depth

    @staticmethod
    def load_mask(file_path: str):
        mask = np.load(file_path)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return mask
