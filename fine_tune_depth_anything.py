import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything.dataset_loader import DepthDataset
from depth_anything.dpt import DepthAnything
from depth_anything.transform import Resize, NormalizeImage, PrepareForNet


class RSELoss(torch.nn.Module):

    def __init__(self):
        super(RSELoss, self).__init__()

    def forward(self, pred, target):
        return torch.mean(((target - pred) ** 2) / target)


class DepthAnythingFineTuner:

    def __init__(self, depth_train_path: Path, depth_val_path: Path, depth_test_path: Path):
        self._depth_train_path = depth_train_path
        self._depth_val_path = depth_val_path
        self._depth_test_path = depth_test_path
        self._image_train_path = depth_train_path.parent / depth_train_path.name.replace('depths', 'images')
        self._image_val_path = depth_val_path.parent / depth_val_path.name.replace('depths', 'images')
        self._image_test_path = depth_test_path.parent / depth_test_path.name.replace('depths', 'images')

        self._depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14')
        self._depth_anything.to('cuda').eval()
        self._transform = self._prepare_transform()

    def fine_tune(self, n_epochs: int = 18, batch_size: int = 4):
        self._adjust_depths_to_pred()

        train_ds = DepthDataset(self._image_train_path, self._depth_train_path, transform=self._transform)
        val_ds = DepthDataset(self._image_val_path, self._depth_val_path, transform=self._transform)
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

        loss_fn = RSELoss()
        optimizer = optim.AdamW(self._depth_anything.parameters(), lr=5e-6, weight_decay=0.01)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda e: 1.0 - (e / (n_epochs - 1)) * 0.9)

        for epoch in range(n_epochs):

            # Train
            self._depth_anything.train()
            total_loss, cnt = 0.0, 0
            train_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{n_epochs}', leave=True)
            for images, depths in train_progress_bar:
                images, depths = images.to('cuda'), depths.to('cuda')
                predictions = self._depth_anything(images)
                loss = loss_fn(predictions, depths)
                loss = loss / batch_size  # New
                cnt += 1
                loss.backward()

                if cnt % batch_size == 0 or cnt == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * batch_size
                train_progress_bar.set_postfix(loss=loss.item() * batch_size, avg_loss=total_loss / cnt)

            scheduler.step()
            print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss / len(train_loader):.4f}')

            ### Val
            self._depth_anything.eval()
            total_err, total_cnt, total_err_sq = 0.0, 0.0, 0
            val_progress_bar = tqdm(os.listdir(self._depth_val_path), desc=f"Val error", leave=True)
            for filename in val_progress_bar:
                img_path = str(self._image_val_path / filename)
                np_orig = cv2.imread(str(self._depth_val_path / filename), cv2.IMREAD_UNCHANGED)
                h, w = np_orig.shape[:2]

                with torch.no_grad():
                    img_to_pred = DepthDataset.load_image(img_path, self._transform).unsqueeze(0).to('cuda')
                    depth = self._depth_anything(img_to_pred)
                depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0 ** 2
                depth = depth.cpu().numpy().astype(np.uint16)

                scale, shift = self._compute_scale_and_shift_np(depth, np_orig)
                depth = scale * depth + shift

                total_err_sq += np.mean(((np_orig - depth) ** 2) / np_orig)
                total_err += self._silog(depth, np_orig)
                total_cnt += 1
                val_progress_bar.set_postfix(silog_err=total_err / total_cnt, rse=total_err_sq / total_cnt)
            print(f"Val loss: silog: {(total_err / total_cnt):.4f}, sq: {(total_err_sq / total_cnt):.4f}")

    @staticmethod
    def _prepare_transform():
        return Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    @staticmethod
    def _compute_scale_and_shift_np(prediction, target):
        a_00 = np.sum(prediction * prediction, (0, 1))
        a_01 = np.sum(prediction, (0, 1))
        a_11 = np.sum(np.ones(prediction.shape), (0, 1))
        b_0 = np.sum(prediction * target, (0, 1))
        b_1 = np.sum(target, (0, 1))
        x_0 = np.zeros_like(b_0)
        x_1 = np.zeros_like(b_1)
        det = a_00 * a_11 - a_01 * a_01
        valid = det > 0
        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]
        return x_0, x_1

    def _adjust_depths_to_pred(self, thresh=255.):
        filenames = sorted(os.listdir(self._depth_train_path))
        random.shuffle(filenames)
        ch_progress_bar = tqdm(filenames, desc=f"Adjust to pred", leave=True)
        for img_filename in ch_progress_bar:
            img_path = str(self._image_train_path / img_filename)
            depth_path = str(self._depth_train_path / img_filename)
            np_orig_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 255 ** 2
            h, w = np_orig_depth.shape[:2]

            with torch.no_grad():
                img_to_pred = DepthDataset.load_image(img_path, self._transform).unsqueeze(0).to('cuda')
                pred_depth = self._depth_anything(img_to_pred)

            pred_depth = F.interpolate(pred_depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
            pred_depth = pred_depth.cpu().numpy() * 255.

            scale, shift = self._compute_scale_and_shift_np(np_orig_depth, pred_depth)
            new_np_orig_depth = scale * np_orig_depth + shift

            new_np_orig_depth[new_np_orig_depth < thresh] = thresh
            new_np_orig_depth[new_np_orig_depth > 255 ** 2 - thresh] = 255 ** 2 - thresh

            new_np_orig_depth = new_np_orig_depth.astype(np.uint16)
            cv2.imwrite(depth_path, new_np_orig_depth)

    @staticmethod
    def _silog(pred: np.ndarray, target: np.ndarray, epsilon=1e-6):
        pred = np.clip(pred, epsilon, None)
        target = np.clip(target, epsilon, None)
        log_diff = np.log(pred) - np.log(target)
        silog_err = np.sqrt(np.mean(log_diff ** 2) - (np.mean(log_diff) ** 2))
        return silog_err * 100
