import copy
from pathlib import Path

import torch
from tqdm import tqdm

import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from players_fine_tuning.dataset_loader import PlayerDataset
from players_fine_tuning.dataset_preparer import DatasetPreparer
from players_fine_tuning.model import EnhancedUNet


class MaskedMSE(nn.Module):

    def __init__(self):
        super(MaskedMSE, self).__init__()

    def forward(self, pred, target, mask=None):
        pred, target = pred.unsqueeze(0), target.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)
        return ((pred - target) ** 2)[mask].mean()

class PlayerFineTuner:

    def __init__(self, train_path: Path, val_path: Path, device: str = 'cuda'):
        self._device = device
        self._model = EnhancedUNet()
        self._model.to(device).eval()

        self._best_model, self._best_score = None, 9999999.

        self._train_path = train_path
        self._val_path = val_path

    def _get_datasets(self):
        train_ds = PlayerDataset(
            image_paths=self._train_path / DatasetPreparer.DEPTHS_FOLDER_NAME,
            diff_paths=self._train_path / DatasetPreparer.DIFFS_FOLDER_NAME,
            mask_paths=self._train_path / DatasetPreparer.MASKS_FOLDER_NAME
        )
        val_ds = PlayerDataset(
            image_paths=self._val_path / DatasetPreparer.DEPTHS_FOLDER_NAME,
            diff_paths=self._val_path / DatasetPreparer.DIFFS_FOLDER_NAME,
            mask_paths=self._val_path / DatasetPreparer.MASKS_FOLDER_NAME
        )
        return train_ds, val_ds

    def fine_tune(self, n_epochs: int = 15, batch_size: int = 32, use_scheduler: bool = False):
        train_ds, val_ds = self._get_datasets()
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

        loss_fn = MaskedMSE()
        optimizer = optim.AdamW(self._model.parameters(), lr=1e-3)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda e: 1.0 - (e / (n_epochs - 1)) * 0.9) if use_scheduler else None

        for epoch in range(n_epochs):
            self._model.train()

            total_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=True)

            cnt = 0
            for inputs, targets, masks in progress_bar:
                inputs, targets, masks = inputs.to(self._device), targets.to(self._device), masks.to(self._device)
                optimizer.zero_grad()
                predictions = self._model(inputs)
                loss = loss_fn(predictions, targets, masks)
                cnt += 1
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item(), avg_loss=total_loss / cnt)

            if use_scheduler:
                scheduler.step()
            print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss / len(train_loader):.4f}')

            self._model.eval()
            val_progress_bar = tqdm(val_loader, desc=f"Val error", leave=True)
            val_loss, val_cnt, ratio_ok_loss, ratio_nok_loss, ratio = 0, 0, 0, 0, 0
            for inputs, targets, masks in val_progress_bar:
                inputs, targets, masks = inputs.to(self._device), targets.to(self._device), masks.to(self._device)
                with torch.no_grad():
                    predictions = self._model(inputs)
                loss = (predictions - targets) ** 2
                loss = loss[masks].mean()
                val_loss += loss.item()
                val_cnt += 1
                val_progress_bar.set_postfix(loss=val_loss / val_cnt)
            score = val_loss / val_cnt
            if score < self._best_score:
                self._best_score = score
                self._best_model = copy.deepcopy(self._model)
            print(f'Val loss: {(val_loss / val_cnt):.8f}, {self._best_score=}')

    def save(self, path_to_save: Path):
        torch.save(self._best_model.state_dict(), path_to_save)

    def predict_tmp(self, player_depth_path: Path):
        import numpy as np
        player_depth = PlayerDataset.load_image(player_depth_path, PlayerDataset.SIZE)
        player_depth = player_depth.unsqueeze(0).to(self._device)
        self._model.eval()
        with torch.no_grad():
            pred = self._model(player_depth)
        pred = pred.squeeze(0).cpu().numpy()
        pred = pred * 255 + 127
        return pred.astype(np.uint8)
