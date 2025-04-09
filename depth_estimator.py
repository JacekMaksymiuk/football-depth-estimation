from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from da_fine_tuning.dataset_loader import DepthDataset
from da_fine_tuning.fine_tune_depth_anything import DepthAnythingFineTuner
from depth_anything.dpt import DepthAnything
from players_fine_tuning.dataset_loader import PlayerDataset
from players_fine_tuning.model import EnhancedUNet
from players_fine_tuning.segment import PlayerSegmenter, PlayerMask


class DepthEstimator:

    _HF_REPO = 'JacekMa/football-depth-estimation'
    _HF_TOKEN = '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'

    def __init__(
            self, player_ft_model_path: str | None = None, depth_anything_model_path: str | None = None,
            device: str = 'cuda'):
        self._depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14')
        self._da_ft_path = depth_anything_model_path or self._download_model_from_hf('da_ft_epoch24.pth')
        self._depth_anything.load_state_dict(torch.load(self._da_ft_path))
        self._depth_anything.to(device).eval()
        self._da_transform = DepthAnythingFineTuner.prepare_transform()

        self._segmenter = PlayerSegmenter()

        self._player_fine_tuner = EnhancedUNet()
        self._player_ft_path = player_ft_model_path or self._download_model_from_hf('player_ft.pth')
        self._player_fine_tuner.load_state_dict(torch.load(self._player_ft_path))
        self._player_fine_tuner.to(device).eval()

    def predict(self, img_path: str, output_path: str):
        torch_img_to_pred = DepthDataset.load_image(img_path, self._da_transform)
        torch_img_to_pred = torch_img_to_pred.unsqueeze(0).to('cuda')
        with torch.no_grad():
            depth = self._depth_anything(torch_img_to_pred)

        h, w = torch_img_to_pred.shape[2:]
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = depth.cpu().numpy()

        player_masks: list[PlayerMask] = self._segmenter.segment_one(Path(img_path))
        for player_mask in player_masks:
            x1, y1, x2, y2 = player_mask.x1, player_mask.y1, player_mask.x2, player_mask.y2
            depth_crop = depth[y1:y2, x1:x2].copy()

            player_depth = PlayerDataset.np_img_to_input_tensor(depth_crop, PlayerDataset.SIZE)
            player_depth = player_depth.unsqueeze(0).unsqueeze(0).to('cuda')
            with torch.no_grad():
                pred = self._player_fine_tuner.predict(player_depth)

            pred = F.interpolate(pred, (y2 - y1, x2 - x1), mode='bilinear', align_corners=False)[0, 0]
            pred = pred.cpu().numpy()

            depth[y1:y2, x1:x2][player_mask.mask] += pred[player_mask.mask] / 2

        depth = depth.clip(depth, 0., 1.)
        depth = np.round(depth * 255 ** 2).astype(np.uint16)
        cv2.imwrite(output_path, depth)
        
    def _download_model_from_hf(self, filename: str) -> str:
        return hf_hub_download( repo_id=self._HF_REPO, filename=filename, token=self._HF_TOKEN)
