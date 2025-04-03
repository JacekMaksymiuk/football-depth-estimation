import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from utils.misc import compute_scale_and_shift_np


class DatasetPreparer:

    DIFF_SCALAR = 1.

    MASKS_FOLDER_NAME = 'masks'
    DEPTHS_FOLDER_NAME = 'depths'
    DIFFS_FOLDER_NAME = 'diffs'

    def __init__(
            self,
            train_segments_path: Path,
            val_segments_path: Path,
            test_segments_path: Path,

            train_orig_depth_path: Path,
            val_orig_depth_path: Path,
            test_orig_depth_path: Path,

            train_pred_path: Path,
            val_pred_path: Path,
            test_pred_path: Path
    ):
        self._train_segments_path = train_segments_path
        self._val_segments_path = val_segments_path
        self._test_segments_path = test_segments_path

        self._train_orig_depth_path = train_orig_depth_path
        self._val_orig_depth_path = val_orig_depth_path
        self._test_orig_depth_path = test_orig_depth_path

        self._train_pred_path = train_pred_path
        self._val_pred_path = val_pred_path
        self._test_pred_path = test_pred_path

        self._train_mask_path = train_segments_path / self.MASKS_FOLDER_NAME
        self._val_mask_path = val_segments_path / self.MASKS_FOLDER_NAME
        self._test_mask_path = test_segments_path / self.MASKS_FOLDER_NAME

        train_filenames = os.listdir(train_segments_path)
        self._train_mask_path.mkdir()
        for filename in train_filenames:
            os.rename(train_segments_path / filename, self._train_mask_path / filename)

        val_filenames = os.listdir(val_segments_path)
        self._val_mask_path.mkdir()
        for filename in val_filenames:
            os.rename(val_segments_path / filename, self._val_mask_path / filename)

        test_filenames = os.listdir(test_segments_path)
        self._test_mask_path.mkdir()
        for filename in test_filenames:
            os.rename(test_segments_path / filename, self._test_mask_path / filename)

        self._train_depth_path = train_segments_path / self.DEPTHS_FOLDER_NAME
        self._val_depth_path = val_segments_path / self.DEPTHS_FOLDER_NAME
        self._test_depth_path = test_segments_path / self.DEPTHS_FOLDER_NAME
        self._train_depth_path.mkdir(), self._val_depth_path.mkdir(), self._test_depth_path.mkdir()

        self._train_diff_path = train_segments_path / self.DIFFS_FOLDER_NAME
        self._val_diff_path = val_segments_path / self.DIFFS_FOLDER_NAME
        self._test_diff_path = test_segments_path / self.DIFFS_FOLDER_NAME
        self._train_diff_path.mkdir(), self._val_diff_path.mkdir(), self._test_diff_path.mkdir()

    def prepare_dataset(self):
        self._prepare(self._train_segments_path, self._train_orig_depth_path, self._train_pred_path)
        self._prepare(self._val_segments_path, self._val_orig_depth_path, self._val_pred_path)
        self._prepare(self._test_segments_path, self._test_orig_depth_path, self._test_pred_path)

    def _prepare(self, ds_path: Path, orig_depth_path: Path, pred_path: Path):
        group_by_name = {}
        for filename in os.listdir(ds_path / self.MASKS_FOLDER_NAME):
            name = '_'.join(filename.split('_')[:3])
            if name not in group_by_name:
                group_by_name[name] = []
            group_by_name[name].append(filename)

        for group_name, filenames in tqdm(group_by_name.items(), desc=f'Process {ds_path.name}', leave=True):

            depth = cv2.imread(str(orig_depth_path / f'{group_name}.png'), cv2.IMREAD_UNCHANGED) / 255 ** 2
            pred = cv2.imread(str(pred_path / f'{group_name}.png'), cv2.IMREAD_UNCHANGED) / 255 ** 2
            pred = pred - pred.min()
            pred = pred / pred.max()

            scale, shift = compute_scale_and_shift_np(depth, pred, np.ones_like(depth))
            scaled_depth = scale * depth + shift
            diff = scaled_depth - pred
            diff = diff * self.DIFF_SCALAR

            for filename in filenames:
                x1, y1, x2, y2 = map(int, filename.replace('.npy', '').split('_')[3:])
                crop_diff, crop_depth = diff[y1:y2, x1:x2], pred[y1:y2, x1:x2]
                np.save(str(ds_path / self.DIFFS_FOLDER_NAME / filename), crop_diff)
                np.save(str(ds_path / self.DEPTHS_FOLDER_NAME / filename), crop_depth)
