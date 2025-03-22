import os
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np
from scipy import ndimage


class DatasetPreparer:

    def prepare_dataset(self, path_to_images: Path, mask_output_path: Path):
        mask_output_path.mkdir()
        for image_filename in tqdm(os.listdir(path_to_images), desc=f'Prepare masks', leave=True):
            img = cv2.imread(str(path_to_images / image_filename))
            mask = self._get_mask(img)
            np.save(mask_output_path / f'{image_filename.replace(".png", ".npy")}', mask)

    @staticmethod
    def _get_mask(img_to_mask):
        blue = (img_to_mask - np.array([40, 0, 0], dtype=np.uint8)).sum(axis=2) == 0
        yellow = (img_to_mask - np.array([0, 219, 219], dtype=np.uint8)).sum(axis=2) == 0
        score_mask = np.logical_or(blue, yellow)
        score_mask = ndimage.binary_dilation(score_mask, structure=ndimage.generate_binary_structure(2, 2)).astype(bool)
        result_mask = np.zeros_like(score_mask)

        labeled, n = ndimage.label(score_mask)
        for i in range(1, n + 1):
            m = np.equal(labeled, i)
            if m.sum() < 20 ** 2:
                continue
            idx = np.argwhere(m)
            x1, y1 = idx.min(axis=0)
            x2, y2 = idx.max(axis=0) + 1
            result_mask[x1:x2, y1:y2] = True

        return result_mask
