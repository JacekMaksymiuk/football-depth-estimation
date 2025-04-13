import os
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm


class DataPreparer:

    _HB_REPO_ID = 'SoccerNet/SN-Depth-2025'

    TEST = 0
    VALID = 1
    TRAIN = 2
    ALL = [TEST, VALID, TRAIN]

    _HF_NAMES = {
        TEST: 'test',
        VALID: 'valid',
        TRAIN: 'train'
    }

    _NAMES = {
        TEST: 'Test',
        VALID: 'Validation',
        TRAIN: 'Train'
    }
    
    def __init__(self, workdir: Path):
        self._workdir = workdir

    def prepare(self, datasets: list[int] = None):
        for dataset in datasets or self.ALL:
            _, depths_path = self._download_and_extract_dataset(dataset)
            self._fix_depth_for_dataset(depths_path)
            
    def _download_and_extract_dataset(self, dataset_type: int) -> tuple[Path, Path]:
        print(f'Downloading {self._HF_NAMES[dataset_type]} dataset')
        zip_filename = self._download_from_hf(dataset_type)

        print(f'Extracting {self._HF_NAMES[dataset_type]} dataset')
        with zipfile.ZipFile(self._workdir / zip_filename, 'r') as zip_ref:
            zip_ref.extractall(self._workdir)
        os.remove(self._workdir / zip_filename)

        extracted_path = self._workdir / self._NAMES[dataset_type]
        prefix = self._HF_NAMES[dataset_type]
        images_path = self._workdir / f'{prefix}_images'
        depths_path = self._workdir / f'{prefix}_depths'

        os.mkdir(str(images_path))
        os.mkdir(str(depths_path))
        for game in tqdm(os.listdir(extracted_path), desc=f'Process games', leave=True):
            p = extracted_path / game / 'video_1'
            depths = set(os.listdir(p / 'depth_r'))
            for filename in os.listdir(p / 'color'):
                if not filename.endswith('.png') or not filename[:-4].isdigit():
                    continue
                if filename not in depths:
                    continue
                shutil.copy(str(p / 'color' / filename), str(images_path / f'{game}_{filename}'))
                shutil.copy(str(p / 'depth_r' / filename), str(depths_path / f'{game}_{filename}'))

        shutil.rmtree(str(extracted_path))
        return images_path, depths_path

    def _download_from_hf(self, dataset_type: int) -> str:
        zip_filename = f'{self._HF_NAMES[dataset_type]}.zip'
        hf_hub_download(
            repo_id=self._HB_REPO_ID,
            filename=zip_filename,
            repo_type='dataset',
            local_dir=str(self._workdir)
        )
        return zip_filename

    def _fix_depth_for_dataset(self, path: Path):
        for filename in tqdm(os.listdir(path), desc=f'Fix depths in {path}', leave=True):
            self._fix_depth_file(str(path / filename))

    @staticmethod
    def _fix_depth_file(img_path: str, n_parts: int = 10):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        rows_value = np.mean(img, axis=1)
        rows_value_split = np.array_split(rows_value, n_parts)
        rows_value_means = np.array([np.mean(part) for part in rows_value_split])
        to_color_revert = (np.diff(rows_value_means) > 0).sum() >= n_parts // 2

        if to_color_revert:
            reverted_img = np.iinfo(img.dtype).max - img
            cv2.imwrite(img_path, reverted_img)
