import os
import cv2
from pathlib import Path
from tqdm import tqdm

import numpy as np
from ultralytics import YOLO

class PlayerSegmenter:

    _PLAYER_CLASS_ID = 0

    def __init__(self, img_sz=(1088, 1920)):
        self._img_sz = img_sz
        self._model = YOLO("yolov8x-seg.pt").to('cuda')

    def segment(self, images_path: Path, output_path: Path):
        for image_filename in tqdm(os.listdir(images_path), desc=f'Process segmentation', leave=True):
            img_path = images_path / image_filename
            self._pred(img_path, output_path)

    def _pred(self, img_path: Path, output_path: Path):
        result = self._model(img_path, save=False, iou=0.25, verbose=False, conf=0.6, imgsz=self._img_sz)[0]
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        img_h, img_w = img.shape[:2]

        for e, box in enumerate(result.boxes):
            if int(box.cls) != self._PLAYER_CLASS_ID:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            seg = result.masks.data.cpu().numpy()[e]
            seg = cv2.resize(seg, (img_w, img_h))
            seg = cv2.dilate(seg, np.ones((3, 3), np.uint8), iterations=1)
            seg = seg[y1:y2, x1:x2] > 0.9
            seg = seg.astype(bool)

            new_filename = f"{img_path.name.replace('.png', '')}_{x1}_{y1}_{x2}_{y2}.npy"
            np.save(output_path / new_filename, seg)
