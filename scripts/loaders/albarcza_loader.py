import os
import random
from pathlib import Path

import cv2

from scripts.loaders.base_loader import BaseDatasetLoader
from scripts.vars import ALBARCZA_IMAGES_PATH


# Dataset contains gestues with labels 0-9 and a-z
# 0-9 get 0-9 lables mapped, a-z get: a - 10; b - 11 and so.
class AlbarczaDatasetLoader(BaseDatasetLoader):
    def get_learning_files(base_path=ALBARCZA_IMAGES_PATH, limit=None, shuffle=True):
        image_files = []
        for root, _, files in os.walk(base_path):
            root = Path(root).resolve()
            for file in files:
                if not file.lower().endswith((".png")):
                    continue
                label = file[6]
                if label.isalpha():
                    label_int = ord(label.upper()) - 64 + 9
                else:
                    label_int = int(label) + 1
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                img_h, img_w = img.shape[:2]
                image_files.append(
                    (img_path, f"{label_int} (0 0) ({img_w} {img_h}) ({img_w} {img_h})", None)
                )

        if shuffle: random.shuffle(image_files)

        return image_files[:limit]
