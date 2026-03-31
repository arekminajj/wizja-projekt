import os
import random
from enum import Enum
from pathlib import Path

import cv2

from scripts.loaders.base_loader import BaseDatasetLoader
from scripts.vars import SEBASTEIN_MARCEL_IMAGES_PATH


class SebasteinMarcelEnum(Enum):
    A = 1
    B = 2
    C = 3
    FIVE = 4
    POINT = 5
    V = 6


class SebasteinMarcelDatasetLoader(BaseDatasetLoader):
    def get_learning_files(base_path=SEBASTEIN_MARCEL_IMAGES_PATH, limit=None, shuffle=True):
        image_files = []
        for root, _, files in os.walk(base_path):
            root = Path(root).resolve()
            for file in files:
                if not file.lower().endswith((".ppm")):
                    continue
                parent_dir_name = root.name.upper()

                if parent_dir_name not in SebasteinMarcelEnum.__members__:
                    continue

                label_int = SebasteinMarcelEnum[parent_dir_name].value
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                img_h, img_w = img.shape[:2]
                image_files.append(
                    (img_path, f"{label_int} (0 0) ({img_w} {img_h}) ({img_w} {img_h})", None)
                )

        if shuffle: random.shuffle(image_files)
        return image_files[:limit]
