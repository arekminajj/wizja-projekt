import os
import random
from enum import Enum
from pathlib import Path

import cv2

from scripts.loaders.base_loader import BaseDatasetLoader
from scripts.vars import LEAP_MOTION_IMAGES_PATH


class LeapMotionEnum(Enum):
    PALM = 1
    L = 2
    FIST = 3
    FIST_MOVED = 4
    THUMB = 5
    INDEX = 6
    OK = 7
    PALM_MOVED = 8
    C = 9
    DOWN = 10


class LeapMotionDatasetLoader(BaseDatasetLoader):
    def get_learning_files(base_path=LEAP_MOTION_IMAGES_PATH, limit=None, shuffle=True):
        image_files = []
        for root, _, files in os.walk(base_path):
            root = Path(root).resolve()
            for file in files:
                if not file.lower().endswith((".png")):
                    continue
                parent_dir_name = root.name.upper()
                parent_dir_name = parent_dir_name[3:]

                if parent_dir_name not in LeapMotionEnum.__members__:
                    continue

                label_int = LeapMotionEnum[parent_dir_name].value
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                img_h, img_w = img.shape[:2]
                image_files.append(
                    (img_path, f"{label_int} (0 0) ({img_w} {img_h}) ({img_w} {img_h})", None)
                )

        if shuffle: random.shuffle(image_files)
        return image_files[:limit]
