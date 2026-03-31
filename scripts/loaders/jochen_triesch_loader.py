import os
import random
from pathlib import Path

import cv2

from scripts.loaders.base_loader import BaseDatasetLoader
from scripts.vars import JOCHEN_TRIESCH_IMAGES_PATH


def map_label(label):
    # maps labels: {1, 2, 3, 4, 7, 8, 9, 12, 22, 25} to 1-10 to work correctly with Gesture10 enum.
    match label:
        case 25:
            return 5
        case 22:
            return 6
        case 12:
            return 10
        case _:
            return label


class JochenTrieschDatasetLoader(BaseDatasetLoader):
    def get_learning_files(base_path=JOCHEN_TRIESCH_IMAGES_PATH, limit=None, shuffle=True):
        image_files = []
        for root, _, files in os.walk(base_path):
            root = Path(root).resolve()
            for file in files:
                if not file.lower().endswith((".pgm")):
                    continue
                label = file[-6]
                if not label.isalpha():
                    continue
                label_int = ord(label.upper()) - 64
                label_int = map_label(label_int)
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                img_h, img_w = img.shape[:2]
                image_files.append(
                    (img_path, f"{label_int} (0 0) ({img_w} {img_h}) ({img_w} {img_h})", None)
                )

        if shuffle: random.shuffle(image_files)
        return image_files[:limit]
