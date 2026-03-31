import os
import random
from pathlib import Path

import cv2

from scripts.loaders.base_loader import BaseDatasetLoader
from scripts.vars import NUS_II_IMAGES_PATH


class NUSIIDatasetLoader(BaseDatasetLoader):
    def get_learning_files(base_path=NUS_II_IMAGES_PATH, limit=None, shuffle=True):
        image_files = []
        for root, _, files in os.walk(base_path):
            root = Path(root).resolve()
            for file in files:
                if not file.lower().endswith((".jpg")):
                    continue
                label = file[0]
                if not label.isalpha():
                    continue
                label_int = ord(label.upper()) - 64
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                img_h, img_w = img.shape[:2]
                image_files.append(
                    (img_path, f"{label_int} (0 0) ({img_w} {img_h}) ({img_w} {img_h})", None)
                )

        if shuffle: random.shuffle(image_files)
        return image_files[:limit]
