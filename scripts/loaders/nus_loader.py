import os
import random
import re
from pathlib import Path

import cv2

from scripts.loaders.base_loader import BaseDatasetLoader
from scripts.vars import NUS_IMAGES_PATH


class NUSDatasetLoader(BaseDatasetLoader):
    def get_learning_files(base_path=NUS_IMAGES_PATH, limit=None, shuffle=True):
        image_files = []
        for root, _, files in os.walk(base_path):
            for file in files:
                if file.endswith(".jpg"):
                    match = re.match(r"g(\d+)", file)
                    if match:
                        label_int = int(match.group(1))
                        root = Path(root).resolve()
                        img_path = os.path.join(root, file)
                        img = cv2.imread(img_path)
                        img_h, img_w = img.shape[:2]
                        image_files.append(
                            (img_path, f"{label_int} (0 0) ({img_w} {img_h}) ({img_w} {img_h})", None)
                        )

        if shuffle: random.shuffle(image_files)
        return image_files[:limit]
