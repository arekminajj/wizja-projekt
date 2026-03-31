import os
import random
from pathlib import Path

import cv2

from scripts.loaders.base_loader import BaseDatasetLoader
from scripts.vars import THOMAS_MOESLUND_IMAGES_PATH


# Dataset needs to be preprocessed before loading:
# The original database is a gzip archive and since get_learning_files should return 
# path for each file the database needs to be uncompressed before loading.
# Also all images with label 'ae'* need to be renamed to 'j'* for ex. ae1.tiff to j1.tiff.


class ThomasMoeslundDatasetLoader(BaseDatasetLoader):
    def get_learning_files(base_path=THOMAS_MOESLUND_IMAGES_PATH, limit=None, shuffle=True):
        image_files = []
        for root, _, files in os.walk(base_path):
            root = Path(root).resolve()
            for file in files:
                if file.endswith(".tif"):
                    label = file[0]
                    label_int = ord(label.upper()) - 64
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img_h, img_w = img.shape[:2]
                    image_files.append(
                        (img_path, f"{label_int} (0 0) ({img_w} {img_h}) ({img_w} {img_h})", None)
                    )

        if shuffle: random.shuffle(image_files)
        return image_files[:limit]
