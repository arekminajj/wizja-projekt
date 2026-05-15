import os
import pickle

import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier


def load_manual_masks(image_paths: list, masks_dir: str) -> tuple:
    images, masks = [], []
    for path in image_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        mask_path = os.path.join(masks_dir, stem + "_mask.png")
        if not os.path.exists(mask_path):
            continue
        img = cv2.imread(path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        images.append(cv2.resize(img, (400, 400)))
        masks.append(cv2.resize(mask, (400, 400)))
    return images, masks


class SkinMLP:
    def __init__(self):
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(5, 10),
            activation='logistic',
            max_iter=2000,
            random_state=42,
        )

    def train(self, images: list, masks: list):
        X, y = [], []
        rng = np.random.default_rng(42)
        for img, mask in zip(images, masks):
            pixels = img.reshape(-1, 3).astype(np.float32) / 255.0
            labels = (mask.reshape(-1) > 0).astype(int)
            skin_idx = np.where(labels == 1)[0]
            nonskin_idx = np.where(labels == 0)[0]
            if len(skin_idx) == 0 or len(nonskin_idx) == 0:
                continue
            n_sample = min(500, len(skin_idx), len(nonskin_idx))
            idx = np.concatenate([
                rng.choice(skin_idx, n_sample, replace=False),
                rng.choice(nonskin_idx, n_sample, replace=False),
            ])
            X.append(pixels[idx])
            y.append(labels[idx])
        if not X:
            raise ValueError("No valid training samples for SkinMLP")
        self.mlp.fit(np.vstack(X), np.concatenate(y))

    def _forward(self, pixels: np.ndarray) -> np.ndarray:
        a = pixels
        for coef, intercept in zip(self.mlp.coefs_[:-1], self.mlp.intercepts_[:-1]):
            a = 1.0 / (1.0 + np.exp(-(a @ coef + intercept)))
        a = 1.0 / (1.0 + np.exp(-(a @ self.mlp.coefs_[-1] + self.mlp.intercepts_[-1])))
        return a.ravel() >= 0.5

    def segment(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        pixels = image.reshape(-1, 3).astype(np.float32) / 255.0
        return self._forward(pixels).reshape(h, w).astype(np.uint8) * 255

    def segment_all(self, images: list) -> list:
        """Segment a list of same-size images in one batched forward pass."""
        h, w = images[0].shape[:2]
        n = len(images)
        pixels = np.stack(images).reshape(-1, 3).astype(np.float32) / 255.0
        preds = self._forward(pixels).reshape(n, h, w).astype(np.uint8) * 255
        return [preds[i] for i in range(n)]

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.mlp, f)

    @classmethod
    def load(cls, path: str) -> 'SkinMLP':
        instance = cls()
        with open(path, 'rb') as f:
            instance.mlp = pickle.load(f)
        return instance
