import os
import pickle
from enum import Enum
from typing import Tuple

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from method.method_payload import MethodPayload
from scripts.gestures import Gesture10
from definitions import ROOT_DIR


def _skin_mask(img: np.ndarray) -> np.ndarray:
    # params source: https://github.com/CHEREF-Mehdi/SkinDetection
    # https://www.sciencedirect.com/science/article/abs/pii/S0262885620300573?via%3Dihub
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lo, hi = np.array([0,   15,  0]), np.array([17,  170, 255])
    return cv2.inRange(hsv, lo, hi)


def _saliency_u8(img: np.ndarray) -> np.ndarray:
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    _, sal_map = saliency.computeSaliency(img)
    return (sal_map * 255).astype(np.uint8)


def _hog_image(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, hog_vis = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2), visualize=True)
    if hog_vis.max() > 0:
        return (hog_vis / hog_vis.max() * 255).astype(np.uint8)
    return np.zeros_like(gray, dtype=np.uint8)


def extract_features(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    F1 = original grayscale
    F2 = skin AND saliency        — saliency gated to skin regions
    F3 = Canny OR HOG             — combined edge/gradient structure
    F4 = (F2 AND F3) XOR skin     — hand shape refined by skin
    """
    img = cv2.resize(img, (64, 64))

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    skin  = _skin_mask(img)
    sal   = _saliency_u8(img)
    canny = cv2.Canny(gray, 50, 150)
    hog_img = _hog_image(img)

    F1 = gray
    F2 = cv2.bitwise_and(skin, sal)
    F3 = cv2.bitwise_or(canny, hog_img)
    F4 = cv2.bitwise_xor(cv2.bitwise_and(F2, F3), skin)

    return F1, F2, F3, F4


class Method:
    @staticmethod
    def process_image(payload: MethodPayload) -> np.ndarray:
        _, F2, F3, F4 = extract_features(payload.image)
        return np.stack([F2, F3, F4], axis=-1)

    def learn(learning_data: list, target_model_path: str, custom_options: dict = None) -> float:
        X, y = [], []
        total = len(learning_data)
        for i, data in enumerate(learning_data, 1):
            print(f"\r{i}/{total}", end="", flush=True)
            processed = Method.process_image(MethodPayload(image=cv2.imread(data.image_path)))
            X.append(processed.flatten())
            y.append(data.label.value - 1)
        print()

        svm = SVC(probability=True, kernel='linear')
        print("Uczenie SVM...")
        svm.fit(X, y)
        print("Gotowe!")

        with open(os.path.join(target_model_path, 'method_svm.pkl'), 'wb') as f:
            pickle.dump(svm, f)

        return accuracy_score(y, svm.predict(X))

    @staticmethod
    def classify(payload: MethodPayload, custom_model_path=None,
                 custom_options: dict = None) -> Tuple[Enum, int]:
        model_dir = custom_model_path if custom_model_path is not None else os.path.join(ROOT_DIR, "sgrf_trained_models")

        with open(os.path.join(model_dir, 'method_svm.pkl'), 'rb') as f:
            model = pickle.load(f)

        processed = Method.process_image(payload=payload).flatten().reshape(1, -1)

        proba = model.predict_proba(processed)[0]
        predicted_label = np.argmax(proba)
        certainty = int(np.max(proba) * 100)

        return Gesture10(predicted_label + 1), certainty
