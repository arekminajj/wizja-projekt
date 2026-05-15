import os
import pickle
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from method.method_payload import MethodPayload
from method.skin_mlp import SkinMLP
from scripts.gestures import Gesture10
from definitions import ROOT_DIR


def _hsv_skin_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([0, 20, 70], dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8))


class Method():
    @staticmethod
    def process_image(payload: MethodPayload, skin_mlp: Optional[SkinMLP] = None) -> np.ndarray:
        img = payload.image
        img = cv2.resize(img, (400, 400))

        if skin_mlp is not None:
            mask = skin_mlp.segment(img)
        else:
            mask = _hsv_skin_mask(img)

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        eroded_mask = cv2.erode(mask, h_kernel, iterations=1)

        sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
        closed_mask = cv2.morphologyEx(eroded_mask, cv2.MORPH_CLOSE, sq_kernel)

        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return closed_mask

        approx_mask = np.zeros_like(closed_mask)
        for contour in contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(approx_mask, [approx_contour], -1, (255,), thickness=cv2.FILLED)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.bitwise_and(gray_img, gray_img, mask=approx_mask)


    def learn(learning_data: list, target_model_path: str, custom_options: dict = None) -> float:
        train_images = []
        for data in learning_data:
            image = cv2.imread(data.image_path)
            train_images.append(cv2.resize(image, (400, 400)))

        skin_mlp = SkinMLP()
        skin_mlp.train(train_images, [_hsv_skin_mask(img) for img in train_images])
        skin_mlp.save(os.path.join(target_model_path, 'skin_mlp.pkl'))

        X, y = [], []
        for data, image in zip(learning_data, train_images):
            processed = Method.process_image(MethodPayload(image), skin_mlp=skin_mlp)
            X.append(processed.flatten())
            y.append(data.label.value - 1)

        svm = SVC(probability=True, kernel='linear')
        svm.fit(X, y)

        with open(os.path.join(target_model_path, 'method_svm.pkl'), 'wb') as f:
            pickle.dump(svm, f)

        return accuracy_score(y, svm.predict(X))

    @staticmethod
    def classify(payload: MethodPayload, custom_model_path=None,
                 custom_options: dict = None) -> Tuple[Enum, int]:

        model_dir = custom_model_path if custom_model_path is not None else os.path.join(ROOT_DIR, "sgrf_trained_models")

        skin_mlp_path = os.path.join(model_dir, 'skin_mlp.pkl')
        skin_mlp = SkinMLP.load(skin_mlp_path) if os.path.exists(skin_mlp_path) else None

        with open(os.path.join(model_dir, 'method_svm.pkl'), 'rb') as f:
            model = pickle.load(f)

        processed = Method.process_image(payload=payload, skin_mlp=skin_mlp)
        processed = processed.flatten().reshape(1, -1)

        proba = model.predict_proba(processed)[0]
        predicted_label = np.argmax(proba)
        certainty = int(np.max(proba) * 100)

        return Gesture10(predicted_label + 1), certainty
