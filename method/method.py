import os
import pickle
from enum import Enum
from typing import Tuple

import cv2
import numpy as np
from numpy import ndarray
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from method.method_payload import MethodPayload
from scripts.gestures import Gesture10
from definitions import ROOT_DIR

def _skin_segmentation(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    return skin_mask


def _morphological_processing(mask: np.ndarray) -> np.ndarray:
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    eroded = cv2.erode(mask, horizontal_kernel, iterations=1)

    square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, square_kernel)

    return closed


def _polygonal_approximation(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approx_mask = np.zeros_like(mask)

    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(approx_mask, [approx], -1, (255,), thickness=cv2.FILLED)

    return approx_mask


def _apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = cv2.bitwise_and(gray, gray, mask=mask)
    return result


class Method():
    @staticmethod
    def process_image(payload: MethodPayload) -> np.ndarray:
        image = payload.image

        image = cv2.resize(image, (100, 100))

        skin_mask = _skin_segmentation(image)
        skin_mask = _morphological_processing(skin_mask)
        skin_mask = _polygonal_approximation(skin_mask)
        masked_image = _apply_mask(image, skin_mask)

        return masked_image


    def learn(learning_data: list, target_model_path: str, custom_options: dict = None) -> float:

        X, y = [], []

        for data in learning_data:
            image = cv2.imread(data.image_path)
            processed_image = Method.process_image(MethodPayload(image))
            processed_image = processed_image.flatten()
            #processed_image = processed_image.reshape(1, -1)

            X.append(processed_image)
            y.append(data.label.value - 1)

        svm = SVC(probability=True, kernel='linear')
        svm.fit(X, y)

        model_path = os.path.join(target_model_path, 'method_svm.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(svm, f)

        y_pred = svm.predict(X)
        accuracy = accuracy_score(y, y_pred)

        return accuracy

    @staticmethod
    def classify(payload: MethodPayload, custom_model_path=None,
                 custom_options: dict = None) -> Tuple[Enum, int]:

        model_filename = "method_svm.pkl"
        model_path = os.path.join(custom_model_path, model_filename) if custom_model_path is not None else os.path.join(
            ROOT_DIR, "sgrf_trained_models",
            model_filename)

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        processed_image = Method.process_image(payload=payload)
        processed_image = processed_image.flatten()
        #processed_image = processed_image.reshape(1, -1)

        proba = model.predict_proba(processed_image)[0]
        predicted_label = np.argmax(proba)
        certainty = int(np.max(proba) * 100)

        return Gesture10(predicted_label + 1), certainty
