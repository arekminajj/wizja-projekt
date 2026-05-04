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


class Method():
    @staticmethod
    def process_image(payload: MethodPayload) -> np.ndarray:
        img = payload.image

        img = cv2.resize(img, (100, 100))

        # Segmentacja skóry
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        skin_lower = np.array([0, 20, 70], dtype=np.uint8)
        skin_upper = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv_img, skin_lower, skin_upper)

        # Operacje morfologiczne
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        eroded_mask = cv2.erode(mask, h_kernel, iterations=1)

        sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
        closed_mask = cv2.morphologyEx(eroded_mask, cv2.MORPH_CLOSE, sq_kernel)

        # Aproksymacja wielokątna
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        approx_mask = np.zeros_like(mask)
        for contour in contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(approx_mask, [approx_contour], -1, (255,), thickness=cv2.FILLED)

        # apply mask
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.bitwise_and(gray_img, gray_img, mask=mask)

        return result


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
        processed_image = processed_image.reshape(1, -1)

        proba = model.predict_proba(processed_image)[0]
        predicted_label = np.argmax(proba)
        certainty = int(np.max(proba) * 100)

        return Gesture10(predicted_label + 1), certainty
