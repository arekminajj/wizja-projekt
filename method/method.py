import os
import pickle
from enum import Enum
from typing import Tuple

import cv2
import numpy as np
from numpy import ndarray
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from method.method_payload import MethodPayload
from scripts.gestures import Gesture10
from definitions import ROOT_DIR

class Method():
    @staticmethod
    def process_image(payload: MethodPayload) -> np.ndarray:
        image = payload.image

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        thresh = cv2.bitwise_not(thresh)

        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return gray

        hand_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(hand_contour)
        hand = gray[y:y+h, x:x+w]

        hand = cv2.resize(hand, (64, 64))

        hand = hand.flatten()

        return hand

    @staticmethod
    def learn(learning_data: list, target_model_path: str, custom_options: dict = None) -> float:

        X, y = [], []

        for data in learning_data:
            image = cv2.imread(data.image_path)
            processed_image = Method.process_image(MethodPayload(image))

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
        processed_image = processed_image.reshape(1, -1)

        proba = model.predict_proba(processed_image)[0]
        predicted_label = np.argmax(proba)
        certainty = int(np.max(proba) * 100)

        return Gesture10(predicted_label + 1), certainty
