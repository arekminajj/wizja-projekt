import os
import pickle
from enum import Enum

import cv2
import numpy as np
from numpy import ndarray
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from method.method_payload import MethodPayload
from scripts.gestures import Gesture10
from definitions import ROOT_DIR





class Method():
    @staticmethod
    def process_image(payload: MethodPayload) -> ndarray:
        image = payload.image

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image

    @staticmethod
    def learn(learning_data: list, target_model_path: str, custom_options: dict = None) -> (float, float):

        X, y = [], []

        for data in learning_data:
            image = cv2.imread(data.image_path)
            processed = Method.process_image(MethodPayload(image))
            features = processed.flatten()

            X.append(features)
            y.append(data.label.value - 1)

        X, y = np.array(X), np.array(y)
        pca = PCA(n_components=0.95, svd_solver='full')

        svm = SVC(kernel='rbf', probability=True)
        pipeline = make_pipeline(StandardScaler(), pca, svm)

        pipeline.fit(X, y)

        y_pred = pipeline.predict(X)
        accuracy = accuracy_score(y, y_pred)

        os.makedirs(target_model_path, exist_ok=True)
        model_path = os.path.join(target_model_path, "method.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)

        return accuracy, None

    @staticmethod
    def classify(payload: MethodPayload, custom_model_path=None,
                 custom_options: dict = None) -> (Enum, int):
        options = {
            "gesture_enum": Gesture10
        }
        gesture_enum = options['gesture_enum']

        model_path = os.path.join(custom_model_path or os.path.join(ROOT_DIR, "bdgs_trained_models"), "method.pkl")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        processed_image = Method.process_image(payload)
        features = processed_image.flatten().reshape(1, -1)

        prediction = model.predict(features)
        proba = model.predict_proba(features)
        confidence = round(100 * np.max(proba), 0)

        return gesture_enum(prediction[0] + 1), confidence
