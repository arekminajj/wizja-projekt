import os

import cv2
import numpy as np

from method.method import Method, extract_features
from method.method_payload import MethodPayload
from scripts.gestures import Gesture10
from scripts.loaders import NUSIIDatasetLoader

DATASET_PATH = os.path.relpath("../NUS-Hand-Posture-Dataset-II/Hand Postures")


def preprocessing_visual_test():
    files = NUSIIDatasetLoader.get_learning_files(base_path=DATASET_PATH)

    for img_path, _ in files:
        image = cv2.imread(img_path)
        F1, F2, F3, F4 = extract_features(image)

        row = np.hstack([F1, F2, F3, F4])
        cv2.imshow("F1: original | F2: skin & saliency | F3: canny | hog | F4: hand shape", row)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def classification_visual_test():
    files = NUSIIDatasetLoader.get_learning_files(base_path=DATASET_PATH)

    for img_path, label_int in files:
        image = cv2.imread(img_path)
        predicted, certainty = Method.classify(
            payload=MethodPayload(image=image),
            custom_model_path=".",
        )
        correct = Gesture10(label_int)

        print(f"Predicted: {predicted.name} ({certainty}%)  |  Correct: {correct.name}")
        cv2.imshow(f"Predicted: {predicted.name} ({certainty}%)  |  Correct: {correct.name}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    preprocessing_visual_test()
    # classification_visual_test()
