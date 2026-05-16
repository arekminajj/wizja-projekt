import os

import cv2
import numpy as np

from method.method import Method, extract_features
from method.method_payload import MethodPayload
from scripts.loaders import NUSIIDatasetLoader
from scripts.gestures import Gesture10

path = os.path.relpath("../NUS-Hand-Posture-Dataset-II/Hand Postures")

def image_processing_visual_test():
    files = NUSIIDatasetLoader.get_learning_files(base_path=path)

    for image_file in files:
        image = cv2.imread(image_file[0])
        F1, F2, F3, F4 = extract_features(image)

        row = np.hstack([
            cv2.cvtColor(cv2.resize(image, (64, 64)), cv2.COLOR_BGR2GRAY),
            F2, F3, F4,
        ])
        cv2.imshow("F1(orig)  F2(skin^sal)  F3(canny|hog)  F4(hand shape)", row)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def classification_visual_test():
    files = NUSIIDatasetLoader.get_learning_files(base_path='/home/arekminajj/studia/wizja/NUS-Hand-Posture-Dataset-II/Hand Postures')
    for image_file in files:
        image = cv2.imread(image_file[0])
        payload = MethodPayload(image=image)

        custom_options = {
            "gesture_enum": Gesture10
        }

        result, certainty = Method.classify(payload=payload
                                            , custom_model_path=".")

        cv2.imshow(f"Predicted gesture: {result} ({certainty}%", image)
        print(f"Predictet gesture: {result}, correct gesture: {image_file[1]}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    image_processing_visual_test()
    #classification_visual_test()
