import os

import cv2

from method.method import Method
from method.method_payload import MethodPayload
from scripts.file_coords_parser import parse_file_coords, parse_etiquette
from scripts.loaders import BDGSDatasetLoader, NUSDatasetLoader, NUSIIDatasetLoader, JochenTrieschDatasetLoader, SebasteinMarcelDatasetLoader
from time import sleep
from scripts.gestures import Gesture10


def image_processing_visual_test():
    files = NUSDatasetLoader.get_learning_files(base_path='/home/arekminajj/bdgs/bdgs_third_party_datasets/NUS-Hand-Posture-Dataset')

    for image_file in files:
        image = cv2.imread(image_file[0])

        payload = MethodPayload(image=image)

        processed_image = Method.process_image(payload)

        cv2.imshow("Before Image", image)
        cv2.imshow("Processed Image", processed_image)


        cv2.waitKey(30000)
        cv2.destroyAllWindows()


def classification_visual_test():
    files = NUSDatasetLoader.get_learning_files(base_path='/home/arekminajj/bdgs/bdgs_third_party_datasets/NUS-Hand-Posture-Dataset')
    for image_file in files:
        image = cv2.imread(image_file[0])
        payload = MethodPayload(image=image)

        custom_options = {
            "gesture_enum": Gesture10
        }

        result, certainty = Method.classify(payload=payload
                                            , custom_model_path=".")

        cv2.imshow(f"Predicted gesture: {result} ({certainty}%", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    #image_processing_visual_test()
    classification_visual_test()
