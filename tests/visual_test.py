import os

import cv2

from method.method import Method
from method.method_payload import MethodPayload
from method.skin_mlp import SkinMLP
from scripts.file_coords_parser import parse_file_coords, parse_etiquette
from scripts.loaders import  NUSIIDatasetLoader
from time import sleep
from scripts.gestures import Gesture10

path = os.path.relpath("../NUS-Hand-Posture-Dataset-II/Hand Postures")

def image_processing_visual_test():
    skin_mlp_path = os.path.join(".", "skin_mlp.pkl")
    skin_mlp = SkinMLP.load(skin_mlp_path) if os.path.exists(skin_mlp_path) else None

    files = NUSIIDatasetLoader.get_learning_files(base_path=path)

    for image_file in files:
        image = cv2.imread(image_file[0])

        payload = MethodPayload(image=image)

        processed_image = Method.process_image(payload, skin_mlp=skin_mlp)

        cv2.imshow("Before Image", image)
        cv2.imshow("Processed Image", processed_image)


        cv2.waitKey(30000)
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
