import os
import json
from collections import defaultdict
import cv2

from method.method import Method
from method.method_payload import MethodPayload
from method.skin_mlp import SkinMLP
from scripts.file_coords_parser import parse_file_coords, parse_etiquette
from scripts.loaders import  NUSIIDatasetLoader
from time import sleep
from scripts.gestures import Gesture10

path = os.path.relpath("../NUS-Hand-Posture-Dataset-II/Hand Postures")

test_files_path = "test_files.json"
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


def classification_visual_test(v):
    with open(test_files_path, "r") as f:
        test_files = [tuple(pair) for pair in json.load(f)]
    view=v
    correct = 0
    total = len(test_files)
    results = defaultdict(lambda: {"correct": 0, "total": 0})

    for file_entry in test_files:
        image_path = file_entry[0]
        coords_path = file_entry[1]
        image = cv2.imread(image_path)
        payload = MethodPayload(image=image)

        true_label = Gesture10(parse_etiquette(coords_path))
        predicted_label, certainty = Method.classify(
            payload=payload,
            custom_model_path=".",
            custom_options={"gesture_enum": Gesture10}
        )
 
        is_correct = (predicted_label == true_label)
        if is_correct:
            correct += 1
 
        results[true_label]["correct"] += int(is_correct)
        results[true_label]["total"]   += 1
        if view!=0:
            # Wizualna inspekcja każdego obrazu testowego
            status = "OK" if is_correct else "BŁĄD"
            cv2.imshow(
                f"[{status}] Predykcja: {predicted_label} ({certainty:.1f}%) | Rzeczywista: {true_label}",
                image
            )
            print(
                f"[{status}] Predykcja: {predicted_label} ({certainty:.1f}%) | "
                f"Rzeczywista: {true_label} | Plik: {image_path}"
            )
 
            cv2.waitKey(0)
            cv2.destroyAllWindows()
 
    _print_summary(correct, total, results)


def _print_summary(correct: int, total: int, results: dict) -> None:
    accuracy = correct / total * 100 if total > 0 else 0
    print("\n" + "=" * 52)
    print(f"  DOKŁADNOŚĆ OGÓLNA: {correct}/{total} = {accuracy:.2f}%")
    print("=" * 52)
    print("  Dokładność per klasa:")
    for gesture, stats in sorted(results.items(), key=lambda x: str(x[0])):
        class_acc = stats["correct"] / stats["total"] * 100
        bar = "█" * int(class_acc / 5)  # prosta wizualizacja w konsoli
        print(f"    {str(gesture):25s}: {stats['correct']:3d}/{stats['total']:3d} = {class_acc:5.1f}%  {bar}")
    print("=" * 52)
 
if __name__ == "__main__":
    image_processing_visual_test()
    classification_visual_test(0)
