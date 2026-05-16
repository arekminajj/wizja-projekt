import json
import os
from collections import defaultdict

import cv2

from method.method import Method
from method.method_payload import MethodPayload
from scripts.gestures import Gesture10

DATASET_PATH = os.path.relpath("../NUS-Hand-Posture-Dataset-II/Hand Postures")
TEST_FILES_PATH = "test_files.json"


def classification_test(show_images=False):
    with open(TEST_FILES_PATH, "r") as f:
        test_files = [tuple(pair) for pair in json.load(f)]

    correct = 0
    total = len(test_files)
    results = defaultdict(lambda: {"correct": 0, "total": 0})

    for image_path, label_int in test_files:
        image = cv2.imread(image_path)
        payload = MethodPayload(image=image)

        true_label = Gesture10(label_int)
        predicted_label, certainty = Method.classify(
            payload=payload,
            custom_model_path=".",
        )

        is_correct = predicted_label == true_label
        if is_correct:
            correct += 1

        results[true_label]["correct"] += int(is_correct)
        results[true_label]["total"] += 1

        if show_images:
            status = "OK" if is_correct else "WRONG"
            label = f"[{status}] Predicted: {predicted_label} ({certainty}%) | Actual: {true_label}"
            print(f"{label} | File: {image_path}")
            cv2.imshow(label, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    _print_summary(correct, total, results)


def _print_summary(correct: int, total: int, results: dict) -> None:
    accuracy = correct / total * 100 if total > 0 else 0
    print("\n" + "=" * 52)
    print(f"  Overall accuracy: {correct}/{total} = {accuracy:.2f}%")
    print("=" * 52)
    print("  Accuracy per class:")
    for gesture, stats in sorted(results.items(), key=lambda x: str(x[0])):
        class_acc = stats["correct"] / stats["total"] * 100
        bar = "█" * int(class_acc / 5)
        print(f"    {str(gesture):25s}: {stats['correct']:3d}/{stats['total']:3d} = {class_acc:5.1f}%  {bar}")
    print("=" * 52)


if __name__ == "__main__":
    classification_test()
