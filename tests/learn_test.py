import json
import os

from sklearn.model_selection import train_test_split

from method.method import Method
from method.method_learning_data import MethodLearningData
from scripts.gestures import Gesture10
from scripts.loaders import NUSIIDatasetLoader

DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "NUS-Hand-Posture-Dataset-II", "Hand Postures"
)
TEST_FILES_PATH = "test_files.json"

all_files = NUSIIDatasetLoader.get_learning_files(base_path=DATASET_PATH)
labels = [label for _, label in all_files]

train_files, test_files = train_test_split(
    all_files,
    test_size=0.2,
    random_state=42,
    stratify=labels,
)

with open(TEST_FILES_PATH, "w") as f:
    json.dump([list(pair) for pair in test_files], f, indent=2)

learning_data = [
    MethodLearningData(image_path=img_path, label=Gesture10(label))
    for img_path, label in train_files
]

acc = Method.learn(
    learning_data=learning_data,
    custom_options={},
    target_model_path=".",
)

print(acc)
