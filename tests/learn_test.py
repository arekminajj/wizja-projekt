import json
import os
import shutil
from sklearn.model_selection import train_test_split

from method.method import Method
from method.method_learning_data import MethodLearningData
from scripts.gestures import Gesture10
from scripts.loaders import NUSIIDatasetLoader

def export_for_yolo_classification(files_list, split_name, output_dir="./yolo/yolo_gestures_cls"):
    print(f"Eksportowanie zbioru '{split_name}' dla YOLO Classification...")
    for img_path, label in files_list:
        class_name = str(label).split('.')[-1] if hasattr(label, 'name') else str(label)
        dest_dir = os.path.join(output_dir, split_name, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(img_path, os.path.join(dest_dir, os.path.basename(img_path)))

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
export_for_yolo_classification(train_files, "train")
export_for_yolo_classification(test_files, "val")
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
