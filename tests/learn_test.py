import os
import json

from method.method_learning_data import MethodLearningData
from scripts.loaders import NUSIIDatasetLoader
from scripts.file_coords_parser import parse_file_coords, parse_etiquette
from scripts.gestures import Gesture10
from method.method import Method

from sklearn.model_selection import train_test_split

path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "NUS-Hand-Posture-Dataset-II",
    "Hand Postures"
)

test_files_path = "test_files.json"
all_files = NUSIIDatasetLoader.get_learning_files(base_path=path)

labels = [parse_etiquette(f[1])for f in all_files]

train_files, test_files = train_test_split(
    all_files,
    test_size=0.2,      # 80% trening / 20% test
    random_state=42,    # reprodukowalność — ten sam podział przy każdym uruchomieniu
    stratify=labels     # zachowuje proporcje klas w obu zbiorach
)

with open(test_files_path, "w") as f:
    json.dump([list(pair) for pair in test_files],f,indent=2)


learning_data = list(map(
    lambda file: MethodLearningData(
        image_path=file[0],
        coords=parse_file_coords(file[1]),
        label=Gesture10(parse_etiquette(file[1]))
    ),
    train_files
))

acc= Method.learn(
    learning_data=learning_data,
    custom_options={},
    target_model_path="."
)

print(acc)