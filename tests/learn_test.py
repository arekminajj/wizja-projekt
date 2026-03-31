import os

from method.method_learning_data import MethodLearningData
from scripts.loaders import NUSDatasetLoader
from scripts.file_coords_parser import parse_file_coords, parse_etiquette
from scripts.gestures import Gesture10
from method.method import Method

files = NUSDatasetLoader.get_learning_files(base_path='/home/arekminajj/bdgs/bdgs_third_party_datasets/NUS-Hand-Posture-Dataset')


learning_data=list(map(lambda file: MethodLearningData(
    image_path=file[0], coords=parse_file_coords(file[1]), label=Gesture10(parse_etiquette(file[1]))
), files))

acc, loss = Method.learn(
    learning_data=learning_data,
    custom_options={},
    target_model_path="."
)

print(acc)