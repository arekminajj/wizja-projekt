import os

from method.method_learning_data import MethodLearningData
from scripts.loaders import NUSIIDatasetLoader
from scripts.file_coords_parser import parse_file_coords, parse_etiquette
from scripts.gestures import Gesture10
from method.method import Method

files = NUSIIDatasetLoader.get_learning_files(base_path='/home/arekminajj/studia/wizja/NUS-Hand-Posture-Dataset-II/Hand Postures')

learning_data=list(map(lambda file: MethodLearningData(
    image_path=file[0], coords=parse_file_coords(file[1]), label=Gesture10(parse_etiquette(file[1]))
), files))

acc= Method.learn(
    learning_data=learning_data,
    custom_options={},
    target_model_path="."
)

print(acc)