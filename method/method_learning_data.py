from enum import Enum

from models.learning_data import LearningData


class MethodLearningData(LearningData):
    def __init__(self, image_path: str, coords: list[tuple[int, int]], label: Enum):
        super().__init__(image_path, label)
        self.coords = coords
