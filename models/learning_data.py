from enum import Enum


class LearningData:
    def __init__(self, image_path: str, label: Enum):
        self.image_path = image_path
        self.label = label
