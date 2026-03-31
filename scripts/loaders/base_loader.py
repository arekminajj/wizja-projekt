from abc import ABC, abstractmethod


class BaseDatasetLoader(ABC):
    @staticmethod
    @abstractmethod
    def get_learning_files(**kwargs):
        """Return list of files for learning."""
        raise NotImplementedError
