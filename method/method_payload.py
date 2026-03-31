import numpy as np

from models.image_payload import ImagePayload

class MethodPayload(ImagePayload):
    def __init__(self, image: np.ndarray):
        super().__init__(image)
