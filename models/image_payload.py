from numpy import ndarray


class ImagePayload:
    def __init__(self, image: ndarray):
        self.image = image
