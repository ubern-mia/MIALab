"""The data structure module holds model classes."""
from enum import Enum


class BrainImageTypes(Enum):
    """Represents the image types."""
    T1 = 1
    T2 = 2
    GroundTruth = 3
    BrainMask = 4


class BrainImage:
    """Represents a brain image."""
    
    def __init__(self, id_: str, path: str, images: dict):
        """Initializes a new instance of the BrainImage class.

        Args:
            id_ (str): An identifier.
            path (str): Full path to the image directory.
            images (dict): The images, where the key is a BrainImageTypes and the value is a SimpleITK image.
        """

        self.id_ = id_
        self.path = path
        self.images = images
        self.feature_matrix = None
