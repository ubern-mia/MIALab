"""The data structure module holds model classes."""
from enum import Enum

import mialab.data.conversion as conversion


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
            images (dict): The images, where the key is a :py:class:`BrainImageTypes` and the value is a SimpleITK image.
        """

        self.id_ = id_
        self.path = path
        self.images = images

        # ensure we have an image to get the image properties
        if len(images) == 0:
            raise ValueError('No images provided')

        self.image_properties = conversion.ImageProperties(self.images[list(self.images.keys())[0]])
        self.feature_images = {}
        self.feature_matrix = None  # a tuple (features, labels),
        # where the shape of features is (n, number_of_features) and the shape of labels is (n, 1)
        # with n being the amount of voxels

