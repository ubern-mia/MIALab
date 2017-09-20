"""The data structure module holds model classes."""
from enum import Enum

import SimpleITK as sitk

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


class PicklableBrainImage:
    """Represents a brain image that can be pickled."""

    def __init__(self, id_: str, path: str, np_images: dict, image_properties):
        """Initializes a new instance of the BrainImage class.

        Args:
            id_ (str): An identifier.
            path (str): Full path to the image directory.
            np_images (dict): The images, where the key is a :py:class:`BrainImageTypes` and the value is a numpy image.
        """

        self.id_ = id_
        self.path = path
        self.np_images = np_images
        self.image_properties = image_properties
        self.np_feature_images = {}
        self.feature_matrix = None  # a tuple (features, labels),
        # where the shape of features is (n, number_of_features) and the shape of labels is (n, 1)
        # with n being the amount of voxels


class BrainImageToPicklableBridge:
    """A :class:`BrainImage` to :class:`PicklableBrainImage` bridge."""

    @staticmethod
    def convert(brain_image: BrainImage)-> PicklableBrainImage:
        """Converts a :class:`BrainImage` to :class:`PicklableBrainImage`.

        Args:
            brain_image (BrainImage): A brain image.

        Returns:
            PicklableBrainImage: The pickable brain image.
        """

        np_images = {}
        for key, img in brain_image.images.items():
            np_images[key] = sitk.GetArrayFromImage(img)
        np_feature_images = {}
        for key, feat_img in brain_image.feature_images.items():
            np_feature_images[key] = sitk.GetArrayFromImage(feat_img)

        pickable_brain_image = PicklableBrainImage(brain_image.id_, brain_image.path, np_images,
                                                   brain_image.image_properties)
        pickable_brain_image.np_feature_images = np_feature_images
        pickable_brain_image.feature_matrix = brain_image.feature_matrix

        return pickable_brain_image


class PicklableToBrainImageBridge:
    """A :class:`PicklableBrainImage` to :class:`BrainImage` bridge."""

    @staticmethod
    def convert(picklable_brain_image: PicklableBrainImage) -> BrainImage:
        """Converts a :class:`PicklableBrainImage` to :class:`BrainImage`.

        Args:
            picklable_brain_image (PicklableBrainImage): A pickable brain image.

        Returns:
            BrainImage: The brain image.
        """

        images = {}
        for key, np_img in picklable_brain_image.np_images.items():
            images[key] = conversion.NumpySimpleITKImageBridge.convert(np_img, picklable_brain_image.image_properties)

        feature_images = {}
        for key, np_feat_img in picklable_brain_image.np_feature_images.items():
            feature_images[key] = conversion.NumpySimpleITKImageBridge.convert(np_feat_img,
                                                                               picklable_brain_image.image_properties)

        brain_image = BrainImage(picklable_brain_image.id_, picklable_brain_image.path, images)
        brain_image.feature_matrix = picklable_brain_image.feature_matrix
        return brain_image
