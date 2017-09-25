from typing import Tuple

import numpy as np
import SimpleITK as sitk
from pathos import multiprocessing as pmp

import mialab.data.structure as structure
import mialab.data.conversion as conversion


class PicklableBrainImage:
    """Represents a brain image that can be pickled."""

    def __init__(self, id_: str, path: str, np_images: dict, image_properties: conversion.ImageProperties):
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
    def convert(brain_image: structure.BrainImage)-> PicklableBrainImage:
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
    def convert(picklable_brain_image: PicklableBrainImage) -> structure.BrainImage:
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

        brain_image = structure.BrainImage(picklable_brain_image.id_, picklable_brain_image.path, images)
        brain_image.feature_matrix = picklable_brain_image.feature_matrix
        return brain_image


class DefaultPickleHelper:

    def make_params_picklable(self, params):
        """Default function called to ensure that all parameters can be pickled before transferred to the new process.
        To be overwritten if non-picklable parameters are contained in `params`.

        Args:
            params (tuple): Parameters to be rendered picklable.

        Returns:
            tuple: The modified parameters.
        """
        return params

    def recover_params(self, params):
        """Default function called to recover (from the pickle state) the original parameters in another process.
        To be overwritten if non-picklable parameters are contained in `params`.

        Args:
            params (tuple): Parameters to be recovered.

        Returns:
            tuple: The recovered parameters.
        """
        return params

    def make_return_value_picklable(self, ret_val):
        """ Default function called to ensure that all return values `ret_val` can be pickled before transferring
        back to the original process.
        To be overwritten if non-picklable objects are contained in `ret_val`.

        Args:
            ret_val: Return values of the function executed in another process.

        Returns:
            The modified return values.
        """
        return ret_val

    def recover_return_value(self, ret_val):
        """ Default function called to ensure that all return values `ret_val` can be pickled before transferring
        back to the original process.
        To be overwritten if non-picklable objects are contained in `ret_val`.

        Args:
            ret_val: Return values of the function executed in another process.

        Returns:
            The modified return values.
        """
        return ret_val


class PreProcessingPickleHelper(DefaultPickleHelper):

    def make_return_value_picklable(self, ret_val: structure.BrainImage) -> PicklableBrainImage:
        """Ensures that all pre-processing return values `ret_val` can be pickled before transferring back to
        the original process.

        Args:
            ret_val: Return values of the pre-processing function executed in another process.

        Returns:
            The modified pre-processing return values.
        """

        return BrainImageToPicklableBridge.convert(ret_val)

    def recover_return_value(self, ret_val: PicklableBrainImage) -> structure.BrainImage:
        """Recovers (from the pickle state) the original pre-processing return values.

        Args:
            ret_val: Pre-processing return values to be recovered.

        Returns:
            The recovered pre-processing return values.
        """
        return PicklableToBrainImageBridge.convert(ret_val)


class PostProcessingPickleHelper(DefaultPickleHelper):

    def make_params_picklable(self, params: Tuple[structure.BrainImage, sitk.Image, sitk.Image]):
        """Ensures that all post-processing parameters can be pickled before transferred to the new process.

                Args:
                    params (tuple): Post-processing parameters to be rendered picklable.

                Returns:
                    tuple: The modified post-processing parameters.
                """
        brain_img, segmentation, probability = params
        picklable_brain_image = BrainImageToPicklableBridge.convert(brain_img)
        np_segmentation, _ = conversion.SimpleITKNumpyImageBridge.convert(segmentation)
        np_probability, _ = conversion.SimpleITKNumpyImageBridge.convert(probability)
        return picklable_brain_image, np_segmentation, np_probability

    def recover_params(self, params: Tuple[PicklableBrainImage, np.ndarray, np.ndarray]):
        """Recovers (from the pickle state) the original post-processing parameters in another process.

                Args:
                    params (tuple): Post-processing parameters to be recovered.

                Returns:
                    tuple: The recovered post-processing parameters.

                """
        picklable_img, np_segmentation, np_probability = params
        img = PicklableToBrainImageBridge.convert(picklable_img)
        segmentation = conversion.NumpySimpleITKImageBridge.convert(np_segmentation, picklable_img.image_properties)
        probability = conversion.NumpySimpleITKImageBridge.convert(np_probability, picklable_img.image_properties)
        return img, segmentation, probability

    def make_return_value_picklable(self, ret_val: sitk.Image) -> Tuple[np.ndarray, conversion.ImageProperties]:
        """Ensures that all post-processing return values `ret_val` can be pickled before transferring back to
                the original process.

                Args:
                    ret_val: Return values of the post-processing function executed in another process.

                Returns:
                    The modified post-processing return values.
                """
        np_img, image_properties = conversion.SimpleITKNumpyImageBridge.convert(ret_val)
        return np_img, image_properties

    def recover_return_value(self, ret_val: Tuple[np.ndarray, conversion.ImageProperties]) -> sitk.Image:
        """Recovers (from the pickle state) the original post-processing return values.

                Args:
                    ret_val: Post-processing return values to be recovered.

                Returns:
                    The recovered post-processing return values.
                """
        np_img, image_properties = ret_val
        return conversion.NumpySimpleITKImageBridge.convert(np_img, image_properties)


class MultiProcessor:
    """Class managing multiprocessing"""

    @staticmethod
    def run(fn: callable, param_list: iter, pickle_helper_cls: type=DefaultPickleHelper):
        """ Executes the function `fn` in parallel (different processes) for each parameter in the parameter list.

        Args:
            fn (callable): Function to be executed in another process.
            param_list (List[tuple]): List containing the parameters for each `fn` call.
            pickle_helper_cls (class:

        Returns:
            list: A list of all return values of the `fn` calls
        """
        helper = pickle_helper_cls()
        param_list = (helper.make_params_picklable(params) for params in param_list)

        with pmp.Pool() as p:
            ret_vals = p.starmap(MultiProcessor._wrap_fn(fn, pickle_helper_cls), param_list)
        ret_vals = [helper.recover_return_value(ret_val) for ret_val in ret_vals]
        return ret_vals

    @staticmethod
    def _wrap_fn(fn, pickle_helper_cls):
        def wrapped_fn(*params):
            # create instance due to possible race condition
            helper = pickle_helper_cls()
            params = helper.recover_params(params)
            ret_val = fn(*params)
            ret_val = helper.make_return_value_picklable(ret_val)
            return ret_val

        return wrapped_fn
