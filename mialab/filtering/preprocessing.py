"""The pre-processing module contains classes for image pre-processing.

Image pre-processing aims to improve the image quality (image intensities) for subsequent pipeline steps.
"""
import warnings

import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk


class ImageNormalization(pymia_fltr.IFilter):
    """Represents a normalization filter."""

    def __init__(self):
        """Initializes a new instance of the ImageNormalization class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: pymia_fltr.IFilterParams = None) -> sitk.Image:
        """Executes a normalization on an image.

        Args:
            image (sitk.Image): The image.
            params (IFilterParams): The parameters (unused).

        Returns:
            sitk.Image: The normalized image.
        """

        img_arr = sitk.GetArrayFromImage(image)

        # todo: normalize the image in some way using numpy
        warnings.warn('No normalization implemented. Returning unprocessed image.')

        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageNormalization:\n' \
            .format(self=self)


class ImageRegistration(pymia_fltr.IFilter):
    """Represents a registration filter."""

    def __init__(self):
        """Initializes a new instance of the ImageRegistration class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: pymia_fltr.IFilterParams = None) -> sitk.Image:
        """Registers an image.

        Args:
            image (sitk.Image): The image.
            params (IFilterParams): The registration parameters.

        Returns:
            sitk.Image: The registered image.
        """

        # todo: replace this filter by a registration. Luckily, you do not need to implement the registration yourself,
        # just use pymia.filtering.registration.MultiModalRegistration. Think about the type of registration, i.e.
        # do you want to register to an atlas or inter-subject?
        warnings.warn('No registration implemented. See pymia.filtering.registration.MultiModalRegistration.')

        return image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageRegistration:\n' \
            .format(self=self)
