"""The pre-processing module contains classes for image pre-processing.

Image pre-processing aims to improve the image quality (image intensities) for subsequent pipeline steps.
"""
import warnings

import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk
import numpy as np


class ImageNormalization(pymia_fltr.Filter):
    """Represents a normalization filter."""

    def __init__(self):
        """Initializes a new instance of the ImageNormalization class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Executes a normalization on an image.

        Args:
            image (sitk.Image): The image.
            params (FilterParams): The parameters (unused).

        Returns:
            sitk.Image: The normalized image.
        """

        img_arr = sitk.GetArrayFromImage(image)

        # Perform image normalization using NumPy
        min_val = np.min(img_arr)
        max_val = np.max(img_arr)

        normalized_img_arr = (img_arr - min_val) / (max_val - min_val)

        # Create a new SimpleITK image from the normalized NumPy array
        img_out = sitk.GetImageFromArray(normalized_img_arr)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageNormalization:\n' \
            .format(self=self)


class SkullStrippingParameters(pymia_fltr.FilterParams):
    """Skull-stripping parameters."""

    def __init__(self, img_mask: sitk.Image):
        """Initializes a new instance of the SkullStrippingParameters

        Args:
            img_mask (sitk.Image): The brain mask image.
        """
        self.img_mask = img_mask


class SkullStripping(pymia_fltr.Filter):
    """Represents a skull-stripping filter."""

    def __init__(self):
        """Initializes a new instance of the SkullStripping class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: SkullStrippingParameters = None) -> sitk.Image:
        """Executes a skull stripping on an image.

        Args:
            image (sitk.Image): The image.
            params (SkullStrippingParameters): The parameters with the brain mask.

        Returns:
            sitk.Image: The skull-stripped image.
        """
        if params is None or params.img_mask is None:
            raise ValueError("SkullStrippingParameters with img_mask is required for skull stripping.")

        mask = params.img_mask  # The brain mask

        # Apply the brain mask to remove non-brain regions
        stripped_image = image * mask

        return stripped_image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'SkullStripping:\n' \
            .format(self=self)


class ImageRegistrationParameters(pymia_fltr.FilterParams):
    """Image registration parameters."""

    def __init__(self, atlas: sitk.Image, transformation: sitk.Transform, is_ground_truth: bool = False):
        """Initializes a new instance of the ImageRegistrationParameters

        Args:
            atlas (sitk.Image): The atlas image.
            transformation (sitk.Transform): The transformation for registration.
            is_ground_truth (bool): Indicates weather the registration is performed on the ground truth or not.
        """
        self.atlas = atlas
        self.transformation = transformation
        self.is_ground_truth = is_ground_truth


class ImageRegistration(pymia_fltr.Filter):
    """Represents a registration filter."""

    def __init__(self):
        """Initializes a new instance of the ImageRegistration class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: ImageRegistrationParameters = None) -> sitk.Image:
        """Registers an image.

        Args:
            image (sitk.Image): The image.
            params (ImageRegistrationParameters): The registration parameters.

        Returns:
            sitk.Image: The registered image.
        """

        # toodo: replace this filter by a registration. Registration can be costly, therefore, we provide you the
        # transformation, which you only need to apply to the image!
        # warnings.warn('No registration implemented. Returning unregistered image') WARNING OUTDATED

        atlas = params.atlas
        transform = params.transformation
        is_ground_truth = params.is_ground_truth  # the ground truth will be handled slightly different

        # Create a registration object
        registration = sitk.ImageRegistrationMethod()

        # Choose a similarity metric (e.g., mutual information)
        registration.SetMetricAsMattesMutualInformation()

        # Choose an optimizer (e.g., gradient descent)
        registration.SetOptimizerAsGradientDescent(learningRate=0.1, numberOfIterations=100,
                                                   estimateLearningRate=registration.EachIteration)

        # Create the transformation (e.g., affine)
        initial_transform = sitk.AffineTransform(3)

        # Set the initial transformation parameters (if needed)
        initial_transform.SetCenter([0, 0, 0])  # Adjust as needed

        registration.SetInitialTransform(initial_transform, inPlace=False)

        # Execute the registration
        final_transform = registration.Execute(sitk.Cast(image, sitk.sitkFloat32), sitk.Cast(atlas, sitk.sitkFloat32))

        # Apply the final transformation to the image
        registered_image = sitk.Resample(image, atlas, final_transform, sitk.sitkLinear, 0.0, sitk.sitkUInt16)


        # note: if you are interested in registration, and want to test it, have a look at
        # pymia.filtering.registration.MultiModalRegistration. Think about the type of registration, i.e.
        # do you want to register to an atlas or inter-subject? Or just ask us, we can guide you ;-)

        return registered_image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageRegistration:\n' \
            .format(self=self)
