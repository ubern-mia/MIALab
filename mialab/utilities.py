"""This module contains utility classes and functions."""
import concurrent.futures
from enum import Enum
import os
from typing import List

import numpy as np
import SimpleITK as sitk

import mialab.data.loading as load
import mialab.data.structure as structure
import mialab.evaluation.evaluator as eval
import mialab.evaluation.metric as metric
import mialab.filtering.feature_extraction as fltr_feat
import mialab.filtering.filter as fltr
import mialab.filtering.postprocessing as fltr_postp
import mialab.filtering.preprocessing as fltr_prep
import mialab.filtering.registration as fltr_reg


atlas_t1 = sitk.Image()
atlas_t2 = sitk.Image()


def load_atlas_images(directory: str):
    """Loads the T1 and T2 atlas images.

    Args:
        directory (str): The atlas data directory.
    """

    global atlas_t1
    global atlas_t2
    atlas_t1 = sitk.ReadImage(os.path.join(directory, 'mni_icbm152_t1_tal_nlin_sym_09a_mask.nii.gz'))
    atlas_t2 = sitk.ReadImage(os.path.join(directory, 'mni_icbm152_t2_tal_nlin_sym_09a.nii.gz'))


class BrainImageFilePathGenerator(load.FilePathGenerator):
    """Represents a brain image file path generator.

    The generator is used to convert a human readable image identifier to an image file path,
    which allows to load the image.
    """

    def __init__(self):
        """Initializes a new instance of the BrainImageFilePathGenerator class."""
        pass

    @staticmethod
    def get_full_file_path(id_: str, root_dir: str, file_key, file_extension: str) -> str:
        """Gets the full file path for an image.

        Args:
            id_ (str): The image identification.
            root_dir (str): The image' root directory.
            file_key (object): A human readable identifier used to identify the image.
            file_extension (str): The image' file extension.

        Returns:
            str: The images' full file path.
        """

        # the commented file_names are for the registration group

        if file_key == structure.BrainImageTypes.T1:
            # file_name = 'T1native'
            file_name = 'T1mni_biasfieldcorr_noskull'
        elif file_key == structure.BrainImageTypes.T2:
            # file_name = 'T2native'
            file_name = 'T2mni_biasfieldcorr_noskull'
        elif file_key == structure.BrainImageTypes.GroundTruth:
            # file_name = 'labels_native'
            file_name = 'labels_mniatlas'
        elif file_key == structure.BrainImageTypes.BrainMask:
            # file_name = 'Brainmasknative'
            file_name = 'Brainmaskmni'
        else:
            raise ValueError('Unknown key')

        return os.path.join(root_dir, file_name + file_extension)


class DataDirectoryFilter(load.DirectoryFilter):
    """Represents a data directory filter.

    The filter is used to
    """

    def __init__(self):
        """Initializes a new instance of the DataDirectoryFilter class."""
        pass

    @staticmethod
    def filter_directories(dirs: List[str]) -> List[str]:
        """Filters a list of directories.

        Args:
            dirs (List[str]): A list of directories.

        Returns:
            List[str]: The filtered list of directories.
        """

        # currently, we do not filter the directories. but you could filter the directory list like this:
        # return [dir for dir in dirs if not dir.lower().__contains__('atlas')]
        return dirs


class FeatureImageTypes(Enum):
    """Represents the feature image types."""

    ATLAS_COORD = 1
    T1_TEXTURE_1 = 2
    T1_GRADIENT_1 = 3
    T2_TEXTURE_1 = 4
    T2_GRADIENT_1 = 5


class FeatureExtractor:
    """Represents a feature extractor."""
    
    def __init__(self, img: structure.BrainImage, training: bool=True):
        """Initializes a new instance of the FeatureExtractor class.

        Args:
            img (structure.BrainImage): The image to extract features from.
            training (bool): Determines whether to extract the features for training or testing.
        """
        self.img = img
        self.training = training

    def execute(self) -> structure.BrainImage:
        """Extracts features from an image.

        Returns:
            structure.BrainImage: The image with extracted features.
        """
        atlas_coordinates = fltr_feat.AtlasCoordinates()
        self.img.feature_images[FeatureImageTypes.ATLAS_COORD] = \
            atlas_coordinates.execute(self.img.images[structure.BrainImageTypes.T1])

        # initialize first-order texture feature extractor
        first_order_texture = fltr_feat.NeighborhoodFeatureExtractor(kernel=(3,3,3))

        # compute gradient magnitude images
        t1_gradient_magnitude = sitk.GradientMagnitude(self.img.images[structure.BrainImageTypes.T1])
        t2_gradient_magnitude = sitk.GradientMagnitude(self.img.images[structure.BrainImageTypes.T2])

        # self.img.feature_images[FeatureImageTypes.T1_TEXTURE_1] = \
        #     first_order_texture.execute(self.img.images[structure.BrainImageTypes.T1])
        #
        # self.img.feature_images[FeatureImageTypes.T1_GRADIENT_1] = \
        #     first_order_texture.execute(t1_gradient_magnitude)
        #
        # self.img.feature_images[FeatureImageTypes.T2_TEXTURE_1] = \
        #     first_order_texture.execute(self.img.images[structure.BrainImageTypes.T2])
        #
        # self.img.feature_images[FeatureImageTypes.T2_GRADIENT_1] = \
        #     first_order_texture.execute(t2_gradient_magnitude)

        self.img.feature_images[FeatureImageTypes.T1_TEXTURE_1] = self.img.images[structure.BrainImageTypes.T1]
        self.img.feature_images[FeatureImageTypes.T2_TEXTURE_1] = self.img.images[structure.BrainImageTypes.T2]

        self._generate_feature_matrix()

        return self.img

    def _generate_feature_matrix(self):
        """Generates a feature matrix."""

        mask = None

        if self.training:
            # generate a randomized mask where 1 represents voxels used for training
            # the mask needs to be binary, where the value 1 is considered as a voxel which is to be loaded
            # we have following labels:
            # - 0 (background): circa 18000000 voxels
            # - 1 (white matter): circa 1300000 voxels
            # - 2 (grey matter): circa 1800000 voxels
            # - 3 (ventricles): circa 130000 voxels

            # you can exclude background voxels from the training mask generation
            # mask_background = self.img.images[structure.BrainImageTypes.BrainMask]
            # and use background_mask=mask_background in get_mask()

            mask = fltr_feat.RandomizedTrainingMaskGenerator.get_mask(self.img.images[structure.BrainImageTypes.GroundTruth],
                                                                      [0, 1, 2, 3],
                                                                      [0.0003, 0.004, 0.003, 0.04])

            # convert the mask to a logical array where value 1 is False and value 0 is True
            mask = sitk.GetArrayFromImage(mask)
            mask = np.logical_not(mask)

        # generate features
        data = np.concatenate([self._image_as_numpy_array(image, mask) for id_, image in self.img.feature_images.items()],
                               axis=1)

        # generate labels (note that we assume to have a ground truth even for testing)
        labels = self._image_as_numpy_array(self.img.images[structure.BrainImageTypes.GroundTruth], mask)

        self.img.feature_matrix = (data.astype(np.float32), labels.astype(np.int16))

    @staticmethod
    def _image_as_numpy_array(image: sitk.Image, mask: np.ndarray=None):
        """Gets an image as numpy array where each row is a voxel and each column is a feature.

        Args:
            image (sitk.Image): The image.
            mask (np.ndarray): A mask defining which voxels to return. True is background, False is a masked voxel.

        Returns:
            np.ndarray: An array where each row is a voxel and each column is a feature.
        """

        number_of_components = image.GetNumberOfComponentsPerPixel()  # the number of features for this image
        no_voxels = np.prod(image.GetSize())
        image = sitk.GetArrayFromImage(image)

        if mask is not None:
            no_voxels = np.size(mask) - np.count_nonzero(mask)

            if number_of_components == 1:
                masked_image = np.ma.masked_array(image, mask=mask)
            else:
                # image is a vector image, make a vector mask
                vector_mask = np.expand_dims(mask, axis=3)  # shape is now (z, x, y, 1)
                vector_mask = np.repeat(vector_mask, number_of_components,
                                        axis=3)  # shape is now (z, x, y, number_of_components)
                masked_image = np.ma.masked_array(image, mask=vector_mask)

            image = masked_image[~masked_image.mask]

        return image.reshape((no_voxels, number_of_components))


def process(id_: str, paths: dict, training: bool) -> structure.BrainImage:
    """Loads and processes an image.

    The processing includes:

    - Registration
    - Pre-processing
    - Feature extraction

    Args:
        id_ (str): An image identifier.
        paths (dict): A dict, where the keys are an image identifier of type structure.BrainImageTypes
            and the values are paths to the images.
        training (bool): Determines whether to extract the features for training or testing.

    Returns:
        (structure.BrainImage):
    """

    print('-' * 10, 'Processing', id_)

    # load image
    path = paths.pop(id_, '')  # the value with key id_ is the root directory of the image
    img = {img_key: sitk.ReadImage(path) for img_key, path in paths.items()}
    img = structure.BrainImage(id_, path, img)

    # NOTE: uncomment the code lines to enable the registration

    # construct T1 pipeline
    pipeline_t1 = fltr.FilterPipeline()
    pipeline_t1.add_filter(fltr_prep.NormalizeZScore())
    # pipeline_t1.add_filter(fltr_reg.MultiModalRegistration())
    # pipeline_t1.set_param(fltr_reg.MultiModalRegistrationParams(atlas_t1), 1)

    # execute pipeline on T1 image
    img.images[structure.BrainImageTypes.T1] = pipeline_t1.execute(img.images[structure.BrainImageTypes.T1])

    # get transformation
    # transform = pipeline_t1.filters[1].transform

    # construct T2 pipeline
    pipeline_t2 = fltr.FilterPipeline()
    pipeline_t2.add_filter(fltr_prep.NormalizeZScore())

    # execute pipeline on T2 image
    img.images[structure.BrainImageTypes.T2] = pipeline_t2.execute(img.images[structure.BrainImageTypes.T2])

    # apply transformation of T1 image registration to T2 image
    # image_t2 = img.images[structure.BrainImageTypes.T2]
    # image_t2 = sitk.Resample(image_t2, atlas_t1, transform, sitk.sitkLinear, 0.0,
    #                          image_t2.GetPixelIDValue())
    # img.images[structure.BrainImageTypes.T2] = image_t2

    # apply transformation of T1 image registration to ground truth
    # image_ground_truth = img.images[structure.BrainImageTypes.GroundTruth]
    # image_ground_truth = sitk.Resample(image_ground_truth, atlas_t1, transform, sitk.sitkNearestNeighbor, 0,
    #                                    image_ground_truth.GetPixelIDValue())
    # img.images[structure.BrainImageTypes.GroundTruth] = image_ground_truth

    # extract the features
    feature_extractor = FeatureExtractor(img, training)
    img = feature_extractor.execute()

    return img


def post_process(img: structure.BrainImage, segmentation: sitk.Image, probability: sitk.Image) -> sitk.Image:
    """Post-processes a segmentation.

    Args:
        img (structure.BrainImage): The image.
        segmentation (sitk.Image): The segmentation (label image).
        probability (sitk.Image): The probabilities images (a vector image).

    Returns:
        sitk.Image: The post-processed image.
    """

    print('-' * 10, 'Post-processing', img.id_)
    
    # construct pipeline
    pipeline = fltr.FilterPipeline()
    pipeline.add_filter(fltr_postp.DenseCRF())
    pipeline.set_param(fltr_postp.DenseCRFParams(img.images[structure.BrainImageTypes.T1],
                                                 img.images[structure.BrainImageTypes.T2],
                                                 probability), 0)
    
    return pipeline.execute(segmentation)


def init_evaluator(directory: str, result_file_name: str='results.csv') -> eval.Evaluator:
    """Initializes an evaluator.

    Args:
        directory (str): The directory for the results file.
        result_file_name (str): The result file name (CSV file).

    Returns:
        eval.Evaluator: An evaluator.
    """
    os.makedirs(directory, exist_ok=True)  # generate result directory, if it does not exists

    evaluator = eval.Evaluator(eval.ConsoleEvaluatorWriter(5))
    evaluator.add_writer(eval.CSVEvaluatorWriter(os.path.join(directory, result_file_name)))
    evaluator.add_label(1, "WhiteMatter")
    evaluator.add_label(2, "GreyMatter")
    evaluator.add_label(3, "Ventricles")
    evaluator.metrics = [metric.DiceCoefficient()]
    return evaluator


def process_batch(data_dir: str,
                  image_keys: List[structure.BrainImageTypes],
                  training: bool) -> List[structure.BrainImage]:
    """Loads and processes a batch of images in parallel.

    The processing includes:

    - Registration
    - Pre-processing
    - Feature extraction

    Args:
        data_dir (str): The path to the root directory, which contains subdirectories with the data.
        image_keys (List[structure.BrainImageTypes]): A list of image identifiers.
        training (bool): Determines whether to extract the features for training or testing.

    Returns:
        List[structure.BrainImage]: A list of images.
    """

    # crawl the training image directories
    crawler = load.FileSystemDataCrawler(data_dir,
                                         image_keys,
                                         BrainImageFilePathGenerator(),
                                         DataDirectoryFilter())

    # create a thread pool to parallelize the image processing
    with concurrent.futures.ThreadPoolExecutor(10) as executor:
        futures = [executor.submit(process, id_, paths, training) for id_, paths in crawler.data.items()]
        images = [future.result() for future in concurrent.futures.as_completed(futures)]

    return images
