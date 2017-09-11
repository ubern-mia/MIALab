"""This module contains utility classes and functions."""

from enum import Enum
import os
from typing import List
import timeit

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
        directory (str): The data directory.
    """

    global atlas_t1
    global atlas_t2
    atlas_t1 = sitk.ReadImage(os.path.join(directory,
                                           'Atlas_mni_icbm152_nlin_sym_09a',
                                           'mni_icbm152_t1_tal_nlin_sym_09a_mask.nii.gz'))
    atlas_t2 = sitk.ReadImage(os.path.join(directory,
                                           'Atlas_mni_icbm152_nlin_sym_09a',
                                           'mni_icbm152_t2_tal_nlin_sym_09a.nii.gz'))


class BrainImageFilePathGenerator(load.FilePathGenerator):

    @staticmethod
    def get_full_file_path(id_: str, root_dir: str, file_key, file_extension: str) -> str:
        if file_key == structure.BrainImageTypes.T1:
            file_name = 'T1native_biasfieldcorr'
        elif file_key == structure.BrainImageTypes.T2:
            file_name = 'T2native_biasfieldcorr'
        elif file_key == structure.BrainImageTypes.GroundTruth:
            file_name = 'labels_native'
        elif file_key == structure.BrainImageTypes.BrainMask:
            file_name = 'Brainmasknative'
        else:
            raise ValueError('Unknown key')

        return os.path.join(root_dir, file_name + file_extension)


class DataDirectoryFilter(load.DirectoryFilter):

    @staticmethod
    def filter_directories(dirs: List[str]) -> List[str]:
        return [dir for dir in dirs if not dir.lower().__contains__('atlas')]


class FeatureImageTypes(Enum):
    """Represents the feature image types."""
    ATLAS_COORD = 1
    T1_TEXTURE_1 = 2
    T1_GRADIENT_1 = 3
    T2_TEXTURE_1 = 4
    T2_GRADIENT_1 = 5


class FeatureExtractor:
    """Represents a feature extractor."""
    
    def __init__(self, img: structure.BrainImage):
        self.img = img

    def execute(self):

        # normalized_coordinates = fltr_feat.NormalizedAtlasCoordinates()
        # self.img.feature_images[FeatureImageTypes.ATLAS_COORD] = \
        #     normalized_coordinates.execute(self.img.images[structure.BrainImageTypes.T1])

        # initialize first-order texture feature extractor
        first_order_texture = fltr_feat.NeighborhoodFeatureExtractor(kernel=(3,3,3))

        # compute gradient magnitude images
        t1_gradient_magnitude = sitk.GradientMagnitude(self.img.images[structure.BrainImageTypes.T1])
        t2_gradient_magnitude = sitk.GradientMagnitude(self.img.images[structure.BrainImageTypes.T2])

        self.img.feature_images[FeatureImageTypes.T1_TEXTURE_1] = \
            first_order_texture.execute(self.img.images[structure.BrainImageTypes.T1])

        self.img.feature_images[FeatureImageTypes.T1_GRADIENT_1] = \
            first_order_texture.execute(t1_gradient_magnitude)

        self.img.feature_images[FeatureImageTypes.T2_TEXTURE_1] = \
            first_order_texture.execute(self.img.images[structure.BrainImageTypes.T2])

        self.img.feature_images[FeatureImageTypes.T2_GRADIENT_1] = \
            first_order_texture.execute(t2_gradient_magnitude)

        return self.img


def process(id_: str, paths: dict) -> structure.BrainImage:
    """todo(fabianbalsiger): comment
    Args:
        id_ (str): An image identifier.
        paths (dict):

    Returns:
        (structure.BrainImage):
    """

    print('-' * 5, 'Processing', id_)

    # load image
    path = paths.pop(id_, '')
    img = {img_key: sitk.ReadImage(path) for img_key, path in paths.items()}
    img = structure.BrainImage(id_, path, img)

    # construct pipeline
    pipeline = fltr.FilterPipeline()
    pipeline.add_filter(fltr_prep.NormalizeZScore())
    pipeline.add_filter(fltr_reg.MultiModalRegistration())
    pipeline.set_param(fltr_reg.MultiModalRegistrationParams(atlas_t1), 1)

    # execute pipeline on T1 image
    img.images[structure.BrainImageTypes.T1] = pipeline.execute(img.images[structure.BrainImageTypes.T1])

    # change fixed image for T2 pipeline
    pipeline.set_param(fltr_reg.MultiModalRegistrationParams(atlas_t2), 1)

    # execute pipeline on T2 image
    img.images[structure.BrainImageTypes.T2] = pipeline.execute(img.images[structure.BrainImageTypes.T2])

    # extract the features
    feature_extractor = FeatureExtractor(img)
    img = feature_extractor.execute()

    return img


def some_test(values):
    for id_, paths in values.items():
        start_time = timeit.default_timer()

        path = paths.pop(id_, '')
        img = {img_key: sitk.ReadImage(path) for img_key, path in paths.items()}
        img = structure.BrainImage(id_, path, img)

        # construct pipeline
        pipeline = fltr.FilterPipeline()
        pipeline.add_filter(fltr_prep.NormalizeZScore())
        pipeline.add_filter(fltr_reg.MultiModalRegistration())
        pipeline.set_param(fltr_reg.MultiModalRegistrationParams(atlas_t1), 1)

        # execute pipeline on T1 image
        img.images[structure.BrainImageTypes.T1] = pipeline.execute(img.images[structure.BrainImageTypes.T1])

        # change fixed image for T2 pipeline
        pipeline.set_param(fltr_reg.MultiModalRegistrationParams(atlas_t2), 1)

        # execute pipeline on T2 image
        img.images[structure.BrainImageTypes.T2] = pipeline.execute(img.images[structure.BrainImageTypes.T2])

        # extract the features
        feature_extractor = FeatureExtractor(img)
        img = feature_extractor.execute()

        sitk.WriteImage(img.images[structure.BrainImageTypes.T1], os.path.join(img.path, 'T1reg.nii.gz'))
        sitk.WriteImage(img.images[structure.BrainImageTypes.T2], os.path.join(img.path, 'T2reg.nii.gz'))

        print(' Time elapsed:', timeit.default_timer() - start_time, 's')


def postprocess(id_: str, img: structure.BrainImage, segmentation, probability) -> sitk.Image:
    
    print('-' * 5, 'Post-process', id_)
    
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
