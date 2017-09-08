"""A medical image analysis pipeline.

The pipeline is used for brain compartment segmentation using a decision forest classifier.
"""
import argparse
import datetime
import multiprocessing
import os
import sys
import timeit

import numpy as np
import SimpleITK as sitk
from tensorflow.python.platform import app

import mialab.classifier.decision_forest as df
import mialab.data.loading as load
import mialab.data.structure as structure
import mialab.evaluation.evaluator as eval
import mialab.evaluation.metric as metric
import mialab.evaluation.validation as valid
import mialab.filtering.filter as fltr
import mialab.filtering.preprocessing as fltr_prep
import mialab.filtering.postprocessing as fltr_postp
import mialab.filtering.registration as fltr_reg

FLAGS = None  # the program flags
IMAGE_KEYS = [structure.BrainImageTypes.T1, structure.BrainImageTypes.T2, structure.BrainImageTypes.GroundTruth]  # the list of images we will load
T1_ATLAS_IMG = sitk.Image()
T2_ATLAS_IMG = sitk.Image()


def execute_pipeline(id_: str, paths: dict):
    """todo(fabianbalsiger): comment

    Args:
        id_ (str): An image identifier.
        paths (str):

    Returns:

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
    pipeline.set_param(fltr_reg.MultiModalRegistrationParams(T1_ATLAS_IMG), 1)

    img.images[structure.BrainImageTypes.T1] = pipeline.execute(img.images[structure.BrainImageTypes.T1])
    pipeline.set_param(fltr_reg.MultiModalRegistrationParams(T2_ATLAS_IMG), 1)
    img.images[structure.BrainImageTypes.T2] = pipeline.execute(img.images[structure.BrainImageTypes.T2])

    return img


def execute_postprocessing(id_: str, img: BrainImage, segmentation, probability) -> sitk.Image:
    
    print('-' * 5, 'Post-process', id_)
    
    # construct pipeline
    pipeline = fltr.FilterPipeline()
    pipeline.add_filter(fltr_postp.DenseCRF())
    pipeline.set_param(fltr_postp.DenseCRFParams(img.images[structure.BrainImageTypes.T1],
                                                 img.images[structure.BrainImageTypes.T2],
                                                 probability), 0)
    
    return pipeline.execute(segmentation)

    
def init_evaluator(directory: str) -> eval.Evaluator:
    """Initializes an evaluator.

    Args:
        directory (str): The directory for the results file.

    Returns:
        eval.Evaluator: An evaluator.
    """
    os.makedirs(FLAGS.result_dir, exist_ok=True)  # generate result directory, if it does not exists

    evaluator = eval.Evaluator(eval.ConsoleEvaluatorWriter(5))
    evaluator.add_writer(eval.CSVEvaluatorWriter(os.path.join(directory, 'results.csv')))
    evaluator.add_label(1, "WhiteMatter")
    evaluator.metrics = [metric.DiceCoefficient()]
    return evaluator


class BrainImageFilePathGenerator(load.FilePathGenerator):

    @staticmethod
    def get_full_file_path(id_: str, root_dir: str, file_key, file_extension: str) -> str:
        if file_key == structure.BrainImageTypes.T1:
            file_name = 'T1native_biasfieldcorr_noskull'
        elif file_key == structure.BrainImageTypes.T2:
            file_name = 'T2native_biasfieldcorr_noskull'
        elif file_key == structure.BrainImageTypes.GroundTruth:
            file_name = 'labels_native'
        elif file_key == structure.BrainImageTypes.BrainMask:
            file_name = 'Brainmasknative'
        else:
            raise ValueError('Unknown key')

        return os.path.join(root_dir, file_name + file_extension)


def load_atlas(dir_path: str):
    T1_ATLAS_IMG = sitk.ReadImage(os.path.join(dir_path, 'sometext.nii.gz'))
    T2_ATLAS_IMG = sitk.ReadImage(os.path.join(dir_path, 'sometext.nii.gz'))

def main(_):
    """Brain tissue segmentation using decision forests.

    The main routine executes the medical image analysis pipeline:
        - Image loading
        - ...
    """

    load_atlas(os.path.join(FLAGS.data_dir, 'atlas'))

    # crawl the image directories
    crawler = load.FileSystemDataCrawler(FLAGS.data_dir, IMAGE_KEYS, BrainImageFilePathGenerator())

    # create a pool to parallelize the image processing
    with multiprocessing.Pool() as p:
        images = p.starmap(execute_pipeline, crawler.data.items())

    # initialize decision forest parameters
    params = df.DecisionForestParameters()
    evaluator = init_evaluator()

    # leave-one-out cross-validation
    for train_idx, test_idx in valid.LeaveOneOutCrossValidator(len(images)):
        # generate training data
        train_data = np.concatenate([images[idx].training_data for idx in train_idx])
        # train_data = train_data[:, feat_selector]
        train_labels = np.concatenate([images[idx].training_labels for idx in train_idx])

        print('-' * 5, 'Training...')
        # generate a model directory (use datetime to ensure that the directory is empty)
        # we need an empty directory because TensorFlow will continue training an existing model if it is not empty
        model_dir = os.path.join(FLAGS.model_dir, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        os.makedirs(model_dir, exist_ok=True)
        params.model_dir = model_dir

        forest = df.DecisionForest(params)
        start_time = timeit.default_timer()
        forest.train(train_data, train_labels)
        print(' Time elapsed:', timeit.default_timer() - start_time, 's')

        print('-' * 5, 'Testing...')


if __name__ == "__main__":
    """The program's entry point."""

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./mia-model',
        help='Base directory for output models.'
    )

    parser.add_argument(
        '--result_dir',
        type=str,
        default='./mia-result',
        help='Directory for results.'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='./../data/',
        help='Directory with data.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)
