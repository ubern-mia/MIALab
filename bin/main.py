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
from tensorflow.python.platform import app

import mialab.data.structure as ds
import mialab.classifier.decision_forest as df
import mialab.data.loading as load
import mialab.evaluation.evaluator as eval
import mialab.evaluation.metric as metric
import mialab.evaluation.validation as valid

FLAGS = None  # the program flags


def load_images(img_id: str, path: str, img):
    """todo(fabianbalsiger): comment

    Args:
        img_id (str):
        path (str):

    Returns:

    """
    print(img_id)
    return img


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
        if file_key == ds.BrainImageTypes.T1:
            file_name = 'T1native_biasfieldcorr_noskull'
        elif file_key == ds.BrainImageTypes.T2:
            file_name = 'T2native_biasfieldcorr_noskull'
        elif file_key == ds.BrainImageTypes.GroundTruth:
            file_name = 'labels_native'
        elif file_key == ds.BrainImageTypes.BrainMask:
            file_name = 'Brainmasknative'
        else:
            raise ValueError('Unknown key')

        return os.path.join(root_dir, file_name + file_extension)


def main(_):
    """Brain tissue segmentation using decision forests.

    The main routine executes the medical image analysis pipeline:
        - Image loading
        - ...
    """
    # the list of images we will load
    image_list = [ds.BrainImageTypes.T1, ds.BrainImageTypes.T2, ds.BrainImageTypes.GroundTruth]

    # load the images
    crawler = load.FileSystemCrawler(FLAGS.data_dir)
    image_directories = crawler.image_path_dict
    image_loader = load.SITKImageLoader(image_directories, image_list, BrainImageFilePathGenerator())

    with multiprocessing.Pool() as p:
        images = p.starmap(load_images, list(image_loader))

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
