"""A medical image analysis pipeline.

The pipeline is used for brain compartment segmentation using a decision forest classifier.
"""
import argparse
import datetime
from functools import partial
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
import mialab.evaluation.validation as valid
import mialab.utilities as util

FLAGS = None  # the program flags
IMAGE_KEYS = [structure.BrainImageTypes.T1, structure.BrainImageTypes.T2, structure.BrainImageTypes.GroundTruth]  # the list of images we will load


def main(_):
    """Brain tissue segmentation using decision forests.

    The main routine executes the medical image analysis pipeline:
        - Image loading
        - ...
    """

    util.load_atlas_images(FLAGS.data_dir)

    # crawl the image directories
    crawler = load.FileSystemDataCrawler(FLAGS.data_dir,
                                         IMAGE_KEYS,
                                         util.BrainImageFilePathGenerator(),
                                         util.DataDirectoryFilter())

    util.some_test(crawler.data)

    # create a pool to parallelize the image processing
    with multiprocessing.Pool() as p:
        # images will be of type list of BrainImage
        images = p.starmap(util.process, crawler.data.items())

    # initialize decision forest parameters
    params = df.DecisionForestParameters()

    # initialize evaluator
    evaluator = util.init_evaluator(FLAGS.result_dir)

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
