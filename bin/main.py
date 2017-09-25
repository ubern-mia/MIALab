"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
"""
import argparse
import datetime
import os
import sys
import timeit
import gc

import SimpleITK as sitk
import numpy as np
from tensorflow.python.platform import app

import mialab.classifier.decision_forest as df
import mialab.data.conversion as conversion
import mialab.data.structure as structure
import mialab.data.loading as load
import mialab.utilities.file_access_utilities as futil
import mialab.utilities.pipeline_utilities as putil

FLAGS = None  # the program flags
IMAGE_KEYS = [structure.BrainImageTypes.T1, structure.BrainImageTypes.T2, structure.BrainImageTypes.GroundTruth]  # the list of images we will load
TRAIN_BATCH_SIZE = 2
TEST_BATCH_SIZE = 2


def main(_):
    """Brain tissue segmentation using decision forests.

    The main routine executes the medical image analysis pipeline:
        - Image loading
        - Registration
        - Pre-processing
        - Feature extraction
        - Decision forest classifier model building
        - Segmentation using the decision forest classifier model on unseen images
        - Post-processing of the segmentation
        - Evaluation of the segmentation
    """

    # load atlas images
    putil.load_atlas_images(FLAGS.data_atlas_dir)

    print('-' * 5, 'Training...')

    # generate a model directory (use datetime to ensure that the directory is empty)
    # we need an empty directory because TensorFlow will continue training an existing model if it is not empty
    t = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
    model_dir = os.path.join(FLAGS.model_dir, t)
    os.makedirs(model_dir, exist_ok=True)

    # crawl the training image directories
    crawler = load.FileSystemDataCrawler(FLAGS.data_train_dir,
                                         IMAGE_KEYS,
                                         futil.BrainImageFilePathGenerator(),
                                         futil.DataDirectoryFilter())
    data_items = list(crawler.data.items())

    pre_process_params = {'zscore_pre': True,
                          'coordinates_feature': True,
                          'intensity_feature': True,
                          'gradient_intensity_feature': True}

    # initialize decision forest parameters
    df_params = df.DecisionForestParameters()
    df_params.num_classes = 4
    df_params.num_trees = 20
    df_params.max_nodes = 1000
    df_params.model_dir = model_dir
    forest = None

    for i in range(0, len(crawler.data), TRAIN_BATCH_SIZE):
        batch_data = dict(data_items[i: i+TRAIN_BATCH_SIZE])  # slicing manages out of range; no need to worry
        # load images for training and pre-process
        images = putil.pre_process_batch(batch_data, pre_process_params, multi_process=True)

        # generate feature matrix and label vector
        data_train = np.concatenate([img.feature_matrix[0] for img in images])
        labels_train = np.concatenate([img.feature_matrix[1] for img in images])

        if forest is None:
            df_params.num_features = images[0].feature_matrix[0].shape[1]
            print(df_params)
            forest = df.DecisionForest(df_params)

        start_time = timeit.default_timer()
        forest.train(data_train, labels_train)
        print(' Time elapsed:', timeit.default_timer() - start_time, 's')

        # todo(alain): verify
        del images, labels_train, data_train
        gc.collect()

    print('-' * 5, 'Testing...')
    result_dir = os.path.join(FLAGS.result_dir, t)
    os.makedirs(result_dir, exist_ok=True)

    # initialize evaluator
    evaluator = putil.init_evaluator(result_dir)

    # crawl the training image directories
    crawler = load.FileSystemDataCrawler(FLAGS.data_test_dir,
                                         IMAGE_KEYS,
                                         futil.BrainImageFilePathGenerator(),
                                         futil.DataDirectoryFilter())
    # load images for testing and pre-process
    pre_process_params['training'] = False
    images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=True)

    for img in images_test:
        data_test = img.feature_matrix[0]
        labels_test = img.feature_matrix[1]

        print('-' * 10, 'Testing', img.id_)
        start_time = timeit.default_timer()
        probabilities, predictions = forest.predict(data_test)
        print(' Time elapsed:', timeit.default_timer() - start_time, 's')

        # print feature importances if calculated
        if df_params.report_feature_importances:
            results = forest.evaluate(data_test, labels_test)
            for key in sorted(results):
                print('%s: %s' % (key, results[key]))

        # convert prediction and probabilities back to SimpleITK images
        image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.uint8),
                                                                        img.image_properties)

        image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)

        # evaluate segmentation without post-processing
        evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

        # post-process segmentation and evaluate with post-processing
        image_post_processed = putil.post_process(img, image_prediction, image_probabilities, crf_post=True)
        # image_post_processed = putil.post_process_batch([img], [image_prediction], [image_probabilities])[0]
        evaluator.evaluate(image_post_processed, img.images[structure.BrainImageTypes.GroundTruth], img.id_ + '-PP')

        # save results
        sitk.WriteImage(image_prediction, os.path.join(result_dir, img.id_ + '_SEG.mha'), True)
        sitk.WriteImage(image_post_processed, os.path.join(result_dir, img.id_ + '_SEG-PP.mha'), True)


if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')
    parser.add_argument(
        '--model_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, 'mia-model')),
        help='Base directory for output models.'
    )

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, 'mia-result')),
        help='Directory for results.'
    )

    parser.add_argument(
        '--data_atlas_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/atlas')),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_train_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/train/')),
        help='Directory with training data.'
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/test/')),
        help='Directory with testing data.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)
