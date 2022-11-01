import datetime
import os
import sys
import timeit

import pymia.data.conversion as conversion

try:
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
except ImportError:
    # Append the MIALab root directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil


class ClassificationController():

    def __init__(self, classifiers: list, X_train, X_target, Y_target):
        self.classifiers = classifiers
        self.X_train = X_train
        self.X_target = X_target
        self.Y_target = Y_target

        for clf in self.classifiers:
            print(f'Classifier: {clf}')



    def train(self):
        for clf in self.classifiers:
            print('-' * 5, f'[{clf.name}] Training...')

            start_time = timeit.default_timer()
            clf.fit(self.X_train, self.X_target)
            print(f' [{clf.name}] Time elapsed:', timeit.default_timer() - start_time, 's')

    def feature_importance(self):
        print('TODO: Implement feature importance')
        # TODO: Generalize for other classifiers
        # CURRENTLY ONLY FOR RFC
        """
        # print the feature importance for the training
        featureLabels = ["AtlasCoordsX", "AtlasCoordsY", "AtlasCoordsZ", "T1wIntensities", "T2wIntensities", "T1WGradient",
                        "T2wGradient"]
        featureImportancesOrdered = (-clf.feature_importances_).argsort()
        featureLabelsOrdered = [featureLabels[arg] for arg in featureImportancesOrdered]
        featureImportancePrint = ["{}: {:.4f}".format(label, value) for label, value in
                                zip(featureLabelsOrdered, clf.feature_importances_[featureImportancesOrdered])]
        print("Feature importance in descending order:\n", featureImportancePrint)
        """

    def test(self):
        print('-' * 5, 'Testing...')

        self._prepare_output_dir()

        # initialize evaluator
        evaluator = putil.init_evaluator()

        # crawl the training image directories
        crawler = futil.FileSystemDataCrawler(data_test_dir,
                                              LOADING_KEYS,
                                              futil.BrainImageFilePathGenerator(),
                                              futil.DataDirectoryFilter())

        # load images for testing and pre-process
        pre_process_params['training'] = False
        images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

        images_prediction = []
        images_probabilities = []

        for img in images_test:
            print('-' * 10, 'Testing', img.id_)

            start_time = timeit.default_timer()
            predictions = clf.predict(img.feature_matrix[0])
            probabilities = clf.predict_proba(img.feature_matrix[0])
            print(' Time elapsed:', timeit.default_timer() - start_time, 's')

            # convert prediction and probabilities back to SimpleITK images
            image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.uint8),
                                                                            img.image_properties)
            image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)

            # evaluate segmentation without post-processing
            evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

            images_prediction.append(image_prediction)
            images_probabilities.append(image_probabilities)


    def post_process(self):
        # post-process segmentation and evaluate with post-processing
        # post_process_params = {'simple_post': True}
        # images_post_processed = putil.post_process_batch(images_test, images_prediction, images_probabilities,
        #                                                  post_process_params, multi_process=True)
        pass 

    def evaluate(self):
        for i, img in enumerate(images_test):
            # evaluator.evaluate(images_post_processed[i], img.images[structure.BrainImageTypes.GroundTruth],
            #                    img.id_ + '-PP')

            # save results
            sitk.WriteImage(images_prediction[i], os.path.join(result_dir, images_test[i].id_ + '_SEG.mha'), True)
            # sitk.WriteImage(images_post_processed[i], os.path.join(result_dir, images_test[i].id_ + '_SEG-PP.mha'), True)

        # use two writers to report the results
        os.makedirs(result_dir, exist_ok=True)  # generate result directory, if it does not exists
        result_file = os.path.join(result_dir, 'results.csv')
        writer.CSVWriter(result_file).write(evaluator.results)

        print('\nSubject-wise results...')
        writer.ConsoleWriter().write(evaluator.results)

        # report also mean and standard deviation among all subjects
        result_summary_file = os.path.join(result_dir, 'results_summary.csv')
        functions = {'MEAN': np.mean, 'STD': np.std}
        writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator.results)
        print('\nAggregated statistic results...')
        writer.ConsoleStatisticsWriter(functions=functions).write(evaluator.results)

        # clear results such that the evaluator is ready for the next evaluation
        evaluator.clear()

    def _prepare_output_dir(self):
        # create a result directory with timestamp
        t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        result_dir = os.path.join(result_dir, t)
        os.makedirs(result_dir, exist_ok=True)