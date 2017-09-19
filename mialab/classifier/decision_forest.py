"""The decision forest module holds implementations of decision forests (random forests)."""
import numpy as np
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.tensor_forest.client import eval_metrics
from tensorflow.contrib.tensor_forest.client import random_forest
from tensorflow.contrib.tensor_forest.python import tensor_forest


class DecisionForestParameters:
    """Represents a set of decision forest parameters."""

    def __init__(self):
        """Initializes a new instance of the DecisionForestParameters class."""

        self.num_classes = 2  # number of classes (labels)
        self.num_features = 2  # number of features
        self.num_trees = 30  # number of trees in the forest
        self.max_nodes = 20  # max total nodes in a single tree
        self.batch_size = 128  # number of examples in a training batch
        self.use_training_loss = False  # If True, use training loss as termination criteria
        self.report_feature_importances = False  # If True, report feature importances (only in evaluation() function)
        self.inference_tree_paths = False  # If True, inference tree paths (needs TensorFlow > 1.3)
        self.model_dir = str  # base directory for output models

    def set_max_nodes(self, max_tree_depth: int):
        """Sets the maximum nodes according to the maximum tree depth.

        The maximum number of nodes in a binary tree of depth :math:`k` is given
        by :math:`1 + 2 + 4 + 8 + ... + 2^k = \\frac{2^{k+1} - 1}{2 - 1} = 2^{k + 1} -1`.

        Note that this assumes a fully built tree.

        Args:
            max_tree_depth (int): The maximum tree depth.
        """
        self.max_nodes = 2 ** (max_tree_depth + 1) - 1

    def __str__(self):
        """Gets a printable string representation of the class.

        Returns:
            str: String representation.
        """
        return 'DecisionForestParameters:\n' \
               ' num_classes:                {self.num_classes}\n' \
               ' num_features:               {self.num_features}\n' \
               ' num_trees:                  {self.num_trees}\n' \
               ' max_nodes:                  {self.max_nodes}\n' \
               ' batch_size:                 {self.batch_size}\n' \
               ' use_training_loss:          {self.use_training_loss}\n' \
               ' report_feature_importances: {self.report_feature_importances}\n' \
               ' model_dir:                  {self.model_dir}\n' \
            .format(self=self)


class DecisionForest:
    """Represents a decision forest classifier."""

    def __init__(self, parameters: DecisionForestParameters=DecisionForestParameters()):
        """Initializes an new instance of the DecisionForest class.

        Args:
            parameters (DecisionForestParameters): The parameters.
        """

        self.parameters = None  # tensor_forest.ForestHParams
        self.estimator = None  # estimator.SKCompat
        self.use_training_loss = False
        self.batch_size = 0
        self.use_training_loss = False
        self.report_feature_importances = False
        self.model_dir = ''
        self.set_parameters(parameters)

    def train(self, data: np.ndarray, labels: np.ndarray):
        """Trains the decision forest classifier.

        Args:
            data (np.ndarray): The training data.
            labels (np.ndarray): The labels of the training data.
        """

        # build the estimator
        if self.use_training_loss:
            graph_builder_class = tensor_forest.TrainingLossForest
        else:
            graph_builder_class = tensor_forest.RandomForestGraphs

        self.estimator = estimator.SKCompat(
            random_forest.TensorForestEstimator(
                self.parameters,
                graph_builder_class=graph_builder_class,
                model_dir=self.model_dir,
                report_feature_importances=self.report_feature_importances
        ))

        self.estimator.fit(x=data, y=labels, batch_size=self.batch_size)

    def predict(self, data: np.ndarray):
        """Predicts the labels of the data.

        Args:
            data (np.ndarray): The data to predict.

        Returns:
            (probabilities, predictions): The probabilities and the labels of the prediction.
        """
        if self.estimator is None:
            raise ValueError('Estimator not set')

        result = self.estimator.predict(x=data)
        probabilities = result[eval_metrics.INFERENCE_PROB_NAME]
        predictions = result[eval_metrics.INFERENCE_PRED_NAME]

        return probabilities, predictions

    def evaluate(self, data: np.ndarray, labels: np.ndarray):
        """Predicts and directly evaluates the results.

        Examples:
            To evaluate the prediction of the decision forest use:

            >>> results = forest.evaluate(data, labels)
            >>> for key in sorted(results):
            >>>     print('%s: %s' % (key, results[key]))

        Args:
            data (np.ndarray): The data to predict.
            labels (np.ndarray): A numpy array where labels[i] returns the label of observation i.
                E.g. labels.shape = (100, 1), where number of observations=100 with one label.
        Returns:
            dict: A dict of evaluation metrics.
        """

        if self.estimator is None:
            raise ValueError('Estimator not set')

        metrics = {
            'accuracy': metric_spec.MetricSpec(
                eval_metrics.get_metric('accuracy'),
                prediction_key=eval_metrics.get_prediction_key('accuracy')
            )
        }

        if self.report_feature_importances:
            metrics['feature_importance'] = metric_spec.MetricSpec(
                lambda x: x,
                prediction_key=eval_metrics.FEATURE_IMPORTANCE_NAME
            )

        results = self.estimator.score(x=data, y=labels, batch_size=self.batch_size,
                                       metrics=metrics)

        return results

    def set_parameters(self, parameters: DecisionForestParameters):
        """Sets the decision forest parameters.

        Args:
            parameters (DecisionForestParameters): The parameters.
        """
        self.parameters = tensor_forest.ForestHParams(
            num_classes=parameters.num_classes,
            num_features=parameters.num_features,
            num_trees=parameters.num_trees,
            max_nodes=parameters.max_nodes,
            inference_tree_paths=parameters.inference_tree_paths
        ).fill()

        self.batch_size = parameters.batch_size
        self.use_training_loss = parameters.use_training_loss
        self.report_feature_importances = parameters.report_feature_importances
        self.model_dir = parameters.model_dir

    def load_estimator(self):
        """Loads a decision forest estimator from a pre-built model.

        The model_dir in the DecisionForestParameters class needs to be set appropriately.
        """
        self.estimator = estimator.SKCompat(random_forest.TensorForestEstimator(self.parameters,
                                                                                model_dir=self.model_dir))
