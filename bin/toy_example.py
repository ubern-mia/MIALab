"""A decision forest toy example.

Trains and evaluates a decision forest classifier on a 2-D point cloud.
"""

import argparse
import datetime
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw
from tensorflow.python.platform import app

import mialab.classifier.decision_forest as df


class DataCollection:
    """Represents a collection of data (features) with associated labels (if any)."""

    UNKNOWN_LABEL = -1

    def __init__(self, dimension: int):
        """Initializes a new instance of the DataCollection class.

        Args:
            dimension (int): The dimension of the data (number of features).
        """
        self.dimension = dimension
        self.data = None  # use float32 since TensorFlow does not support float64
        self.labels = None

    def add_data(self, data, label: int=UNKNOWN_LABEL):
        """Adds data to the collection.

        Args:
            data (list of float): The data, e.g. [0.1, 0.2].
            label (int): The data's associated label.

        Raises:
            ValueError: If the data's dimension matches not the DataCollection's dimension.
        """
        if len(data) != self.dimension:
            raise ValueError('Data has not expected dimensionality')

        if self.data is not None:
            self.data = np.vstack([self.data, data]).astype(np.float32, copy=False)
        else:
            self.data = np.array(data, dtype=np.float32)

        if label != DataCollection.UNKNOWN_LABEL and self.labels is not None:
            self.labels = np.append(self.labels, [label]).astype(np.int32, copy=False)
        elif label != DataCollection.UNKNOWN_LABEL:
            self.labels = np.array([label], np.int32)

    def has_labels(self) -> bool:
        """Determines whether the data have labels associated.

        Returns:
            bool: True if the data have labels associated; otherwise, False.
        """
        return self.label_count() != 0

    def label_count(self) -> int:
        """Determines the number of labels.

        Returns:
            int: The number of labels.
        """
        unique_labels = np.unique(self.labels)
        return unique_labels.size


class Reader:
    """Represents a point reader, which reads a list of points from a text file.

    The text file needs to have the following format:
    1 	 231.293210 	 201.938881
    1 	 164.756169 	 162.208593
    2 	 859.625948 	 765.342651
    3 	 839.740553 	 228.076223

    Where the first column is the label y of the point. The second and third columns are x1 and x2,
    i.e. the features one and two (or in other words the (x, y) coordinates of the 2-D point).
    The above example contains four points.
    """

    @staticmethod
    def load(file_path) -> DataCollection:
        """Loads a file that contains 2-dimensional data.

        Returns:
            DataCollection: A data collection with points.
        """

        data = DataCollection(2)
        file = open(file_path, 'r')
        for line in file:
            values = re.split(r'\t+', line)
            data.add_data([float(values[1]), float(values[2])], int(values[0]) - 1)

        return data


class Generator:
    """Represents a point generator.

    The points have an integer spacing and no associated labels.
    """

    @staticmethod
    def get_test_data(grid_size: int):
        """Gets testing data.

        Args:
            grid_size (int): The point cloud grid size.

        Returns:
            np.ndarray: An array of test data.
        """

        rng = np.linspace(0, grid_size - 1, grid_size)
        grid = np.meshgrid(rng, rng)
        return np.append(grid[0].reshape(-1, 1), grid[1].reshape(-1, 1), axis=1).astype(np.float32, copy=False)

    @staticmethod
    def get_test_data_with_label(grid_size: int):
        """Gets testing data.

        Args:
            grid_size (int): The point cloud grid size.

        Returns:
            np.ndarray, np.ndarray: Arrays of test data and labels.
        """
        data = Generator.get_test_data(grid_size)
        labels = np.zeros((data.shape[0], 1)).astype(np.int32)
        return data, labels


class Plotter:
    """Represents a point plotter, which plots a list of points as image."""

    def __init__(self):
        """Initializes a new PointPlotter class.

        The plotter supports up to four class labels.
        """
        self.image = Image.new('RGB', (1000, 1000), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)

        # default label colors (add more colors to support more labels)
        self.label_colors = [(183, 170, 8),
                             (194, 32, 14),
                             (4, 154, 10),
                             (13, 26, 188)]

    def save(self, file_name: str):
        """Saves the plot as PNG image.

        Args:
            file_name (str): The file name.
        """
        if not file_name.lower().endswith('.png'):
            file_name += '.png'
        self.image.save(file_name, 'PNG')

    def plot_points(self, data, labels, radius=3):
        """Plots points on an image.

        Args:
            data (np.ndarray): The data (point coordinates) to plot.
            labels (np.ndarray): The data's associated labels.
            radius (int): The point radius.
        """
        it = np.nditer(labels, flags=['f_index'])
        while not it.finished:
            value = data[it.index]
            x = int(value[0])
            y = int(value[1])
            label = labels[it.index]
            fill = self.label_colors[label]
            self.draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=(0, 0, 0))

            it.iternext()

    def plot_pixels_proba(self, data, probabilities):
        """Plots probabilities on an image.

        Args:
            data (np.ndarray): The data (probability coordinates) to plot.
            probabilities (np.ndarray): The data's associated probabilities.
        """
        it = np.nditer(probabilities, flags=['f_index'])
        for idx in range(probabilities.shape[0]):
            value = data[idx]
            x = int(value[0])
            y = int(value[1])
            probability = probabilities[idx]
            fill = self.get_color(probability)
            self.draw.point((x, y), fill=fill)

            it.iternext()

    def get_color(self, label_probabilities: np.ndarray):
        """Gets the color for a probability.

        Args:
            label_probabilities (np.ndarray): The probabilities.

        Returns:
            (int, int, int): A tuple representing an RGB color code.
        """
        color = np.array([0.0, 0.0, 0.0])

        for i in range(label_probabilities.size):
            weighted_color = np.multiply(np.array(self.label_colors[i]), label_probabilities[i])
            color += weighted_color

        return int(color[0]), int(color[1]), int(color[2])


FLAGS = None  # the program flags


def main(_):
    """Trains a decision forest classifier on a two-dimensional point cloud."""

    # generate model directory (use datetime to ensure that the directory is empty)
    t = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
    model_dir = os.path.join(FLAGS.model_dir, t)
    os.makedirs(model_dir, exist_ok=True)

    # generate result directory
    os.makedirs(FLAGS.result_dir, exist_ok=True)

    # read file with training data
    data = Reader.load(FLAGS.input_file)

    # generate testing data
    test_data = Generator.get_test_data(1000)

    # generate decision forest parameters
    params = df.DecisionForestParameters()
    params.num_classes = data.label_count()
    params.num_features = data.dimension
    params.num_trees = 10
    params.max_nodes = 100  # or params.set_max_nodes(...)
    params.use_training_loss = False
    params.report_feature_importances = True
    params.model_dir = model_dir
    print(params)

    # train the forest
    forest = df.DecisionForest(params)
    print('Decision forest training...')
    forest.train(data.data, data.labels)
    # or use load_estimator to load a model (create a DecisionForestParameters object and set the model_dir)
    # forest.load_estimator()

    # apply the forest to test data
    print('Decision forest testing...')
    probabilities, predictions = forest.predict(test_data)

    # or directly evaluate when labels are known
    # this can be used to see the feature importance
    # eval_data, eval_labels = Generator.get_test_data_with_label(50)
    # results = forest.evaluate(eval_data, eval_labels)
    # for key in sorted(results):
    #     print('%s: %s' % (key, results[key]))

    # plot the result
    print('Plotting...')
    plotter = Plotter()
    plotter.plot_pixels_proba(test_data, np.array(probabilities))
    plotter.plot_points(data.data, data.labels)
    plotter.save(os.path.join(FLAGS.result_dir, 'result_{}.png'.format(t)))


if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='2-dimensional point classification with decision forests')
    parser.add_argument(
        '--model_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, 'toy-example-model')),
        help='Base directory for output models.'
    )

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, 'toy-example-result')),
        help='Directory for results.'
    )

    parser.add_argument(
        '--input_file',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/exp1_n2.txt')),
        help='Input file with 2-dimensional coordinates and corresponding label.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)
