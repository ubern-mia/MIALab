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
    """Represents a collection of data points for machine learning."""

    UNKNOWN_LABEL = -1

    def __init__(self, dimension: int):
        """Initializes a new instance of the DataCollection class.

        Args:
            dimension (int): The dimension of the data (number of features).
        """
        self.dimension = dimension
        self.data = None  # use float32 since TensorFlow does not support float64
        self.labels = None

    def add_data(self, data, label=UNKNOWN_LABEL):
        """
        TODO: not efficient (see http://stackoverflow.com/a/32179912)
        :param data:
        :param label:
        :return: None.
        :raises: ValueError
        """
        if len(data) != self.dimension:
            raise ValueError('Data has not expected dimensionality')

        if self.data is not None:
            self.data = np.vstack([self.data, data]).astype(np.float32, copy=False)
        else:
            self.data = np.array(data, dtype=np.float32)

        if label != DataCollection.UNKNOWN_LABEL and self.labels is not None:
            self.labels = np.append(self.labels, [label]).astype(np.uint8, copy=False)
        elif label != DataCollection.UNKNOWN_LABEL:
            self.labels = np.array([label], np.uint8)

    def has_labels(self):
        """
        Determines whether the data have labels associated.
        :return: True if the data have labels associated; otherwise, False.
        """
        return self.label_count() != 0

    def label_count(self):
        unique_labels = np.unique(self.labels)
        return unique_labels.size

class Reader:
    """Represents a point reader, which reads a list of points from a text file."""

    @staticmethod
    def load(file_path) -> DataCollection:
        """Loads a file that contains 2-dimensional data."""

        data = DataCollection(2)
        file = open(file_path, "r")
        for line in file:
            values = re.split(r'\t+', line)
            data.add_data([float(values[1]), float(values[2])], int(values[0]) - 1)

        return data


class Generator:
    """Represents a point generator."""

    @staticmethod
    def get_test_data(grid_size):
        """Gets testing data.

        :return: An array of test data.
        """
        rng = np.linspace(0, grid_size - 1, grid_size)
        grid = np.meshgrid(rng, rng)
        return np.append(grid[0].reshape(-1, 1), grid[1].reshape(-1, 1), axis=1).astype(np.float32, copy=False)


class Plotter:
    """Represents a point plotter, which plots a list of points as image."""

    def __init__(self):
        """Initializes a new PointPlotter class.

        The plotter supports up to four class labels.
        """
        self.image = Image.new("RGB", (1000, 1000), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)

        # default label colors (add more colors to support more labels)
        self.label_colors = [(183, 170, 8),
                             (194, 32, 14),
                             (4, 154, 10),
                             (13, 26, 188)]

    def save(self, file_name):
        """Saves the plot as PNG image.

        :param file_name: The file name.
        :return: None.
        """
        if not file_name.lower().endswith(".png"):
            file_name += ".png"
        self.image.save(file_name, "PNG")

    def plot_points(self, data, labels, radius=3):
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
        it = np.nditer(probabilities, flags=['f_index'])
        for idx in range(probabilities.shape[0]):
            value = data[idx]
            x = int(value[0])
            y = int(value[1])
            probability = probabilities[idx]
            fill = self.get_color(probability)
            self.draw.point((x, y), fill=fill)

            it.iternext()

    def get_color(self, label_probabilities):
        color = np.array([0.0, 0.0, 0.0])

        for i in range(label_probabilities.size):
            weighted_color = np.multiply(np.array(self.label_colors[i]), label_probabilities[i])
            color += weighted_color

        return int(color[0]), int(color[1]), int(color[2])


FLAGS = None  # the program flags


def main(_):
    """Trains a decision forest classifier on a two-dimensional point cloud."""

    # generate model directory (use datetime to ensure that the directory is empty)
    model_dir = os.path.join(FLAGS.model_dir, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    os.makedirs(model_dir, exist_ok=True)

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
    # or use load_estimator to load a model (note to set the params.model_dir)
    # forest.load_estimator()

    # apply the forest to test data
    print("Decision forest testing...")
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
    plotter.save(os.path.join(FLAGS.result_dir, 'result.png'))


if __name__ == "__main__":
    """The program's entry point."""

    parser = argparse.ArgumentParser(description="2-dimensional point classification with decision forests")
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./toy-example-model',
        help='Base directory for output models.'
    )

    parser.add_argument(
        '--result_dir',
        type=str,
        default='./toy-example-result',
        help='Directory for results.'
    )

    parser.add_argument(
        '--input_file',
        type=str,
        default='./data/exp1_n2.txt',
        help='Input file with 2-dimensional coordinates and corresponding label.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)
