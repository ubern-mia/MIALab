"""A decision forest toy example.

Trains and evaluates a decision forest classifier on a 2-D point cloud.
"""

import argparse
import datetime
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sklearn.ensemble as sk_ensemble
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))  # append the MIALab root directory to Python path
# fixes the ModuleNotFoundError when executing main.py in the console after code changes (e.g. git pull)
# somehow pip install does not keep track of packages

def main(savefig: bool, result_dir: str, numtrees: int, treedepth: int):
    """Trains a decision forest classifier on a the iris dataset."""

    # generate result directory
    os.makedirs(result_dir, exist_ok=True)

    # load iris data

    data = make_moons(n_samples=1500, noise=0.23, random_state=None)
    features, labels = data[0], data[1]

    # split into training and  testing data
    feat_train, feat_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.5, random_state=42)

    # initialize the forest
    forest = sk_ensemble.RandomForestClassifier(max_features=feat_train.shape[1],
                                                n_estimators=args.numtrees,
                                                max_depth=args.treedepth)

    # train the forest
    print('Decision forest training...')
    forest.fit(feat_train, labels_train)

    # apply the forest to test data
    print('Decision forest testing...')
    predictions_test = forest.predict(feat_test)
    predictions_train = forest.predict(feat_train)

    # let's have a look at the feature importance
    print('Feature importances:')
    print(forest.feature_importances_)

    # calculate training and testing accuracies
    train_acc = accuracy_score(labels_train, predictions_train)
    test_acc = accuracy_score(labels_test, predictions_test)

    print("Training accuracy: {0:.2%}".format(train_acc))
    print("Testing accuracy: {0:.2%}".format(test_acc))


    # plot the result
    h = .02  # step size in the mesh
    # set font for text in figure
    font = {'family': 'sans-serif',
            'color': 'black',
            'weight': 'normal',
            'size': 16,
            }

    figure = plt.figure(figsize=(10, 10))
    x_min, x_max = features[:, 0].min() - .5, features[:, 0].max() + .5
    y_min, y_max = features[:, 1].min() - .5, features[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    #ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

    # Plot the training points
    plt.scatter(feat_train[:, 0], feat_train[:, 1], c=labels_train, cmap=cm_bright,
               edgecolors='k')
    # and testing points
    plt.scatter(feat_test[:, 0], feat_test[:, 1], c=labels_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Random Forest Classification Exercise")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    prob_boundary = forest.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    prob_boundary = prob_boundary.reshape(xx.shape)
    plt.contourf(xx, yy, prob_boundary, cmap=cm, alpha=.8)

    # add model information
    plt.text(x_min+0.2, y_max-0.2, "Number of trees: {0:d}".format(numtrees), fontdict=font)
    plt.text(x_min+0.2, y_max-0.3, "Max. tree depth: {0:d}".format(treedepth), fontdict=font)

    # add accuracy information to plot
    plt.text(x_max-2, y_max-0.2, "Training accuracy: {0:.2%}".format(train_acc), fontdict=font)
    plt.text(x_max-2, y_max-0.3, "Testing accuracy: {0:.2%}".format(test_acc), fontdict=font)

    plt.show()

    # save figure if flag is set
    if savefig:
        t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        figure.savefig(os.path.join(result_dir, 'rfplot_{}.png'.format(t)))
        print('Plot saved as ' + os.path.join(result_dir, 'rfplot_{}.png'.format(t)))



if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='2-dimensional point classification with decision forests')

    parser.add_argument(
        '--savefig',
        type=bool,
        default=True,
        help='Set to True to save plot to result_dir.'
    )

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, 'randomforest_plots')),
        help='Directory for results.'
    )

    parser.add_argument(
        '--numtrees',
        type=int,
        default=1,
        help='Number of trees in the random forest classifier.'
    )

    parser.add_argument(
        '--treedepth',
        type=int,
        default=1,
        help='Maximum depth of the trees in the random forest classifier.'
    )

    args = parser.parse_args()
    main(args.savefig, args.result_dir, args.numtrees, args.treedepth)
