"""A hello world.

Uses the main libraries to verify the environment installation.
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import tensorflow as tf
from tensorflow.python.platform import app


def main(_):

    print('Welcome to MIALab 2017!')

    # --- TensorFlow

    # Create a Constant op
    # The op is added as a node to the default graph.
    #
    # The value returned by the constructor represents the output
    # of the Constant op.
    hello = tf.constant('Hello, TensorFlow!')

    # Start tf session
    sess = tf.Session()

    # Run the op
    print(sess.run(hello).decode(sys.getdefaultencoding()))

    # --- SimpleITK
    image = sitk.Image(256, 128, 64, sitk.sitkInt16)
    print('Image dimension:', image.GetDimension())
    print('Voxel value before setting:', image.GetPixel(0, 0, 0))
    image.SetPixel(0, 0, 0, 1)
    print('Voxel value after setting:', image.GetPixel(0, 0, 0))

    # --- numpy and matplotlib
    array = np.array([1, 23, 2, 4])
    plt.plot(array)
    plt.ylabel('Some meaningful numbers')
    plt.xlabel('The x-axis')
    plt.title('Wohoo')

    plt.show()

    print('Everything seems to work fine!')

if __name__ == "__main__":
    """The program's entry point."""

    app.run(main=main, argv=[sys.argv[0]])
