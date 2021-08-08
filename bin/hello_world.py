"""A hello world.

Uses the main libraries to verify the environment installation.
"""

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pymia.evaluation.evaluator as pymia_eval
import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble


def main():

    print(f'Welcome to MIALab {datetime.now().year}!')

    # --- scikit-learn
    sk_ensemble.RandomForestClassifier(max_depth=2, random_state=0)

    # --- SimpleITK
    image = sitk.Image(256, 128, 64, sitk.sitkInt16)
    print('Image dimension:', image.GetDimension())
    print('Voxel intensity before setting:', image.GetPixel(0, 0, 0))
    image.SetPixel(0, 0, 0, 1)
    print('Voxel intensity after setting:', image.GetPixel(0, 0, 0))

    # --- numpy and matplotlib
    array = np.array([1, 23, 2, 4])
    plt.plot(array)
    plt.ylabel('Some meaningful numbers')
    plt.xlabel('The x-axis')
    plt.title('Wohoo')

    plt.show()

    # --- pymia
    pymia_eval.SegmentationEvaluator([], {})

    print('Everything seems to work fine!')


if __name__ == "__main__":
    """The program's entry point."""

    main()
