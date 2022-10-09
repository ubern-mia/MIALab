import argparse
import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(result_file: str, result_dir: str):
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus)
    #  in a boxplot

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
    # pass  # pass is just a placeholder if there is no other code
    df = pd.read_csv(os.path.join(result_file), sep=";")
    df.boxplot(by='LABEL', column='DICE', grid=False)
    plt.savefig(os.path.join(result_dir, 'boxplot_dice.png'), format="png")
    plt.show()




if __name__ == '__main__':
    """Plot the metrics as boxplots"""

    script_dir = os.path.dirname(sys.argv[0])
    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')

    parser.add_argument(
        '--result_file',
        type=str,
        default='./mia-result/2022-10-08-19-32-34/results.csv',
        help='Name of the file containing the results.'
    )

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-result/2022-10-08-19-32-34')),
        help='Directory for results.'
    )

    args = parser.parse_args()
    plot(args.result_file, args.result_dir)
