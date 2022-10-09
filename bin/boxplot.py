import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os

def main():
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus)
    #  in a boxplot

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')

    base_path = os.path.dirname(os.path.realpath(__file__))
    file = glob.glob(f'{base_path}/mia-result/*/results.csv')[0]
    results = pd.read_csv(file, delimiter=';')
    results.boxplot(column='DICE', by='LABEL')
    plt.savefig(f'{base_path}/mia-result/dice-by-label.png')


if __name__ == '__main__':
    main()
