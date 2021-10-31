import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def main():

    # Root directory
    dir = './mia-result/'

    # Get all the results folders
    resultsPaths = os.listdir(dir)

    # Load the data frame (of the latest results folder)
    results = pd.read_csv(dir + resultsPaths[-1] + '/results.csv', sep=';')

    # Plot the data
    results.boxplot(column='DICE', by='LABEL')
    plt.show()

    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus) in a boxplot

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
    pass  # pass is just a placeholder if there is no other code


if __name__ == '__main__':
    main()
