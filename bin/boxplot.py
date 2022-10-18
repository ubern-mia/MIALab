import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus)
    #  in a boxplot

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
    #pass  # pass is just a placeholder if there is no other code
    print('halllllllllllllllllllllllllo')
    results = pd.read_csv('mia-result/2022-10-12-08-25-55/results.csv', sep=';')
    test = results.boxplot(by='LABEL', column=['DICE'], grid=False)
    test.plot()
    plt.show()

if __name__ == '__main__':
    main()
