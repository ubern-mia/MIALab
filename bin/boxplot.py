import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # todo: load the "results.csv" file from the mia-results directory
    df = pd.read_csv(r"C:\Users\newto\OneDrive\BioMedicalEngineering\Notes and Class Documents\FS2024\05_Medical Image Analysis Lab\final_project\MIALab_Lukas_Studer\bin\mia-result")
    # todo: read the data into a list
    # Extract the Dice coefficients for each label
    labels = ['white matter', 'gray matter', 'hippocampus', 'amygdala', 'thalamus']

    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus)
    # Create a boxplot for each label
    plt.figure(figsize=(10, 6))
    plt.boxplot([df[label] for label in labels], labels=labels)
    plt.title('Dice Coefficients per Label')
    plt.ylabel('Dice Coefficient')
    plt.xlabel('Label')
    plt.grid(axis='y')
    plt.show()
    #  in a boxplot

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
    pass  # pass is just a placeholder if there is no other code


if __name__ == '__main__':
    main()
