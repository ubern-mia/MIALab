import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import seaborn as sns


def main():
    #
    # load the "results.csv" file from the mia-results directory
    try:
        file_path = "../MIALab_Lukas_Studer/bin/mia-result/2023-11-03-12-27-54/results.csv"
        #df = pd.read_csv(r"results.csv", delimiter=';')
        df = pd.read_csv(file_path, delimiter=';')

    except: #added an exit if directory wrong
        print("File 'results.csv' not found. Please verify the file path.")
        return

    # Check DataFrame columns to understand the column names
    #print(df.columns)

    # Ensure if 'LABEL' column exists in the DataFrame
    #if 'LABEL' in df.columns:
    #    print("Column 'LABEL' exists.")
    #else:
    #    print("Column 'LABEL' does not exist.")

    # read the data into a list
    # Extract the Dice coefficients for each label
    labels = ['WhiteMatter', 'GreyMatter', 'Hippocampus', 'Amygdala', 'Thalamus'] #changes the label names
    filtered_df = df[df['LABEL'].isin(labels)]

    # plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus)
    # Create a boxplot for each label
    plt.figure(figsize=(10, 6))
    plt.boxplot([filtered_df[filtered_df['LABEL'] == label]['DICE'] for label in labels], labels=labels)
    plt.title('Dice Coefficients per Label')
    plt.ylabel('Dice Coefficient')
    plt.xlabel('Label')
    plt.grid(axis='y')
    plt.show()
    #  in a boxplot

    #Comparison between HDRFDST and the DICE
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df['HDRFDST'])
    plt.title('Hausdorf distance Boxplot')

    plt.subplot(1, 2, 2)
    sns.boxplot(y=df['DICE'])
    plt.title('Dice Coefficient Boxplot')

    plt.tight_layout()
    plt.show()


    #Trying to do some correlation stuff:
    metrics = ['HDRFDST', 'DICE']
    # Create a correlation matrix using the selected metrics
    correlation_matrix = df[metrics].corr()

    # Plot the correlation matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Metrics')
    plt.show()

    ##JUST SOME TEST STUFF TO DEBUG
    #print(filtered_df.head())
    #print(df.head())# Print the first few rows of the filtered DataFrame
    #print(filtered_df['DICE'].unique())
    #print(df['DICE'].unique())# Check unique values in the 'DICE' column

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
    #pass  # pass is just a placeholder if there is no other code


if __name__ == '__main__':
    main()
