import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from  dataVisualization import read_data
from classAdaline import Adaline
from dataAnimation import save_animation
from plotVisualization import plot_performance
from normalization import normalization
from Kfold import Kfold, visualize_folds
from defineLearningRate import create_testing_data


data = read_data('../data/winequality-red.csv')

data = data.assign(highQ = pd.Series(data['quality'] > 5))
features = ['volatile acidity', 'alcohol', 'quality', 'highQ']
cleaned_data = data[(data['quality'] > 6) | (data['quality'] < 5)][features]
cleaned_data = cleaned_data.reset_index(drop=True)

cleaned_data['alcohol'] = normalization(cleaned_data['alcohol'])
cleaned_data['volatile acidity'] = normalization(cleaned_data['volatile acidity'])
X = cleaned_data.loc[:,['volatile acidity', 'alcohol']]
Y = cleaned_data['highQ']

def main():
    k = 5
    i = 0
    folds = Kfold(k, cleaned_data, True)
    for test, valid in folds:
        print('Lengths of training and validation sets for %d fold: ( %d, %d ) ' % (i, len(test), len(valid)))
        i += 1

    figure = visualize_folds(folds, ['volatile acidity', 'alcohol'])
    plt.show()
    plt.savefig('../images/folds-plot.png')


if __name__ == "__main__":
    main()
