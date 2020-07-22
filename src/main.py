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
features = ['pH', 'alcohol', 'sulphates', 'fixed acidity', 'volatile acidity', 'highQ', 'quality']

cleaned_data = data[(data['quality'] > 6) | (data['quality'] < 5)][features]

cleaned_data = cleaned_data.reset_index(drop=True)

to_norme = ['pH', 'alcohol', 'sulphates', 'fixed acidity', 'volatile acidity']

for feature in to_norme:
        cleaned_data[feature] = normalization(cleaned_data[feature])


def cross_validation(folds, features, lr=0.05, epochs=500, mode='batch', verbose=False):
    sum_accuracy = 0

    model = Adaline(lr=lr)
    for i, fold in enumerate(folds):
        X_train = fold[0][features]
        Y_train = fold[0]['highQ']
        
        stats = model.fit(X_train.values, Y_train, epochs, mode, verbose)
        
        X_test = fold[1][features]
        Y_test = fold[1]['highQ']
        accuracy = model.evaluation(X_test.values, Y_test)
        sum_accuracy += accuracy
    
    print('=================================')
    print('Mean model accuracy: %.6f' %(sum_accuracy / len(folds)))
    print('=================================')


def main():
    k = 5
    i = 0
    folds = Kfold(k, cleaned_data, True)
    cross_validation(folds, ['alcohol', 'volatile acidity', 'pH'])
    cross_validation(folds, ['alcohol', 'volatile acidity', 'pH', 'sulphates'])
    cross_validation(folds, ['alcohol', 'volatile acidity', 'sulphates', 'fixed acidity'])
    cross_validation(folds, ['alcohol', 'volatile acidity', 'pH', 'fixed acidity'])


if __name__ == "__main__":
    main()
