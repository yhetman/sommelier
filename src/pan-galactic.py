import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from  dataVisualization import read_data, plot_scatter_matrix
from classAdaline import Adaline
from dataAnimation import save_animation
from plotVisualization import plot_performance
from normalization import normalization
from Kfold import Kfold, visualize_folds
from defineLearningRate import create_testing_data

import math

data = read_data('../data/pan-galactic.csv')

#good_treshold = 7
#bad_treshold = 4

features = ['wonderflonium', 'fallian marsh gas']
normed_db = data
for feature in features:
        normed_db[feature] = normalization(normed_db[feature])

X, Y = data['wonderflonium'], data['fallian marsh gas']
data = data.assign(metrics = pd.Series(X ** 2 + Y ** 2).pow(1/2))
data = data.assign(azimuth = [math.atan2(x, y) for x, y in zip(X, Y)])

#figure = plot_scatter_matrix(data, good_treshold, bad_treshold, save_plot = True)

def transform(data, features, good_treshold, bad_treshold):
    good = data[(data['quality'] > good_treshold)].copy()
    bad = data[(data['quality'] < bad_treshold)].copy()

    good.loc[:, 'good'] = 1
    bad.loc[:, 'good'] = 0

    dt = pd.concat([good, bad])
    dt.drop(dt.columns.difference(features + ['good', 'quality']), 1, inplace=True)

    return dt

transformed = transform(data, ['metrics', 'azimuth'], 7, 4)
print(transformed)
transformed = transformed.reset_index(drop=True)
