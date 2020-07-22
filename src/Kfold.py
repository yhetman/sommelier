import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from  dataVisualization import read_data
from classAdaline import Adaline
from dataAnimation import save_animation
from plotVisualization import plot_performance
from normalization import normalization


def shuffle_data(data):
    shuffled = pd.DataFrame()
    while not data.empty:
        i = random.randrange(0, data.shape[0])
        shuffled = shuffled.append(data.iloc[i, :])[data.columns.tolist()]
        data = data.drop(data.index[i])
    return shuffled


def Kfold(k, data, shuffle = True):
    folds = []

    if shuffle :
        data = shuffle_data(data)

    for i in range(k):
        if i < data.shape[0] % k:
            fold_size = data.shape[0] // k + 1
        else:
            data.shape[0] // k
        test_data = data.iloc[i * fold_size : (i + 1) * fold_size, : ]
        train_data = data.iloc[data.index.difference(test_data.index), : ]
        folds.append((train_data, test_data))
    return folds


                                                

