import pandas as pd
import matplotlib.pyplot as plt

from  dataVisualization import read_data
from classPerceptron import Perceptron

def normalize_series(series):
    if not isinstance(series, pd.Series):
        raise ValueError('input argument is not an instance of pandas.Series class')
    return (series - series.mean()) / (series.max() - series.min())


def main():
    data = read_data('../data/winequality-red.csv')
    X, Y, samples = dataCleaning(data)
    
if __name__ == "__main__":
    main()

